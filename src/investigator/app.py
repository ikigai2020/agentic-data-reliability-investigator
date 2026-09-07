"""CLI entrypoint and end-to-end runner (§20).

Usage:
    uv run investigator investigate --scenario diagnosed_transformation_filter
    uv run investigator investigate --scenario diagnosed_transformation_filter --checkpoint
    uv run investigator investigate --scenario inconclusive_missing_evidence --thread-id inc-1
    uv run investigator list-scenarios
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .agents.reasoning import build_reasoning
from .config import PROVIDERS, AppConfig, load_config, provider_api_key
from .graph.journal import build_journal
from .graph.parent_graph import build_parent_graph
from .mcp_client.client import InvestigatorMCPClient
from .observability import tracing
from .persistence import sqlite_checkpointer, thread_config


def _load_raw_alert(cfg: AppConfig, scenario_id: str) -> dict[str, Any]:
    alert_path = Path(cfg.scenarios_dir) / scenario_id / "alert.json"
    if not alert_path.exists():
        raise FileNotFoundError(f"no alert.json for scenario '{scenario_id}' at {alert_path}")
    return json.loads(alert_path.read_text(encoding="utf-8"))


def _load_meta(cfg: AppConfig, scenario_id: str) -> dict[str, Any]:
    meta_path = Path(cfg.scenarios_dir) / scenario_id / "meta.json"
    if not meta_path.exists():
        return {}
    return dict(json.loads(meta_path.read_text(encoding="utf-8")))


def _load_risk_policy_inputs(cfg: AppConfig, scenario_id: str) -> dict[str, Any]:
    """Load a scenario's optional ``risk`` block from meta.json (FR-1105 policy inputs).

    These are deterministic policy inputs (data sensitivity / impact flags), supplied by
    the controller — not operational evidence and not read by agents.
    """
    return dict(_load_meta(cfg, scenario_id).get("risk", {}))


async def run_investigation(
    scenario_id: str,
    cfg: AppConfig | None = None,
    *,
    risk_overrides: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
    reasoning: Any | None = None,
) -> dict[str, Any]:
    """Run one full investigation for a scenario and return the final graph state.

    Pass ``checkpointer`` (and a stable ``thread_id``) to make the run resumable
    (FR-804); re-invoking the same thread replays from the last checkpoint instead of
    starting over, and the state reducers keep evidence and budgets from doubling
    (FR-205).

    ``reasoning`` injects a prebuilt engine; when omitted one is built from config and
    closed on the way out.
    """
    cfg = cfg or load_config()
    raw_alert = _load_raw_alert(cfg, scenario_id)
    risk_policy_inputs = {**_load_risk_policy_inputs(cfg, scenario_id), **(risk_overrides or {})}
    graph = build_parent_graph(checkpointer=checkpointer)

    # One engine per investigation: an LLM-backed engine then reuses a single HTTP client
    # and a single usage ledger across every node and every parallel specialist.
    engine = reasoning or build_reasoning(cfg)
    owns_engine = reasoning is None

    configurable: dict[str, Any] = {"client": None, "app_config": cfg, "reasoning": engine}
    if checkpointer is not None:
        configurable.update(thread_config(thread_id or f"thread_{scenario_id}"))

    # M5.2: switched on, this names and tags the run so the trace is attributable and
    # sliceable per component (FR-1206, FR-1204). Off, it costs one dictionary.
    tracing.activate(cfg)
    run_config: dict[str, Any] = {"configurable": configurable}
    if tracing.is_enabled(cfg):
        run_config.update(tracing.run_config(cfg, scenario_id=scenario_id))

    try:
        async with InvestigatorMCPClient(scenario_id) as client:
            configurable["client"] = client
            with tracing.collect(cfg) as collected:
                final_state = await graph.ainvoke(
                    {
                        "raw_alert": raw_alert,
                        "scenario_id": scenario_id,
                        "risk_policy_inputs": risk_policy_inputs,
                    },
                    config=run_config,
                )
    finally:
        aclose = getattr(engine, "aclose", None)
        if owns_engine and aclose is not None:
            await aclose()

    write_journal(final_state, cfg, trace_link=collected.link)
    return final_state


def write_journal(
    state: dict[str, Any], cfg: AppConfig, *, trace_link: Any = None
) -> Path | None:
    """Persist the run journal (M5.1).

    Written here rather than inside the graph so that ``persist_result`` — the stage that
    writes the report — appears in its own journal. A node cannot record the write it is
    still performing.

    ``trace_link`` carries the LangSmith run this investigation produced, when it was
    traced at all (M5.2) — a trace nobody can find from the run is not much of a trace.
    """
    from .evaluation.versions import fingerprint

    if not state.get("journal"):
        return None
    journal = build_journal(
        state,
        versions=fingerprint(cfg).as_dict(),
        tracing=trace_link.as_dict() if trace_link is not None else None,
    )
    cfg.journal_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.journal_dir / f"{journal.investigation_id}.json"
    path.write_text(journal.model_dump_json(indent=2), encoding="utf-8")
    return path


def _print_summary(state: dict[str, Any]) -> None:
    report = state.get("report") or {}
    print("=" * 72)
    print(f"Investigation : {state.get('investigation_id')}")
    print(f"Scenario      : {state.get('scenario_id')}")
    print(f"Verification  : {state.get('verification_status')}")
    print(f"Outcome       : {state.get('outcome')}")
    print(f"Confidence    : {state.get('confidence')}")
    print(f"Risk tier     : {state.get('risk_tier')}")
    print(f"Release status: {state.get('release_status')}")
    print(f"Blocked calls : {state.get('blocked_attempts')}")
    print(f"Rounds        : {state.get('rounds_used')}")
    fallbacks = state.get("reasoning_fallbacks") or []
    if fallbacks:
        print(f"LLM fallbacks : {len(fallbacks)} (see report.guardrail_and_budget_events)")
    print(f"Stop reason   : {state.get('stop_reason')}")
    lead = report.get("leading_hypothesis")
    if lead:
        print(f"Leading cause : {lead['category']} — {lead['statement']}")
    influence = state.get("retrieval_influence")
    if influence is not None:
        print(f"Retrieval     : {influence.kind} (accepted={influence.accepted_doc_ids}, "
              f"rejected={influence.rejected_doc_ids})")
        if influence.kind == "reordered":
            print(f"  order       : {influence.order_before} -> {influence.order_after}")
    print(f"Supporting ev : {report.get('supporting_evidence_ids')}")
    tools = report.get("tools", {})
    print(f"Tools called  : {tools.get('called')}")
    if tools.get("unavailable"):
        print(f"Tools failed  : {tools.get('unavailable')}")
    checks = report.get("guardrail_and_budget_events", {}).get("diagnosis_checks")
    print(f"Diagnosis chk : {checks}")
    if state.get("escalation_package"):
        print("-" * 72)
        print("HUMAN ESCALATION PACKAGE:")
        pkg = state["escalation_package"]
        print(f"  stop reason : {pkg.get('stop_reason')}")
        print(f"  next checks : {pkg.get('recommended_next_checks')}")
        print(f"  unavailable : {pkg.get('unavailable_sources')}")
    print("=" * 72)


def _apply_engine_overrides(args: argparse.Namespace) -> None:
    """CLI flags win over config for this process only (read back by ``load_config``)."""
    if args.engine:
        os.environ["INVESTIGATOR_REASONING_ENGINE"] = args.engine
    if args.provider:
        os.environ["INVESTIGATOR_LLM_PROVIDER"] = args.provider
        # Clear any inherited model so the new provider's default applies.
        os.environ.pop("INVESTIGATOR_LLM_MODEL", None)
        os.environ.pop("OPENROUTER_MODEL", None)
        os.environ.setdefault("INVESTIGATOR_REASONING_ENGINE", "llm")
    if args.model:
        os.environ["INVESTIGATOR_LLM_MODEL"] = args.model
        os.environ.setdefault("INVESTIGATOR_REASONING_ENGINE", "llm")
    if args.engine or args.model or args.provider:
        load_config.cache_clear()


async def _investigate(args: argparse.Namespace) -> dict[str, Any]:
    _apply_engine_overrides(args)
    cfg = load_config()
    if not (args.checkpoint or args.thread_id):
        return await run_investigation(args.scenario, cfg)
    async with sqlite_checkpointer(cfg.checkpoint_path) as saver:
        return await run_investigation(
            args.scenario, cfg, checkpointer=saver, thread_id=args.thread_id
        )


def _cmd_investigate(args: argparse.Namespace) -> int:
    state = asyncio.run(_investigate(args))
    if args.json:
        print(json.dumps(state.get("report"), indent=2, default=str))
    else:
        _print_summary(state)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """List models the configured provider will serve this key — queried, not hard-coded."""
    from dataclasses import replace as _replace

    from .agents.llm import list_models

    cfg = load_config()
    reasoning = cfg.reasoning
    if args.provider:
        reasoning = _replace(reasoning, provider=args.provider, model="", base_url="")

    try:
        models = asyncio.run(
            list_models(reasoning, provider_api_key(reasoning), free_only=args.free)
        )
    except Exception as exc:  # noqa: BLE001 - network failure is a normal CLI outcome
        print(f"could not reach {reasoning.resolved_base_url}: {exc}")
        return 1
    if not models:
        print("(no models reported)")
        return 1
    print(f"# {reasoning.provider} ({reasoning.resolved_base_url})")
    for model_id in models:
        marker = " <- configured" if model_id == reasoning.resolved_model else ""
        print(f"{model_id}{marker}")
    return 0


def _load_report(cfg: AppConfig, investigation_id: str) -> dict[str, Any]:
    path = Path(cfg.outputs_dir) / f"{investigation_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no persisted run for '{investigation_id}' at {path}")
    return json.loads(path.read_text(encoding="utf-8")).get("report", {})


def _cmd_review(args: argparse.Namespace) -> int:
    """Record a human disposition for an escalated investigation (FR-1108)."""
    from .governance import ReviewStore

    cfg = load_config()
    store = ReviewStore(cfg.reviews_dir)
    decision = store.record(
        args.investigation,
        reviewer_id=args.reviewer,
        reviewer_role=args.role,
        decision=args.decision,
        accepted_claims=args.accept or [],
        rejected_claims=args.reject or [],
        override_reason=args.override_reason,
        requested_next_check=args.next_check,
        confirmed_root_cause=args.root_cause,
        remediation_outcome=args.remediation,
        notes=args.notes or "",
    )
    print(f"recorded {decision.review_id}: {decision.decision} by {decision.reviewer_role}")
    if decision.is_promotable:
        print(f"promotable — run: investigator promote --investigation {args.investigation}")
    return 0


def _cmd_reviews(args: argparse.Namespace) -> int:
    """List review records and the FR-1305 disposition-completeness metric."""
    from .governance import ReviewStore

    store = ReviewStore(load_config().reviews_dir)
    records = store.all()
    if not records:
        print("(no review records)")
        return 0
    for record in records:
        mark = "open" if not record.is_complete else record.decision
        who = record.reviewer_role or "-"
        print(f"{record.investigation_id}  {mark:<10} {who}")
    stats = store.disposition_stats()
    print(
        f"\nanswered: {stats.complete}/{stats.total} ({stats.queue_drain_rate:.0%}), "
        f"{stats.pending} open | record completeness: {stats.record_completeness:.0%}"
    )
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    """Promote a human-confirmed investigation into the retrieval corpus (FR-1109)."""
    from .governance import PromotionRefused, ReviewStore, promote

    cfg = load_config()
    store = ReviewStore(cfg.reviews_dir)
    try:
        report = _load_report(cfg, args.investigation)
        result = promote(report, store.get(args.investigation), incidents_dir=cfg.incidents_dir)
    except (PromotionRefused, FileNotFoundError) as exc:
        print(f"refused: {exc}")
        return 1
    verb = "created" if result.created else "updated"
    print(f"{verb} {result.path.name} — confirmed {result.document.category}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the labeled scenario set and report metrics + acceptance gates (§18)."""
    _apply_engine_overrides(args)
    from .evaluation import evaluate_scenarios

    cfg = load_config()
    report = asyncio.run(
        evaluate_scenarios(cfg, scenario_ids=args.scenario or None, repeats=args.repeat)
    )
    payload = report.as_dict()

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_evaluation(payload)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")

    return 0 if report.acceptance.passed else 1


def _print_evaluation(payload: dict[str, Any]) -> None:
    versions = payload["versions"]
    print("=" * 72)
    print(f"Evaluation — {payload['sample_size']} scenario(s)")
    print(
        f"engine={versions['engine']} model={versions['model'] or '-'} "
        f"prompts={versions['prompts_digest']} corpus={versions['corpus_digest']} "
        f"policy={versions['policy_digest']}"
    )
    print("-" * 72)
    for row in payload["matrix"]:
        mark = "ok " if row["correct"] else "ERR"
        errors = f"  {', '.join(row['errors'])}" if row["errors"] else ""
        print(
            f"[{mark}] {row['scenario_id']:<38} {str(row['outcome']):<14}"
            f"{row['confidence'] or '-':<12}{errors}"
        )
    print("-" * 72)
    for name, value in payload["metrics"].items():
        note = ""
        if name == "cost_per_investigation_usd" and payload.get("unpriced_models"):
            note = f"  (unpriced: {', '.join(payload['unpriced_models'])})"
        elif name == "cost_per_investigation_usd" and not payload["metrics"].get("mean_tokens"):
            note = "  (no model calls)"
        print(f"  {name:<42} {value:.3f}{note}")
    print(f"  {'weighted_error (severity-adjusted)':<42} {payload['weighted_error']:.3f}")

    if payload["confidence_bands"]:
        print("-" * 72)
        print("  confidence bands (FR-1308):")
        for band, stats in payload["confidence_bands"].items():
            print(
                f"    {band:<16} n={int(stats['sample_size'])}  "
                f"accuracy={stats['accuracy']:.3f}  "
                f"false-confident={int(stats['false_confident_diagnoses'])}"
            )

    if payload.get("stability"):
        print("-" * 72)
        print("  stability across repeats (FR-1309):")
        for metric, stats in payload["stability"].items():
            print(
                f"    {metric:<40} mean={stats['mean']:.3f} "
                f"[{stats['min']:.3f}–{stats['max']:.3f}] over {int(stats['runs'])} run(s)"
            )

    print("-" * 72)
    acceptance = payload["acceptance"]
    for gate in acceptance["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{mark}] {gate['name']:<30} {gate['metric']}={gate['value']:g}")
    if acceptance["unmeasured"]:
        print(f"  [SKIP] unmeasured: {', '.join(acceptance['unmeasured'])}")
    print(f"\nFR-1305 acceptance: {'PASSED' if acceptance['passed'] else 'FAILED'}")
    print("=" * 72)


def _cmd_backlog(args: argparse.Namespace) -> int:
    """List or add closed-loop improvement backlog items (FR-1207)."""
    from .governance import BacklogRefused, BacklogStore

    store = BacklogStore(load_config().backlog_dir)
    if args.title:
        try:
            item = store.propose(
                title=args.title,
                change_kind=args.kind,
                evidence_source=args.source,
                failure_evidence=args.evidence or [],
                regression_test=args.test or "",
                investigation_id=args.investigation,
                review_id=args.review,
                rationale=args.rationale or "",
            )
        except BacklogRefused as exc:
            print(f"refused: {exc}")
            return 1
        print(f"recorded {item.item_id}: {item.title}")
        return 0

    items = store.all()
    if not items:
        print("(backlog empty)")
        return 0
    for item in items:
        print(f"{item.item_id}  {item.status:<10} {item.change_kind:<10} {item.title}")
        print(f"    evidence: {'; '.join(item.failure_evidence)}")
        print(f"    test:     {item.regression_test}")
    return 0


def _cmd_journal(args: argparse.Namespace) -> int:
    """Replay how an investigation moved, stage by stage (M5.1).

    The report answers *what was concluded*; this answers *how it got there* — which is
    the question a reviewer actually asks when a verdict looks wrong.
    """
    from .schemas.journal import RunJournal

    cfg = load_config()
    if args.list or not args.investigation:
        journals = sorted(cfg.journal_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not journals:
            print("(no run journals — run an investigation first)")
            return 1
        for path in journals[-args.limit :]:
            journal = RunJournal.model_validate_json(path.read_text(encoding="utf-8"))
            print(
                f"{journal.investigation_id}  {journal.scenario_id or '-':<38}"
                f"{journal.outcome or '-':<15}{len(journal.stages)} stages"
            )
        return 0

    path = cfg.journal_dir / f"{args.investigation}.json"
    if not path.exists():
        print(f"no journal for '{args.investigation}' at {path}")
        return 1
    journal = RunJournal.model_validate_json(path.read_text(encoding="utf-8"))

    if args.json:
        print(journal.model_dump_json(indent=2))
        return 0

    print("=" * 78)
    print(f"Journal       : {journal.investigation_id}  ({journal.scenario_id})")
    print(f"Outcome       : {journal.outcome} / {journal.release_status}")
    print(
        f"Stages        : {len(journal.stages)} across {len(journal.rounds())} round(s), "
        f"{journal.total_duration_ms} ms total"
    )
    if journal.llm_calls:
        tokens = sum(call.get("total_tokens", 0) for call in journal.llm_calls)
        print(f"Model calls   : {len(journal.llm_calls)} ({tokens} tokens)")
    if journal.tracing:
        trace = journal.tracing
        print(
            f"LangSmith     : {trace.get('url') or trace.get('run_id')} "
            f"(project {trace.get('project')})"
        )
    versions = journal.versions
    print(
        f"Versions      : engine={versions.get('engine')} model={versions.get('model') or '-'} "
        f"policy={versions.get('policy_digest')}"
    )
    stages = journal.for_round(args.round) if args.round is not None else journal.stages
    for stage in stages:
        print("-" * 78)
        print(
            f"[{stage.sequence:>3}] r{stage.round_number} {stage.node} "
            f"({stage.duration_ms} ms)"
        )
        print(f"      {stage.summary}")
        if args.delta:
            for key, value in stage.delta.items():
                print(f"      · {key}: {json.dumps(value, default=str)[:160]}")
        for call in stage.llm_calls:
            print(
                f"      · llm {call.get('purpose')} {call.get('model')} "
                f"{call.get('latency_ms')}ms {call.get('total_tokens')}tok"
            )
    print("=" * 78)
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Launch the investigation UI (M5.3).

    Streamlit owns its own process and argv, so this hands over rather than importing it:
    the UI is an optional extra, and a missing one should read as a missing extra, not as
    an import error halfway through a demo.
    """
    import subprocess
    import sys
    from pathlib import Path as _Path

    app = _Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.port", str(args.port)]
    if args.headless:
        command += ["--server.headless", "true"]
    try:
        return subprocess.call(command)
    except FileNotFoundError:
        print("streamlit is not installed. Install the UI extra:  uv sync --extra ui")
        return 1


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    cfg = load_config()
    scenarios_dir = Path(cfg.scenarios_dir)
    if not scenarios_dir.exists():
        print("(no scenarios directory)")
        return 1
    for path in sorted(scenarios_dir.iterdir()):
        if path.is_dir() and (path / "alert.json").exists():
            print(path.name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="investigator", description="Data Reliability Investigator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("investigate", help="run an investigation for a scenario")
    p_inv.add_argument(
        "--scenario", required=True, help="scenario id (directory under data/scenarios)"
    )
    p_inv.add_argument("--json", action="store_true", help="print the full report as JSON")
    p_inv.add_argument(
        "--checkpoint",
        action="store_true",
        help="persist checkpoints to SQLite so the run can resume (FR-804)",
    )
    p_inv.add_argument(
        "--thread-id",
        default=None,
        help="resumable thread id; re-run with the same id to resume (implies --checkpoint)",
    )
    p_inv.set_defaults(func=_cmd_investigate)

    p_inv.add_argument(
        "--engine",
        choices=["deterministic", "llm"],
        default=None,
        help="reasoning engine for this run (default: config `reasoning.engine`)",
    )
    p_inv.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=None,
        help="LLM provider for this run (implies --engine llm)",
    )
    p_inv.add_argument(
        "--model",
        default=None,
        help="model id for the provider, e.g. 'gpt-4o-mini'. Implies --engine llm.",
    )

    p_rev = sub.add_parser("review", help="record a human decision on an escalation (FR-1108)")
    p_rev.add_argument("--investigation", required=True)
    p_rev.add_argument("--reviewer", required=True, help="reviewer identity")
    p_rev.add_argument("--role", required=True, help="reviewer role, e.g. 'data-platform-oncall'")
    p_rev.add_argument(
        "--decision", required=True, choices=["confirmed", "rejected", "modified", "pending"]
    )
    p_rev.add_argument("--root-cause", default=None, help="confirmed root-cause category")
    p_rev.add_argument("--accept", action="append", help="accepted claim (repeatable)")
    p_rev.add_argument("--reject", action="append", help="rejected claim (repeatable)")
    p_rev.add_argument("--override-reason", default=None)
    p_rev.add_argument("--next-check", default=None)
    p_rev.add_argument("--remediation", default=None)
    p_rev.add_argument("--notes", default=None)
    p_rev.set_defaults(func=_cmd_review)

    p_revs = sub.add_parser("reviews", help="list review records and disposition completeness")
    p_revs.set_defaults(func=_cmd_reviews)

    p_prom = sub.add_parser(
        "promote", help="promote a confirmed investigation into the corpus (FR-1109)"
    )
    p_prom.add_argument("--investigation", required=True)
    p_prom.set_defaults(func=_cmd_promote)

    p_eval = sub.add_parser("evaluate", help="run the labeled scenario set and score it (§18)")
    p_eval.add_argument("--scenario", action="append", help="limit to a scenario (repeatable)")
    p_eval.add_argument("--repeat", type=int, default=1, help="repeated runs (FR-1309)")
    p_eval.add_argument("--json", action="store_true", help="print the full report as JSON")
    p_eval.add_argument("--out", default=None, help="also write the report to this path")
    p_eval.add_argument("--engine", choices=["deterministic", "llm"], default=None)
    p_eval.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    p_eval.add_argument("--model", default=None)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_back = sub.add_parser("backlog", help="closed-loop improvement backlog (FR-1207)")
    p_back.add_argument("--title", default=None, help="add an item with this title")
    p_back.add_argument(
        "--kind",
        default="policy",
        choices=["prompt", "routing", "guardrail", "tool", "policy", "corpus"],
    )
    p_back.add_argument(
        "--source",
        default="evaluation",
        choices=["human_review", "safety_event", "monitoring_alert", "evaluation",
                 "production_trace"],
    )
    p_back.add_argument("--evidence", action="append", help="observed failure evidence")
    p_back.add_argument("--test", default=None, help="regression test that would catch it")
    p_back.add_argument("--investigation", default=None)
    p_back.add_argument("--review", default=None, help="human review id (required for traces)")
    p_back.add_argument("--rationale", default=None)
    p_back.set_defaults(func=_cmd_backlog)

    p_journal = sub.add_parser(
        "journal", help="replay how an investigation moved, stage by stage (M5.1)"
    )
    p_journal.add_argument("--investigation", default=None, help="investigation id")
    p_journal.add_argument("--list", action="store_true", help="list recent run journals")
    p_journal.add_argument("--limit", type=int, default=15, help="how many to list")
    p_journal.add_argument("--round", type=int, default=None, help="only this round")
    p_journal.add_argument("--delta", action="store_true", help="show each stage's state delta")
    p_journal.add_argument("--json", action="store_true", help="print the raw journal")
    p_journal.set_defaults(func=_cmd_journal)

    p_ui = sub.add_parser("ui", help="launch the investigation UI (M5.3; needs the ui extra)")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.add_argument("--headless", action="store_true", help="do not open a browser")
    p_ui.set_defaults(func=_cmd_ui)

    p_ls = sub.add_parser("list-scenarios", help="list available scenarios")
    p_ls.set_defaults(func=_cmd_list_scenarios)

    p_models = sub.add_parser("models", help="list models the configured provider serves")
    p_models.add_argument(
        "--provider", choices=sorted(PROVIDERS), default=None, help="override the provider"
    )
    p_models.add_argument(
        "--free", action="store_true", help="only zero-cost models (OpenRouter)"
    )
    p_models.set_defaults(func=_cmd_models)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
