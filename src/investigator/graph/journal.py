"""Run-journal capture (M5.1).

Every parent-graph node is wrapped once, at graph-assembly time, rather than each node
recording itself. That keeps the capture in one place and means a node added later is
journaled by default instead of by remembering to.

Two rules shape the compaction below:

**A stage records what changed, not everything that exists.** The delta is built from the
node's own returned update, so a key nobody wrote never appears. Full evidence payloads
are summarised to ids and verdicts — the report already holds them, and repeating them
per node would make the file unreadable for the question it exists to answer.

**Model calls belong to the stage that made them.** The engine's ledger is drained around
each node, so a completion is attributed to the node that caused it rather than to the
run as a whole.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig

from ..observability.logging import get_logger
from ..schemas.journal import RunJournal, StageRecord
from ..schemas.state import GlobalInvestigationState


class NodeFn(Protocol):
    """A parent-graph node.

    Declared as a protocol with *named* parameters rather than as a bare ``Callable``,
    because LangGraph's own node protocols accept ``state`` and ``config`` by keyword. A
    ``Callable`` alias is positional-only, so a wrapped node would fail to satisfy
    ``add_node`` — and the resulting error surfaces at graph assembly, far from here.
    """

    def __call__(
        self, state: GlobalInvestigationState, config: RunnableConfig
    ) -> Coroutine[Any, Any, dict[str, Any]]: ...


# --------------------------------------------------------------------------- #
# Delta compaction
# --------------------------------------------------------------------------- #
def _hypotheses(value: Any) -> Any:
    return [
        {"id": h.hypothesis_id, "category": h.category.value, "rank": h.rank, "status": h.status}
        for h in value
    ]


def _branches(value: Any) -> Any:
    return [
        {
            "id": b.branch_id,
            "status": b.status,
            "score": b.score,
            "depth": b.depth,
            "prune_reason": b.prune_reason,
        }
        for b in value
    ]


def _tasks(value: Any) -> Any:
    return [
        {"id": t.task_id, "agent": t.assigned_agent, "tools": t.allowed_tool_names,
         "status": t.status}
        for t in value
    ]


def _evidence(value: Any) -> Any:
    return [
        {
            "id": e.evidence_id,
            "tool": e.tool_name,
            "source_kind": e.source_kind,
            "summary": e.summary,
            "supports": e.supports,
            "contradicts": e.contradicts,
            "discriminating": bool(e.provenance.get("discriminating")),
        }
        for e in value
    ]


def _findings(value: Any) -> Any:
    return [{"task_id": f.task_id, "agent": f.agent_name, "status": f.completion_status}
            for f in value]


def _reviews(value: Any) -> Any:
    # The Critic's findings are short strings and they are the whole point of the review;
    # counting them and dropping the text would leave the review unreadable.
    return [
        {
            "id": r.review_id,
            "round": r.round_number,
            "recommendation": r.stop_recommendation,
            "rationale": r.rationale_summary,
            "unsupported_claims": list(r.unsupported_claims),
            "evidence_gaps": list(r.evidence_gaps),
            "pruning_recommendations": list(r.pruning_recommendations),
            "reopening_recommendations": list(r.reopening_recommendations),
            "strongest_hypothesis_id": r.strongest_hypothesis_id,
            "strongest_competitor_id": r.strongest_competitor_id,
        }
        for r in value
    ]


def _retrieved(value: Any) -> Any:
    return [{"doc_id": i.document.doc_id, "score": round(i.score, 4)} for i in value]


def _trust(value: Any) -> Any:
    return [
        {"doc_id": d.doc_id, "accepted": d.accepted, "gates": d.gate_results, "reasons": d.reasons}
        for d in value
    ]


def _influence(value: Any) -> Any:
    return None if value is None else value.model_dump(mode="json")


def _alert(value: Any) -> Any:
    return None if value is None else value.model_dump(mode="json")


# key -> compactor. Anything absent falls through to ``_generic``.
_COMPACTORS: dict[str, Callable[[Any], Any]] = {
    "hypotheses": _hypotheses,
    "branches": _branches,
    "tasks": _tasks,
    "evidence": _evidence,
    "findings": _findings,
    "critic_reviews": _reviews,
    "retrieved_items": _retrieved,
    "trust_decisions": _trust,
    "retrieval_influence": _influence,
    "alert": _alert,
}

# Keys held elsewhere in full. Repeating them per stage would triple the file for nothing.
_OMITTED = frozenset({"report", "raw_alert", "escalation_package", "journal"})


def _generic(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 12:
        return {"count": len(value)}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def compact_delta(update: dict[str, Any]) -> dict[str, Any]:
    """Summarise a node's returned update into something a person can read."""
    delta: dict[str, Any] = {}
    for key, value in update.items():
        if key in _OMITTED:
            delta[key] = "<omitted: held in full elsewhere>"
            continue
        compactor = _COMPACTORS.get(key, _generic)
        try:
            delta[key] = compactor(value)
        except Exception:  # noqa: BLE001 - a summary must never break a run
            delta[key] = repr(value)[:200]
    return delta


# --------------------------------------------------------------------------- #
# Node summaries — one readable line per stage
# --------------------------------------------------------------------------- #
def _summary_initialize(state: dict, update: dict) -> str:
    budgets = update.get("budgets") or {}
    return (
        f"opened {update.get('investigation_id')} — budget "
        f"{budgets.get('max_operational_calls')} call(s) / {budgets.get('max_rounds')} round(s)"
    )


def _summary_compose_report(state: dict, update: dict) -> str:
    report = update.get("report") or {}
    leading = report.get("leading_hypothesis") or {}
    cause = leading.get("category") or "no cause named"
    return (
        f"report composed: {report.get('outcome')} / {cause}, "
        f"{len(report.get('supporting_evidence_ids') or [])} supporting observation(s)"
    )


def _summary_parse_alert(state: dict, update: dict) -> str:
    if update.get("parse_error"):
        return f"rejected as invalid input: {update['parse_error'][:120]}"
    alert = update.get("alert")
    return f"parsed {alert.incident_id} ({alert.symptom_type})" if alert else "parsed"


def _summary_verify(state: dict, update: dict) -> str:
    detail = update.get("verification_detail") or {}
    explanation = detail.get("explanation") or detail.get("reason") or ""
    return f"{update.get('verification_status')}: {explanation}".strip(": ")


def _summary_hypotheses(state: dict, update: dict) -> str:
    hyps = update.get("hypotheses") or []
    return f"{len(hyps)} hypotheses: " + ", ".join(h.category.value for h in hyps)


def _summary_retrieve(state: dict, update: dict) -> str:
    influence = update.get("retrieval_influence")
    if influence is not None and influence.kind == "not_run":
        return f"retrieval not run: {influence.detail}"
    return f"{len(update.get('retrieved_items') or [])} candidate document(s)"


def _summary_trust_gate(state: dict, update: dict) -> str:
    decisions = update.get("trust_decisions") or []
    if not decisions:
        return "no retrieved documents to gate"
    accepted = [d.doc_id for d in decisions if d.accepted]
    influence = update.get("retrieval_influence")
    kind = influence.kind if influence is not None else "unknown"
    return (
        f"{len(accepted)} accepted, {len(decisions) - len(accepted)} rejected -> {kind}"
    )


def _summary_plan(state: dict, update: dict) -> str:
    tasks = update.get("tasks") or []
    round_number = update.get("round_number")
    detail = ", ".join(f"{t.assigned_agent}:{'+'.join(t.allowed_tool_names)}" for t in tasks)
    return f"round {round_number}: {len(tasks)} task(s) — {detail}" if tasks else (
        f"round {round_number}: no expandable branch left to plan"
    )


def _summary_dispatch(state: dict, update: dict) -> str:
    evidence = update.get("evidence") or []
    failures = update.get("failures") or []
    used = (update.get("budgets") or {}).get("calls_used")
    parts = [f"{len(evidence)} observation(s)"]
    if failures:
        parts.append(f"{len(failures)} failure(s)")
    parts.append(f"{used} call(s) used")
    return ", ".join(parts)


def _summary_merge(state: dict, update: dict) -> str:
    return (
        f"{len(state.get('evidence') or [])} evidence, "
        f"{len(state.get('findings') or [])} findings"
    )


def _summary_critic(state: dict, update: dict) -> str:
    reviews = update.get("critic_reviews") or []
    if not reviews:
        return "critic produced no review"
    review = reviews[-1]
    return (
        f"critic recommends '{review.stop_recommendation}' "
        f"({len(review.evidence_gaps)} gap(s), {len(review.pruning_recommendations)} prune(s))"
    )


def _summary_branch_policy(state: dict, update: dict) -> str:
    branches = update.get("branches") or []
    pruned = [b for b in branches if b.status == "pruned"]
    closed = [b for b in branches if b.status == "closed"]
    reopened = [b for b in branches if b.status == "reopened"]
    selected = update.get("beam_selected_ids") or []
    parts = [f"{len(selected)} selected"]
    if pruned:
        parts.append(f"{len(pruned)} pruned")
    if closed:
        parts.append(f"{len(closed)} closed")
    if reopened:
        parts.append(f"{len(reopened)} reopened")
    return ", ".join(parts)


def _summary_stop(state: dict, update: dict) -> str:
    verdict = "continue" if update.get("should_continue") else "stop"
    return f"{update.get('outcome')} ({verdict}) — {update.get('stop_reason')}"


def _summary_grounding(state: dict, update: dict) -> str:
    if not update:
        return "grounded"
    if update.get("outcome") == "inconclusive":
        return "grounding failed; abstaining"
    return "grounded after repairing dangling evidence references"


def _summary_risk(state: dict, update: dict) -> str:
    reasons = update.get("risk_reasons") or []
    return (
        f"risk {update.get('risk_tier')} -> {update.get('release_status')}"
        + (f" ({reasons[0]})" if reasons else "")
    )


def _summary_escalation(state: dict, update: dict) -> str:
    return "escalation package prepared" if update.get("escalation_package") else "no escalation"


def _summary_persist(state: dict, update: dict) -> str:
    return f"persisted {state.get('investigation_id')}"


_SUMMARIES: dict[str, Callable[[dict, dict], str]] = {
    "initialize_investigation": _summary_initialize,
    "parse_alert": _summary_parse_alert,
    "verify_incident": _summary_verify,
    "commander_generate_hypotheses": _summary_hypotheses,
    "retrieve_context": _summary_retrieve,
    "apply_retrieval_trust_gate": _summary_trust_gate,
    "commander_plan_round": _summary_plan,
    "dispatch_specialists": _summary_dispatch,
    "merge_specialist_results": _summary_merge,
    "critic_review": _summary_critic,
    "apply_branch_policy": _summary_branch_policy,
    "evaluate_stop": _summary_stop,
    "commander_compose_report": _summary_compose_report,
    "validate_grounding": _summary_grounding,
    "classify_risk": _summary_risk,
    "prepare_human_escalation": _summary_escalation,
    "persist_result": _summary_persist,
}


def summarize(node: str, state: dict[str, Any], update: dict[str, Any]) -> str:
    handler = _SUMMARIES.get(node)
    if handler is None:
        changed = ", ".join(sorted(update)) or "no state change"
        return f"{node}: {changed}"
    try:
        return handler(state, update)
    except Exception:  # noqa: BLE001 - a summary must never break a run
        return f"{node}: {', '.join(sorted(update)) or 'no state change'}"


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
def journaled(node_name: str, fn: NodeFn) -> NodeFn:
    """Wrap a parent-graph node so its execution is recorded as a stage."""

    async def _run(state: GlobalInvestigationState, config: RunnableConfig) -> dict[str, Any]:
        engine = config.get("configurable", {}).get("reasoning")
        started = datetime.now(UTC)
        clock = time.perf_counter()
        update = await fn(state, config)
        duration_ms = int((time.perf_counter() - clock) * 1000)

        # Completions made while this node ran. Draining here rather than inside each
        # node means a node that forgets to record its calls cannot exist.
        calls = []
        drain = getattr(engine, "drain_calls", None)
        if drain is not None:
            calls = [call.as_log_fields() for call in drain()]

        log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
        for call in calls:
            log.log("llm_call", node=node_name, **call)

        update = dict(update or {})
        sequence = len(state.get("journal") or [])
        # ``round_number`` is read from the update first: the planning node advances the
        # round, and its own stage belongs to the round it opened.
        round_number = int(
            update.get("round_number", state.get("round_number", 0)) or 0
        )
        record = StageRecord(
            stage_id=f"{sequence:03d}-{node_name}",
            sequence=sequence,
            node=node_name,
            round_number=round_number,
            started_at=started,
            duration_ms=duration_ms,
            summary=summarize(node_name, dict(state), update),
            delta=compact_delta(update),
            llm_calls=calls,
        )
        update["journal"] = [record]
        return update

    _run.__name__ = node_name
    return _run


def build_journal(
    state: dict[str, Any],
    versions: dict[str, Any] | None = None,
    tracing: dict[str, Any] | None = None,
) -> RunJournal:
    """Assemble the journal for a finished run."""
    return RunJournal(
        investigation_id=state.get("investigation_id") or "unknown",
        trace_id=state.get("trace_id"),
        scenario_id=state.get("scenario_id"),
        versions=versions or {},
        tracing=tracing,
        outcome=state.get("outcome"),
        release_status=state.get("release_status"),
        rounds_used=int(state.get("rounds_used") or 0),
        stages=list(state.get("journal") or []),
    )
