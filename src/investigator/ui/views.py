"""View models for the investigation UI (M5.3).

Pure functions from a :class:`~investigator.ui.loader.RunView` to plain data. The
Streamlit layer renders what these return and decides nothing — which is what makes the
UI's behaviour testable without a browser.

The tabs are organised around what this system actually does. A chat assistant's UI is a
transcript; this one runs a bounded search over competing hypotheses and then sometimes
refuses to conclude, so the views are built around rounds, branches, evidence, and the
gates that hold a result back.
"""

from __future__ import annotations

from typing import Any

from ..mcp_client.permissions import ROLE_PERMISSIONS, allowed_tools
from ..schemas.enums import AgentRole
from .loader import RunView

# Agents in the order they act. The two with no tools are listed deliberately: a
# permission boundary is easiest to believe when you can see who holds nothing.
AGENTS: tuple[tuple[str, AgentRole, str], ...] = (
    ("incident_commander", "commander", "Coordinates: hypotheses, task planning, report"),
    ("pipeline_investigator", "pipeline_investigator", "Pipeline execution evidence"),
    ("data_investigator", "data_investigator", "Dataset and quality evidence"),
    ("evidence_critic", "critic", "Adversarial review; advisory only"),
)

OUTCOME_TONE: dict[str, str] = {
    "diagnosed": "ok",
    "inconclusive": "warn",
    "not_an_incident": "info",
    "invalid_input": "warn",
}


# --------------------------------------------------------------------------- #
# Header — the verdict, carried wherever you are
# --------------------------------------------------------------------------- #
def header(run: RunView) -> dict[str, Any]:
    report = run.report
    guardrails = run.guardrails()
    outcome = report.get("outcome") or (run.journal.outcome if run.journal else None)
    return {
        "investigation_id": run.investigation_id,
        "scenario_id": run.scenario_id,
        "outcome": outcome,
        "tone": OUTCOME_TONE.get(outcome or "", "info"),
        "confidence": report.get("confidence"),
        "risk_tier": (report.get("risk") or {}).get("tier"),
        "release_status": run.release_status or report.get("release_status"),
        "rounds_used": guardrails.get("rounds_used") or (run.journal.rounds_used
                                                         if run.journal else 0),
        "calls_used": guardrails.get("operational_calls_used"),
        "calls_budget": guardrails.get("max_operational_calls"),
        "escalated": bool(run.escalation_package),
        "is_live": run.is_live,
    }


# --------------------------------------------------------------------------- #
# Investigation — the landing tab
# --------------------------------------------------------------------------- #
def investigation(run: RunView) -> dict[str, Any]:
    report = run.report
    supporting = set(report.get("supporting_evidence_ids") or [])
    index = report.get("evidence_index") or []
    return {
        "alert": report.get("alert") or {},
        "verification": report.get("verification") or {},
        "leading_hypothesis": report.get("leading_hypothesis"),
        "strongest_alternative": report.get("strongest_alternative"),
        "stop_reason": report.get("stop_reason"),
        "uncertainty": report.get("uncertainty"),
        "supporting_evidence": [e for e in index if e["evidence_id"] in supporting],
        "contradicting_evidence": [
            e for e in index if e["evidence_id"] in set(report.get("contradicting_evidence_ids")
                                                        or [])
        ],
        "recommended_next_checks": report.get("recommended_next_checks") or [],
        "escalation_package": run.escalation_package,
    }


# --------------------------------------------------------------------------- #
# Orchestration — the graph, with the path this run actually took
# --------------------------------------------------------------------------- #
_ROUND_NODES = frozenset(
    {
        "commander_plan_round",
        "dispatch_specialists",
        "merge_specialist_results",
        "critic_review",
        "apply_branch_policy",
        "evaluate_stop",
    }
)


def orchestration_dot(run: RunView, round_number: int | None = None) -> str:
    """DOT for the parent graph, highlighting what executed.

    The structure is read from the compiled graph rather than drawn by hand, so the
    picture cannot drift from the code it claims to describe. ``round_number`` dims
    everything outside that round, which is what the round slider walks.
    """
    from ..graph.parent_graph import build_parent_graph

    graph = build_parent_graph().get_graph()
    executed = {s.node for s in run.stages()}
    in_round = (
        {s.node for s in run.stages(round_number=round_number)}
        if round_number is not None
        else executed
    )
    counts: dict[str, int] = {}
    for stage in run.stages():
        counts[stage.node] = counts.get(stage.node, 0) + 1

    lines = [
        "digraph investigation {",
        '  rankdir=TB;',
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#94a3b8"];',
    ]
    for node in graph.nodes:
        if node in {"__start__", "__end__"}:
            lines.append(f'  "{node}" [shape=circle, label="", width=0.2, '
                         f'fillcolor="#64748b", color="#64748b"];')
            continue
        repeats = counts.get(node, 0)
        label = node if repeats <= 1 else f"{node}\\n×{repeats}"
        if node in in_round:
            fill, border, pen = "#1d4ed8", "#1d4ed8", 2
            font = "white"
        elif node in executed:
            fill, border, pen, font = "#dbeafe", "#93c5fd", 1, "#1e3a5f"
        else:
            fill, border, pen, font = "#f1f5f9", "#e2e8f0", 1, "#94a3b8"
        lines.append(
            f'  "{node}" [label="{label}", fillcolor="{fill}", color="{border}", '
            f'penwidth={pen}, fontcolor="{font}"];'
        )

    for edge in graph.edges:
        both_ran = edge.source in executed and edge.target in executed
        style = "solid" if both_ran else "dashed"
        color = "#1d4ed8" if both_ran else "#cbd5e1"
        # The loop back into planning is the multi-round behaviour; it earns a label.
        label = ""
        if edge.source == "evaluate_stop" and edge.target == "commander_plan_round":
            label = ' label="another round"'
        elif edge.conditional:
            label = ' style=dashed'
        lines.append(
            f'  "{edge.source}" -> "{edge.target}" [style={style}, color="{color}"{label}];'
        )
    lines.append("}")
    return "\n".join(lines)


def stage_timeline(run: RunView, round_number: int | None = None) -> list[dict[str, Any]]:
    """The ordered story: what ran, when, for how long, and what it decided."""
    return [
        {
            "sequence": s.sequence,
            "node": s.node,
            "round": s.round_number,
            "duration_ms": s.duration_ms,
            "summary": s.summary,
            "delta_keys": sorted(s.delta),
            "llm_calls": len(s.llm_calls),
            "in_round": round_number is None or s.round_number == round_number,
        }
        for s in run.stages()
    ]


# --------------------------------------------------------------------------- #
# Hypotheses and beam — the FR-703 rubric, itemized
# --------------------------------------------------------------------------- #
RUBRIC_COMPONENTS: tuple[tuple[str, str, int], ...] = (
    ("consistency", "Consistency with current evidence", 4),
    ("discriminating_value", "Discriminating value of next action", 2),
    ("support_independence", "Independence and quality of support", 2),
    ("retrieved_context", "Applicable retrieved context", 1),
    ("cost_efficiency", "Expected cost and latency", 1),
    ("contradiction_penalty", "Critical contradiction penalty", 0),
)

DISPOSITIONS: dict[str, str] = {
    "selected": "in the beam",
    "active": "alive, not selected",
    "pruned": "pruned",
    "closed": "no candidate actions left",
    "reopened": "reopened",
}


def beam_rounds(run: RunView) -> list[dict[str, Any]]:
    """Per round: every branch, its itemized score, and its disposition with the reason."""
    rounds: list[dict[str, Any]] = []
    for stage in run.stages("apply_branch_policy"):
        branches = stage.delta.get("branches") or []
        scores = stage.delta.get("branch_scores") or {}
        selected = set(stage.delta.get("beam_selected_ids") or [])
        rows = []
        for branch in branches:
            score = scores.get(branch["id"], {})
            rows.append(
                {
                    "branch_id": branch["id"],
                    "status": branch["status"],
                    "disposition": DISPOSITIONS.get(branch["status"], branch["status"]),
                    "selected": branch["id"] in selected,
                    "depth": branch.get("depth"),
                    "total": score.get("total", branch.get("score")),
                    "components": {key: score.get(key) for key, _, _ in RUBRIC_COMPONENTS},
                    "reasons": score.get("reasons") or [],
                    "prune_reason": branch.get("prune_reason"),
                }
            )
        rows.sort(key=lambda r: (-(r["total"] or 0), r["branch_id"]))
        rounds.append(
            {
                "round": stage.round_number,
                "summary": stage.summary,
                "selected": sorted(selected),
                "branches": rows,
                "disagreements": stage.delta.get("critic_disagreements") or [],
            }
        )
    return rounds


def hypotheses(run: RunView) -> list[dict[str, Any]]:
    """Final hypothesis states, in rank order."""
    latest = None
    for stage in run.stages():
        if "hypotheses" in stage.delta:
            latest = stage.delta["hypotheses"]
    by_id: dict[str, dict[str, Any]] = {}
    for stage in run.stages():
        for hypothesis in stage.delta.get("hypotheses") or []:
            by_id[hypothesis["id"]] = hypothesis
    ordered = sorted(by_id.values(), key=lambda h: h.get("rank") or 99)
    return ordered or (latest or [])


# --------------------------------------------------------------------------- #
# Agents — who may call what, and what they actually did
# --------------------------------------------------------------------------- #
def agent_cards(run: RunView) -> list[dict[str, Any]]:
    from ..agents.prompts import PROMPT_VERSION

    index = run.report.get("evidence_index") or []
    tasks: dict[str, list[dict[str, Any]]] = {}
    for stage in run.stages("commander_plan_round"):
        for task in stage.delta.get("tasks") or []:
            tasks.setdefault(task["agent"], []).append({**task, "round": stage.round_number})

    tool_calls: dict[str, int] = {}
    for stage in run.stages("dispatch_specialists"):
        for item in stage.delta.get("evidence") or []:
            for agent, role, _ in AGENTS:
                if item.get("tool") in allowed_tools(role):
                    tool_calls[agent] = tool_calls.get(agent, 0) + 1
                    break

    reviews = [r for s in run.stages("critic_review") for r in (s.delta.get("critic_reviews")
                                                                or [])]

    cards = []
    for agent, role, purpose in AGENTS:
        permitted = sorted(allowed_tools(role))
        decisions: list[str] = []
        if agent == "incident_commander":
            decisions = [
                f"round {t['round']}: {t['agent']} → {', '.join(t['tools'])}"
                for group in tasks.values()
                for t in group
            ]
        elif agent == "evidence_critic":
            decisions = [
                f"round {r.get('round')}: recommended '{r.get('recommendation')}'"
                for r in reviews
            ]
        else:
            decisions = [
                f"{e['tool_name']}: {e['summary']}"
                for e in index
                if e.get("tool_name") in permitted
            ][:8]
        cards.append(
            {
                "agent": agent,
                "role": role,
                "purpose": purpose,
                "prompt_version": f"{role}:{PROMPT_VERSION}",
                "permitted_tools": permitted,
                "holds_no_tools": not permitted,
                "servers": sorted(ROLE_PERMISSIONS.get(role, {})),
                "calls_made": tool_calls.get(agent, 0),
                "tasks": tasks.get(role, []),
                "decisions": decisions,
            }
        )
    return cards


# --------------------------------------------------------------------------- #
# Evidence — current versus historical is the system's central claim
# --------------------------------------------------------------------------- #
def evidence_rows(run: RunView, *, current_only: bool = False) -> list[dict[str, Any]]:
    discriminating: dict[str, bool] = {}
    for stage in run.stages():
        for item in stage.delta.get("evidence") or []:
            discriminating[item["id"]] = bool(item.get("discriminating"))

    payloads: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    if run.live_state:
        for evidence in run.live_state.get("evidence") or []:
            payloads[evidence.evidence_id] = evidence.payload
            provenance[evidence.evidence_id] = evidence.provenance

    supporting = set(run.report.get("supporting_evidence_ids") or [])
    rows = []
    for item in run.report.get("evidence_index") or []:
        is_current = item["source_kind"] == "current_operational"
        if current_only and not is_current:
            continue
        rows.append(
            {
                **item,
                "is_current": is_current,
                "discriminating": discriminating.get(item["evidence_id"], False),
                "supports_diagnosis": item["evidence_id"] in supporting,
                "payload": payloads.get(item["evidence_id"]),
                "provenance": provenance.get(item["evidence_id"]),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Memory — six gates, and what memory was allowed to do
# --------------------------------------------------------------------------- #
TRUST_GATES: tuple[str, ...] = (
    "relevance",
    "permitted_type",
    "confirmed",
    "metadata_match",
    "not_stale",
    "no_conflict",
)


def memory_view(run: RunView) -> dict[str, Any]:
    candidates = run.first_delta("retrieve_context", "retrieved_items", []) or []
    decisions = run.first_delta("apply_retrieval_trust_gate", "trust_decisions", []) or []
    influence = (
        run.first_delta("apply_retrieval_trust_gate", "retrieval_influence")
        or run.first_delta("retrieve_context", "retrieval_influence")
        or (run.report.get("historical_context") or {}).get("retrieval_influence")
        or {}
    )
    scores = {c["doc_id"]: c["score"] for c in candidates}
    return {
        "candidates": candidates,
        "gates": TRUST_GATES,
        "decisions": [
            {
                "doc_id": d["doc_id"],
                "accepted": d["accepted"],
                "score": scores.get(d["doc_id"]),
                "gate_results": d.get("gates") or {},
                "failed_gates": [g for g, ok in (d.get("gates") or {}).items() if not ok],
                "reasons": d.get("reasons") or [],
            }
            for d in decisions
        ],
        "influence": influence,
        "order_before": influence.get("order_before") or [],
        "order_after": influence.get("order_after") or [],
        "reordered": bool(influence.get("order_before"))
        and influence.get("order_before") != influence.get("order_after"),
        "note": (run.report.get("historical_context") or {}).get("note"),
    }


# --------------------------------------------------------------------------- #
# Critic — advisory, beside the authority
# --------------------------------------------------------------------------- #
def critic_rounds(run: RunView) -> list[dict[str, Any]]:
    stop_by_round = {
        s.round_number: s.delta for s in run.stages("evaluate_stop")
    }
    rounds = []
    for stage in run.stages("critic_review"):
        for review in stage.delta.get("critic_reviews") or []:
            stop = stop_by_round.get(stage.round_number, {})
            deterministic = stop.get("outcome")
            recommended = review.get("recommendation")
            rounds.append(
                {
                    "round": stage.round_number,
                    "review_id": review.get("id"),
                    "recommendation": recommended,
                    "rationale": review.get("rationale"),
                    "unsupported_claims": review.get("unsupported_claims") or [],
                    "evidence_gaps": review.get("evidence_gaps") or [],
                    "pruning_recommendations": review.get("pruning_recommendations") or [],
                    "reopening_recommendations": review.get("reopening_recommendations") or [],
                    "deterministic_outcome": deterministic,
                    "disagrees": bool(deterministic) and recommended != deterministic,
                }
            )
    return rounds


def disagreements(run: RunView) -> list[str]:
    """Every recorded conflict between advisory judgment and deterministic policy."""
    recorded: list[str] = list(run.guardrails().get("critic_disagreements") or [])
    for stage in run.stages():
        for note in stage.delta.get("critic_disagreements") or []:
            if note not in recorded:
                recorded.append(note)
    return recorded


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #
def traces(run: RunView) -> dict[str, Any]:
    journal = run.journal
    calls = journal.llm_calls if journal else []
    return {
        "llm_calls": calls,
        "total_tokens": sum(c.get("total_tokens", 0) for c in calls),
        "total_latency_ms": sum(c.get("latency_ms", 0) for c in calls),
        "tracing": journal.tracing if journal else None,
        "versions": journal.versions if journal else {},
        "stages": stage_timeline(run),
        "trace_id": journal.trace_id if journal else None,
    }


# --------------------------------------------------------------------------- #
# Safety — the operational dashboard (FR-1204), per run
# --------------------------------------------------------------------------- #
def safety(run: RunView) -> dict[str, Any]:
    report = run.report
    guardrails = run.guardrails()
    risk = report.get("risk") or {}
    grounding_stage = run.stages("validate_grounding")
    grounded = None
    if grounding_stage:
        grounded = "failed" not in grounding_stage[-1].summary

    return {
        "risk_tier": risk.get("tier"),
        "risk_reasons": risk.get("reasons") or [],
        "release_status": run.release_status or report.get("release_status"),
        "released": (run.release_status or report.get("release_status")) == "released",
        "blocked_tool_attempts": guardrails.get("blocked_tool_attempts", 0),
        "grounded": grounded,
        "grounding_note": grounding_stage[-1].summary if grounding_stage else None,
        "tool_failures": guardrails.get("failures") or [],
        "diagnosis_checks": guardrails.get("diagnosis_checks") or {},
        "reasoning_fallbacks": guardrails.get("reasoning_fallbacks") or [],
        "escalation_required": bool(run.escalation_package),
        "calls_used": guardrails.get("operational_calls_used"),
        "calls_budget": guardrails.get("max_operational_calls"),
    }


def acceptance_snapshot(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """FR-1305 gates from an evaluation report, as pass/fail rows."""
    return [
        {
            "name": gate["name"],
            "metric": gate["metric"],
            "value": gate["value"],
            "limit": gate["limit"],
            "passed": gate["passed"],
            "requirement": gate["requirement"],
        }
        for gate in (payload.get("acceptance") or {}).get("gates", [])
    ]
