"""Final report, grounding validation, and escalation package.

FR-1000 structured report, FR-1001 categorical confidence (no invented probabilities),
FR-1002 deterministic grounding validator, FR-1103 escalation package.
"""

from __future__ import annotations

from typing import Any

from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis


def _tool_inventory(evidence: list[Evidence], failures: list[str]) -> dict[str, list[str]]:
    called = sorted({e.tool_name for e in evidence if e.tool_name})
    unavailable = sorted({f.split(":")[0] for f in failures})
    return {"called": called, "unavailable": unavailable}


def _historical_context(state: dict[str, Any], evidence: list[Evidence]) -> dict[str, Any]:
    """FR-1000: historical context and its effect — labeled, never counted as proof (AD-005)."""
    influence = state.get("retrieval_influence")
    memory_ids = [e.evidence_id for e in evidence if e.source_kind != "current_operational"]
    effect = influence.kind if influence is not None else "not_run"
    return {
        "retrieval_influence": influence.model_dump(mode="json") if influence is not None else None,
        "effect": effect,
        "historical_evidence_ids": memory_ids,
        "note": "historical context may reorder investigation but never establishes the diagnosis",
    }


def build_report(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble the FR-1000 structured report from accepted state."""
    evidence: list[Evidence] = state.get("evidence", [])
    hypotheses: list[Hypothesis] = state.get("hypotheses", [])
    alert = state.get("alert")
    lead_id = state.get("leading_hypothesis_id")
    competitor_id = state.get("strongest_competitor_id")

    by_id = {h.hypothesis_id: h for h in hypotheses}
    leading = by_id.get(lead_id) if lead_id else None
    competitor = by_id.get(competitor_id) if competitor_id else None

    supporting_ids = (
        [e.evidence_id for e in evidence if lead_id in e.supports] if lead_id else []
    )
    contradicting_ids = (
        [e.evidence_id for e in evidence if lead_id in e.contradicts] if lead_id else []
    )

    return {
        "investigation_id": state.get("investigation_id"),
        "alert": alert.model_dump(mode="json") if alert is not None else None,
        "verification": {
            "status": state.get("verification_status"),
            "detail": state.get("verification_detail"),
        },
        "outcome": state.get("outcome"),
        "stop_reason": state.get("stop_reason"),
        "confidence": state.get("confidence"),
        "leading_hypothesis": leading.model_dump(mode="json") if leading else None,
        "supporting_evidence_ids": supporting_ids,
        "contradicting_evidence_ids": contradicting_ids,
        "strongest_alternative": competitor.model_dump(mode="json") if competitor else None,
        "historical_context": _historical_context(state, evidence),
        "tools": _tool_inventory(evidence, state.get("failures", [])),
        "uncertainty": state.get("stop_reason") if state.get("outcome") != "diagnosed" else None,
        "recommended_next_checks": _next_checks(state),
        "guardrail_and_budget_events": {
            "operational_calls_used": state.get("budgets", {}).get("calls_used"),
            "max_operational_calls": state.get("budgets", {}).get("max_operational_calls"),
            "failures": state.get("failures", []),
            "diagnosis_checks": state.get("diagnosis_checks", {}),
            "rounds_used": state.get("rounds_used"),
            # FR-1107: advisory critic judgment the deterministic controller overruled.
            "critic_disagreements": state.get("critic_disagreements", []),
            # AD-004: steps that fell back from LLM reasoning to deterministic rules.
            "reasoning_fallbacks": state.get("reasoning_fallbacks", []),
        },
        "branches": [
            {
                "branch_id": b.branch_id,
                "hypothesis_id": b.hypothesis_id,
                "depth": b.depth,
                "score": b.score,
                "status": b.status,
                "prune_reason": b.prune_reason,
                "action_history": b.action_history,
            }
            for b in state.get("branches", [])
        ],
        "evidence_index": [
            {
                "evidence_id": e.evidence_id,
                "tool_name": e.tool_name,
                "source_kind": e.source_kind,
                "summary": e.summary,
                "supports": e.supports,
                "contradicts": e.contradicts,
                "freshness_status": e.freshness_status,
            }
            for e in evidence
        ],
    }


def _next_checks(state: dict[str, Any]) -> list[str]:
    """Open questions a human should pick up (FR-1103).

    Everything except a *rejected* hypothesis is still open. A hypothesis the beam
    weakened because the call budget ran out was never falsified — it is the most useful
    thing to hand a human, so pruning for budget must not silently erase it.
    """
    if state.get("outcome") == "diagnosed":
        return []
    checks: list[str] = []
    hypotheses: list[Hypothesis] = state.get("hypotheses", [])
    for h in sorted(hypotheses, key=lambda h: h.rank):
        if h.status != "rejected":
            checks.append(f"[{h.category.value}] {h.discriminating_question}")
    return checks[:5]


def validate_grounding(report: dict[str, Any], evidence: list[Evidence]) -> tuple[bool, list[str]]:
    """FR-1002: every material claim must reference existing evidence; no conflation.

    Returns (is_grounded, issues). Callers permit one bounded repair, then abstain.
    """
    issues: list[str] = []
    valid_ids = {e.evidence_id for e in evidence}

    for field_name in ("supporting_evidence_ids", "contradicting_evidence_ids"):
        for ev_id in report.get(field_name, []):
            if ev_id not in valid_ids:
                issues.append(f"{field_name} references unknown evidence '{ev_id}'")

    # Current vs historical must not be conflated (AD-005).
    for item in report.get("evidence_index", []):
        if item["source_kind"] != "current_operational" and (
            item["supports"] or item["contradicts"]
        ):
            issues.append(
                f"non-current evidence '{item['evidence_id']}' used as support/contradiction"
            )

    # A diagnosed outcome must have supporting current evidence for the leading hypothesis.
    if report.get("outcome") == "diagnosed" and not report.get("supporting_evidence_ids"):
        issues.append("diagnosed outcome without supporting current evidence")

    return (len(issues) == 0, issues)


def build_escalation_package(state: dict[str, Any]) -> dict[str, Any]:
    """FR-1103 human-escalation package."""
    evidence: list[Evidence] = state.get("evidence", [])
    hypotheses: list[Hypothesis] = state.get("hypotheses", [])
    alert = state.get("alert")
    return {
        "verified_facts": {
            "verification_status": state.get("verification_status"),
            "verification_detail": state.get("verification_detail"),
            "alert": alert.model_dump(mode="json") if alert is not None else None,
        },
        "leading_hypothesis_id": state.get("leading_hypothesis_id"),
        "alternative_hypotheses": [
            h.model_dump(mode="json") for h in hypotheses if h.status != "rejected"
        ],
        "evidence_ids": [e.evidence_id for e in evidence],
        "contradictions": [
            {"evidence_id": e.evidence_id, "contradicts": e.contradicts}
            for e in evidence
            if e.contradicts
        ],
        "unavailable_sources": state.get("failures", []),
        "stop_reason": state.get("stop_reason"),
        "recommended_next_checks": _next_checks(state),
    }
