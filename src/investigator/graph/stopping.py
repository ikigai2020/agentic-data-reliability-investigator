"""Deterministic stopping / outcome evaluator (FR-900, FR-901, FR-902, FR-903).

Only this deterministic code sets an outcome — agents may recommend but never decide
(FR-903). The Evidence Critic scores and recommends; beam search prunes and reopens; the
final call on ``diagnosed`` / ``inconclusive`` / ``not_an_incident`` is made here and
nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import evidence_view
from ..schemas.enums import ConfidenceBand, Outcome, VerificationStatus
from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis


@dataclass
class Evaluation:
    outcome: Outcome
    stop_reason: str
    confidence: ConfidenceBand
    leading_hypothesis_id: str | None = None
    strongest_competitor_id: str | None = None
    escalation_required: bool = False
    updated_hypotheses: list[Hypothesis] = field(default_factory=list)
    diagnosis_checks: dict[str, bool] = field(default_factory=dict)


# Evidence queries live in :mod:`investigator.evidence_view` so the stopping evaluator,
# the branch scorer, and the Evidence Critic all read current evidence identically.
_current_contradictions = evidence_view.contradictions
_independent_support_tools = evidence_view.independent_support_tools
_has_discriminating_support = evidence_view.has_discriminating_support
_was_tested = evidence_view.was_tested


def evaluate(
    *,
    verification_status: VerificationStatus | None,
    hypotheses: list[Hypothesis],
    evidence: list[Evidence],
    calls_used: int,
    max_calls: int,
    failures: list[str],
    min_supporting: int,
) -> Evaluation:
    """Compute the terminal outcome deterministically."""
    # FR-901 / verification gating.
    if verification_status == "not_verified":
        return Evaluation(
            outcome="not_an_incident",
            stop_reason="authoritative current check shows the alert condition is absent",
            confidence="not_applicable",
        )
    if verification_status == "verification_unavailable" or verification_status is None:
        return Evaluation(
            outcome="inconclusive",
            stop_reason="verification could not be performed via MCP",
            confidence="not_applicable",
            escalation_required=True,
        )

    # Rank hypotheses by independent current support, then by ordinal rank.
    def support_key(h: Hypothesis) -> tuple[int, int]:
        return (len(_independent_support_tools(h.hypothesis_id, evidence)), -h.rank)

    ranked = sorted(hypotheses, key=support_key, reverse=True)
    leading = ranked[0] if ranked else None

    updated: dict[str, Hypothesis] = {h.hypothesis_id: h.model_copy(deep=True) for h in hypotheses}

    if leading is None:
        return Evaluation(
            outcome="inconclusive",
            stop_reason="no hypotheses generated",
            confidence="not_applicable",
            escalation_required=True,
        )

    lead_id = leading.hypothesis_id
    independent = _independent_support_tools(lead_id, evidence)
    discriminating = _has_discriminating_support(lead_id, evidence)
    lead_contradictions = _current_contradictions(lead_id, evidence)

    # Strongest competitor = highest-support other hypothesis, else highest-ranked other.
    competitors = [h for h in ranked if h.hypothesis_id != lead_id]
    competitor = competitors[0] if competitors else None
    competitor_id = competitor.hypothesis_id if competitor else None
    competitor_weakened = True
    if competitor is not None:
        comp_supports = _independent_support_tools(competitor.hypothesis_id, evidence)
        comp_contradictions = _current_contradictions(competitor.hypothesis_id, evidence)
        comp_tested = _was_tested(competitor.hypothesis_id, evidence)
        competitor_weakened = comp_tested and (
            bool(comp_contradictions) or len(comp_supports) == 0
        )

    checks = {
        "verified": True,
        "two_independent_current_supports": len(independent) >= min_supporting,
        "at_least_one_discriminating": discriminating,
        "strongest_competitor_weakened": competitor_weakened,
        "no_critical_current_contradiction": len(lead_contradictions) == 0,
        "leading_has_evidence": len(independent) > 0,
    }

    if all(checks.values()):
        # Mark statuses (FR-301). Leading supported; weakened/rejected competitors.
        updated[lead_id].status = "supported"
        for h in competitors:
            if _current_contradictions(h.hypothesis_id, evidence):
                updated[h.hypothesis_id].status = "rejected"
            elif _was_tested(h.hypothesis_id, evidence):
                updated[h.hypothesis_id].status = "weakened"
        strong = len(independent) >= min_supporting + 1
        confidence: ConfidenceBand = "high" if strong else "moderate"
        return Evaluation(
            outcome="diagnosed",
            stop_reason="leading hypothesis meets all FR-900 diagnosis criteria",
            confidence=confidence,
            leading_hypothesis_id=lead_id,
            strongest_competitor_id=competitor_id,
            escalation_required=False,
            updated_hypotheses=list(updated.values()),
            diagnosis_checks=checks,
        )

    # FR-902 inconclusive. Determine the dominant reason for the stop.
    if len(independent) < min_supporting:
        reason = "insufficient independent current evidence to diagnose"
        if calls_used >= max_calls:
            reason += "; operational budget also exhausted"
    elif lead_contradictions:
        reason = "unresolved current contradiction against the leading hypothesis"
    elif not competitor_weakened:
        reason = "strongest competitor not adequately tested/weakened"
    else:
        reason = "diagnosis criteria not met"
    if failures:
        reason += f"; {len(failures)} tool failure(s) recorded"

    return Evaluation(
        outcome="inconclusive",
        stop_reason=reason,
        confidence="not_applicable",
        leading_hypothesis_id=lead_id if independent else None,
        strongest_competitor_id=competitor_id,
        escalation_required=True,
        updated_hypotheses=list(updated.values()),
        diagnosis_checks=checks,
    )
