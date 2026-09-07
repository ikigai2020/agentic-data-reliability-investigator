"""Evidence Critic (FR-130, §6.1 agent definition)."""

from __future__ import annotations

from investigator.agents.evidence_critic import graph as critic_module
from investigator.mcp_client.permissions import allowed_tools
from investigator.search import beam

from .fixtures import (
    H_ORCH,
    H_QUALITY,
    H_SOURCE,
    H_TRANSFORM,
    branches,
    evidence,
    finding,
    hypotheses,
)

CRITIC = critic_module.build()


async def _review(*, ev=None, findings=None, brs=None, scores=None, hyps=None):
    ev = ev or []
    hyps = hyps if hyps is not None else hypotheses()
    brs = brs if brs is not None else beam.advance_branches(branches(), ev)
    result = await CRITIC.ainvoke(
        {
            "round_number": 1,
            "hypotheses": hyps,
            "evidence": ev,
            "findings": findings or [],
            "branches": brs,
            "branch_scores": scores or {},
            "retrieval_influence": None,
            "min_supporting": 2,
            "prune_threshold": 5,
        }
    )
    return result["review"]


# --------------------------------------------------------------------------- #
# §6.1 agent definition
# --------------------------------------------------------------------------- #
def test_critic_is_a_distinct_agent_with_no_operational_tools() -> None:
    assert critic_module.ROLE == "critic"
    assert allowed_tools("critic") == frozenset()
    assert critic_module.PROMPT_ID == "evidence_critic:v1"
    assert "[missing prompt" not in critic_module.PROMPT
    assert "adversarial reviewer" in critic_module.PROMPT


# --------------------------------------------------------------------------- #
# FR-130.2 / FR-130.3 claims, provenance, contradictions, reuse
# --------------------------------------------------------------------------- #
async def test_finding_claiming_unbacked_support_is_flagged() -> None:
    review = await _review(
        findings=[finding(hypothesis_id=H_TRANSFORM, evidence_ids=[], supports=[H_TRANSFORM])]
    )
    assert any("no current observation" in c for c in review.unsupported_claims)


async def test_finding_citing_missing_evidence_is_flagged() -> None:
    review = await _review(
        findings=[finding(hypothesis_id=H_TRANSFORM, evidence_ids=["ev_ghost"])]
    )
    assert any("ev_ghost" in c for c in review.unsupported_claims)


async def test_failed_task_asserting_support_is_flagged() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    review = await _review(
        ev=ev,
        findings=[
            finding(
                hypothesis_id=H_TRANSFORM,
                evidence_ids=[ev[0].evidence_id],
                status="failed",
            )
        ],
    )
    assert any("failed task still asserts support" in c for c in review.unsupported_claims)


async def test_missing_provenance_is_an_evidence_gap() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    stripped = ev[0].model_copy(update={"provenance": {"discriminating": True}})
    review = await _review(ev=[stripped])
    assert any("missing provenance" in g for g in review.evidence_gaps)


async def test_evidence_reused_across_hypotheses_is_flagged() -> None:
    """One observation propping up two hypotheses is not independent corroboration."""
    ev = [
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_TRANSFORM,
            extra_supports=[H_SOURCE],
        )
    ]
    review = await _review(ev=ev)
    assert any("reused as support" in g for g in review.evidence_gaps)


async def test_simultaneous_support_and_contradiction_is_flagged() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(
            tool="get_recent_transformation_changes",
            hypothesis_id=H_TRANSFORM,
            supports=False,
        ),
    ]
    review = await _review(ev=ev)
    assert any("both supports and contradicts" in g for g in review.evidence_gaps)


async def test_untested_hypothesis_is_reported_as_a_gap_not_a_weakness() -> None:
    review = await _review()
    gaps = " ".join(review.evidence_gaps)
    assert "never tested by a current observation" in gaps


# --------------------------------------------------------------------------- #
# FR-130.4 independence
# --------------------------------------------------------------------------- #
async def test_repeated_readings_from_one_tool_are_called_out() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_TRANSFORM,
            round_number=2,
        ),
    ]
    review = await _review(ev=ev)
    assert any("repeated readings are one source" in g for g in review.evidence_gaps)


async def test_support_from_only_non_discriminating_observations_is_flagged() -> None:
    ev = [evidence(tool="get_table_metrics", hypothesis_id=H_QUALITY, discriminating=False)]
    review = await _review(ev=ev)
    assert any("non-discriminating" in g for g in review.evidence_gaps)


# --------------------------------------------------------------------------- #
# FR-130.6 / FR-130.8 ranking and bias
# --------------------------------------------------------------------------- #
async def test_leading_and_competing_hypotheses_are_identified() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]
    review = await _review(ev=ev)
    assert review.strongest_hypothesis_id == H_TRANSFORM
    assert review.strongest_competitor_id is not None
    assert review.strongest_competitor_id != H_TRANSFORM


async def test_untested_competitor_is_challenged_not_treated_as_defeated() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]
    review = await _review(ev=ev)
    text = " ".join(review.evidence_gaps)
    assert "no discriminating check" in text or "confirmation bias risk" in text


async def test_historical_evidence_carrying_support_is_challenged() -> None:
    """AD-005 breach: memory must never be doing evidentiary work."""
    memory = evidence(
        tool=None,  # type: ignore[arg-type]
        hypothesis_id=H_TRANSFORM,
        source_kind="retrieved_incident",
        evidence_id="ev_mem_1",
    )
    ev = [
        memory,
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]
    review = await _review(ev=ev)
    assert any("historical evidence carrying" in g for g in review.evidence_gaps)


# --------------------------------------------------------------------------- #
# FR-130.5 / FR-130.7 recommendations — advisory only (FR-903)
# --------------------------------------------------------------------------- #
async def test_low_scoring_branch_draws_a_prune_recommendation() -> None:
    scores = {f"B-{H_ORCH}": 2.0, f"B-{H_TRANSFORM}": 7.0}
    review = await _review(scores=scores)
    assert any(f"B-{H_ORCH}" in r for r in review.pruning_recommendations)
    assert not any(f"B-{H_TRANSFORM}" in r for r in review.pruning_recommendations)


async def test_falsified_branch_draws_a_prune_recommendation() -> None:
    ev = [
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_SOURCE,
            supports=False,
            discriminating=True,
        )
    ]
    review = await _review(ev=ev)
    assert any("falsified" in r for r in review.pruning_recommendations)


async def test_supported_pruned_branch_draws_a_reopen_recommendation() -> None:
    ev = [evidence(tool="get_quality_results", hypothesis_id=H_QUALITY)]
    brs = [
        b.model_copy(update={"status": "pruned", "prune_reason": "low score"})
        if b.hypothesis_id == H_QUALITY
        else b
        for b in beam.advance_branches(branches(), ev)
    ]
    review = await _review(ev=ev, brs=brs)
    assert any(H_QUALITY in r for r in review.reopening_recommendations)


async def test_recommendation_is_diagnosed_only_with_independent_discriminating_support() -> None:
    weak = await _review(ev=[evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)])
    assert weak.stop_recommendation == "continue"

    strong = await _review(
        ev=[
            evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
            evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
        ]
    )
    assert strong.stop_recommendation == "diagnosed"


async def test_review_is_deterministic() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    first = await _review(ev=ev)
    second = await _review(ev=ev)
    assert first.model_dump() == second.model_dump()
