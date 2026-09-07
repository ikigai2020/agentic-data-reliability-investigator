"""Branch scoring rubric (FR-703)."""

from __future__ import annotations

from investigator.schemas.enums import RootCauseCategory
from investigator.search import thought
from investigator.search.scoring import CONTRADICTION_PENALTY, MAX_SCORE, score_branch

from .fixtures import H_SOURCE, H_TRANSFORM, branches, evidence, hypotheses


def _pair(hypothesis_id: str):
    hyp = next(h for h in hypotheses() if h.hypothesis_id == hypothesis_id)
    branch = next(b for b in branches() if b.hypothesis_id == hypothesis_id)
    return branch, hyp


def _score(hypothesis_id: str, ev, *, calls_remaining: int = 8, retrieval=None):
    branch, hyp = _pair(hypothesis_id)
    tried = {e.tool_name for e in ev if e.tool_name}
    proposals = thought.propose(hyp, branch.branch_id, tried, 2)
    return score_branch(
        branch,
        hyp,
        ev,
        next_thought=proposals[0] if proposals else None,
        retrieval_categories=retrieval,
        calls_remaining=calls_remaining,
    )


def test_untested_branch_scores_below_the_prune_threshold() -> None:
    """Nothing is presumed: an unexamined hypothesis has not earned a place in the beam."""
    score = _score(H_TRANSFORM, [])
    assert score.consistency == 1  # no evidence either way
    assert score.support_independence == 0
    assert score.total < 5


def test_two_independent_supports_raise_consistency_and_independence() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]
    score = _score(H_TRANSFORM, ev)
    assert score.consistency == 3  # 1 + 2 independent tools
    assert score.support_independence == 2
    assert score.total >= 5


def test_repeated_readings_from_one_tool_are_not_independent() -> None:
    """FR-900.2 independence is by distinct tool, so a second reading adds nothing."""
    once = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    twice = [
        *once,
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_TRANSFORM,
            round_number=2,
        ),
    ]
    assert _score(H_TRANSFORM, once).support_independence == 1
    assert _score(H_TRANSFORM, twice).support_independence == 1


def test_contradiction_zeroes_consistency_and_applies_the_penalty() -> None:
    ev = [
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_SOURCE,
            supports=False,
            discriminating=True,
        )
    ]
    score = _score(H_SOURCE, ev)
    assert score.consistency == 0
    assert score.contradiction_penalty == CONTRADICTION_PENALTY
    assert score.total < 5


def test_non_discriminating_contradiction_carries_no_critical_penalty() -> None:
    """Only a discriminating observation is strong enough to be a *critical* contradiction."""
    ev = [
        evidence(
            tool="get_table_metrics",
            hypothesis_id=H_SOURCE,
            supports=False,
            discriminating=False,
        )
    ]
    score = _score(H_SOURCE, ev)
    assert score.consistency == 0
    assert score.contradiction_penalty == 0


def test_retrieved_context_is_worth_exactly_one_point() -> None:
    """AD-005: memory may raise a branch's priority — and that is all it may do.

    One point is enough to keep a branch worth checking, which is the whole purpose of
    FR-604. It contributes nothing to consistency or independence, so a branch carried by
    memory still has zero current support and can never satisfy FR-900.
    """
    without = _score(H_TRANSFORM, [])
    with_memory = _score(H_TRANSFORM, [], retrieval={RootCauseCategory.TRANSFORMATION_LOGIC})

    assert with_memory.retrieved_context == 1
    assert with_memory.total - without.total == 1
    assert with_memory.consistency == without.consistency
    assert with_memory.support_independence == 0


def test_no_affordable_action_removes_the_cost_point() -> None:
    score = _score(H_TRANSFORM, [], calls_remaining=0)
    assert score.cost_efficiency == 0


def test_score_stays_inside_the_rubric_range() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_table_metrics", hypothesis_id=H_TRANSFORM, round_number=2),
    ]
    score = _score(H_TRANSFORM, ev, retrieval={RootCauseCategory.TRANSFORMATION_LOGIC})
    assert CONTRADICTION_PENALTY <= score.total <= MAX_SCORE


def test_every_point_awarded_has_a_reason() -> None:
    """A score has to be auditable, not just numeric."""
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    score = _score(H_TRANSFORM, ev)
    assert len(score.reasons) >= 4
    assert any("consistency" in r for r in score.reasons)
    assert any("independence" in r for r in score.reasons)
