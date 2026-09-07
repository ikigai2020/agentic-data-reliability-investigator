"""Beam search: pruning, diversity, reopening, tie resolution (FR-702, FR-704..706)."""

from __future__ import annotations

from investigator.search import beam
from investigator.search.beam import BeamPolicy

from .fixtures import H_ORCH, H_QUALITY, H_SOURCE, H_TRANSFORM, branches, evidence, hypotheses

POLICY = BeamPolicy()


def _apply(brs, ev, *, policy: BeamPolicy = POLICY, calls_remaining: int = 8, unavailable=None):
    return beam.apply_policy(
        brs,
        hypotheses(),
        ev,
        policy=policy,
        calls_remaining=calls_remaining,
        unavailable_tools=unavailable,
    )


def _branch(decision, hypothesis_id: str):
    return next(b for b in decision.branches if b.hypothesis_id == hypothesis_id)


# --------------------------------------------------------------------------- #
# FR-701 branch bookkeeping
# --------------------------------------------------------------------------- #
def test_depth_counts_evidence_gathering_cycles_not_tool_calls() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM, round_number=1),
        evidence(
            tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM, round_number=1
        ),
        evidence(tool="get_table_metrics", hypothesis_id=H_TRANSFORM, round_number=2),
    ]
    advanced = beam.advance_branches(branches(), ev)
    transform = next(b for b in advanced if b.hypothesis_id == H_TRANSFORM)
    assert len(transform.action_history) == 3  # three tool calls
    assert transform.depth == 2  # across two rounds
    assert len(transform.evidence_ids) == 3


def test_untouched_branch_stays_at_depth_zero() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    advanced = beam.advance_branches(branches(), ev)
    quality = next(b for b in advanced if b.hypothesis_id == H_QUALITY)
    assert quality.depth == 0
    assert quality.action_history == []


# --------------------------------------------------------------------------- #
# FR-704 pruning
# --------------------------------------------------------------------------- #
def test_falsified_branch_is_pruned_with_that_reason() -> None:
    ev = [
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_SOURCE,
            supports=False,
            discriminating=True,
        )
    ]
    decision = _apply(beam.advance_branches(branches(), ev), ev)
    assert _branch(decision, H_SOURCE).status == "pruned"
    assert "falsified" in decision.pruned[f"B-{H_SOURCE}"]


def test_low_scoring_branch_is_pruned_below_the_threshold() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_pipeline_run_status", hypothesis_id=H_ORCH, discriminating=False),
        evidence(tool="get_task_failures", hypothesis_id=H_ORCH, discriminating=False),
    ]
    decision = _apply(beam.advance_branches(branches(), ev), ev)
    # The transformation branch has two independent supports and survives.
    assert _branch(decision, H_TRANSFORM).status == "selected"


def test_exhausted_budget_prunes_every_live_branch() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    decision = _apply(beam.advance_branches(branches(), ev), ev, calls_remaining=0)
    assert decision.selected_ids == []
    assert all("budget" in reason for reason in decision.pruned.values())


def test_no_branch_reopens_once_the_budget_is_gone() -> None:
    """Reopening costs calls; with none left there is nothing to reopen a branch for."""
    pruned = [
        b.model_copy(update={"status": "pruned", "prune_reason": "score 2 below threshold 5"})
        if b.hypothesis_id == H_QUALITY
        else b
        for b in branches()
    ]
    ev = [evidence(tool="get_quality_results", hypothesis_id=H_QUALITY)]
    decision = _apply(pruned, ev, calls_remaining=0)
    assert decision.reopened == {}
    assert _branch(decision, H_QUALITY).status == "pruned"


def test_branch_exceeding_max_depth_is_pruned() -> None:
    deep = [b.model_copy(update={"depth": 4}) for b in branches()]
    decision = _apply(deep, [])
    assert all(b.status in {"pruned", "closed"} for b in decision.branches)
    assert any("depth" in reason for reason in decision.pruned.values())


def test_branch_depending_on_unknown_evidence_is_pruned() -> None:
    """A branch may not rest on evidence that is not in accepted state (FR-1002)."""
    tampered = [
        b.model_copy(update={"evidence_ids": ["ev_does_not_exist"]})
        if b.hypothesis_id == H_TRANSFORM
        else b
        for b in branches()
    ]
    decision = _apply(tampered, [])
    assert _branch(decision, H_TRANSFORM).status == "pruned"
    assert "not in accepted state" in decision.pruned[f"B-{H_TRANSFORM}"]


def test_branch_with_no_remaining_action_is_closed_not_pruned() -> None:
    """Running out of checks is exhaustion, not failure — the distinction is recorded."""
    ev = [
        evidence(tool="get_schema_changes", hypothesis_id=H_QUALITY),
        evidence(tool="get_quality_results", hypothesis_id=H_QUALITY),
        evidence(tool="get_table_metrics", hypothesis_id=H_QUALITY),
        evidence(tool="compare_source_and_target", hypothesis_id=H_QUALITY),
    ]
    decision = _apply(beam.advance_branches(branches(), ev), ev)
    quality = _branch(decision, H_QUALITY)
    assert quality.status == "closed"
    assert decision.closed[quality.branch_id] == "candidate actions exhausted"


def test_unavailable_tool_without_alternative_is_pruned() -> None:
    # Non-discriminating results, so the branch is unresolved rather than falsified —
    # otherwise falsification would be the reason and the tool gap would go unrecorded.
    ev = [
        evidence(tool=tool, hypothesis_id=H_QUALITY, supports=False, discriminating=False)
        for tool in (
            "get_schema_changes",
            "get_quality_results",
            "get_table_metrics",
            "compare_source_and_target",
        )
    ]
    advanced = beam.advance_branches(branches(), ev)
    decision = _apply(advanced, ev, unavailable={"get_quality_results"})
    quality = _branch(decision, H_QUALITY)
    assert quality.status == "pruned"
    assert "unavailable without alternative" in (quality.prune_reason or "")


def test_duplicate_tested_path_is_pruned() -> None:
    same_path = [
        b.model_copy(update={"action_history": ["get_table_metrics"]}) for b in branches()
    ]
    decision = _apply(same_path, [])
    duplicates = [r for r in decision.pruned.values() if "duplicates the tested path" in r]
    assert duplicates, decision.pruned


# --------------------------------------------------------------------------- #
# FR-704 diversity
# --------------------------------------------------------------------------- #
def test_two_diverse_hypotheses_survive_until_each_is_tested() -> None:
    """No evidence yet: every branch scores low, but the beam must not empty itself."""
    decision = _apply(branches(), [])
    survivors = [b for b in decision.branches if b.status in beam.LIVE_STATUSES]
    assert len({b.hypothesis_id for b in survivors}) >= POLICY.min_diverse_branches
    assert decision.protected_ids


def test_diversity_never_rescues_a_falsified_branch() -> None:
    """A hypothesis that was tested and refuted stays dead, however thin the beam gets."""
    ev = [
        evidence(
            tool=tool,
            hypothesis_id=hyp,
            supports=False,
            discriminating=True,
        )
        for tool, hyp in [
            ("compare_source_and_target", H_SOURCE),
            ("get_pipeline_run_status", H_ORCH),
            ("get_quality_results", H_QUALITY),
        ]
    ]
    decision = _apply(beam.advance_branches(branches(), ev), ev)
    for hyp in (H_SOURCE, H_ORCH, H_QUALITY):
        assert _branch(decision, hyp).status == "pruned"
        assert f"B-{hyp}" not in decision.protected_ids


def test_protection_records_what_would_have_happened() -> None:
    decision = _apply(branches(), [])
    protected = _branch(decision, H_TRANSFORM)
    if protected.branch_id in decision.protected_ids:
        assert any("diversity hold" in entry for entry in protected.action_history)
        assert protected.prune_reason is None


# --------------------------------------------------------------------------- #
# FR-705 reopening
# --------------------------------------------------------------------------- #
def test_pruned_branch_reopens_when_new_evidence_supports_it() -> None:
    pruned = [
        b.model_copy(update={"status": "pruned", "prune_reason": "score 2 below threshold 5"})
        if b.hypothesis_id == H_QUALITY
        else b
        for b in branches()
    ]
    ev = [
        evidence(tool="get_quality_results", hypothesis_id=H_QUALITY),
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
    ]
    decision = _apply(pruned, ev)
    quality = _branch(decision, H_QUALITY)
    assert quality.status == "reopened"
    assert quality.branch_id in decision.selected_ids  # reopened *and* back in the beam
    trigger = decision.reopened[quality.branch_id]
    assert "supports this hypothesis" in trigger
    assert "prior prune: score 2 below threshold 5" in trigger  # FR-705 records both


def test_reopen_trigger_and_prior_reason_land_in_the_branch_history() -> None:
    pruned = [
        b.model_copy(update={"status": "pruned", "prune_reason": "score 1 below threshold 5"})
        if b.hypothesis_id == H_QUALITY
        else b
        for b in branches()
    ]
    ev = [evidence(tool="get_quality_results", hypothesis_id=H_QUALITY)]
    decision = _apply(pruned, ev)
    history = _branch(decision, H_QUALITY).action_history
    assert any("reopened:" in entry and "prior prune:" in entry for entry in history)


def test_falsified_branch_never_reopens() -> None:
    pruned = [
        b.model_copy(update={"status": "pruned", "prune_reason": "directly falsified"})
        if b.hypothesis_id == H_SOURCE
        else b
        for b in branches()
    ]
    ev = [
        evidence(
            tool="compare_source_and_target",
            hypothesis_id=H_SOURCE,
            supports=False,
            discriminating=True,
        ),
        # A later non-discriminating reading nominally "supports" it.
        evidence(
            tool="get_processing_watermark",
            hypothesis_id=H_SOURCE,
            round_number=2,
            discriminating=False,
        ),
    ]
    decision = _apply(pruned, ev)
    assert _branch(decision, H_SOURCE).status == "pruned"
    assert f"B-{H_SOURCE}" not in decision.reopened


# --------------------------------------------------------------------------- #
# FR-702 width and FR-706 ties
# --------------------------------------------------------------------------- #
def test_beam_width_caps_the_number_of_expanded_branches() -> None:
    narrow = BeamPolicy(beam_width=1)
    decision = _apply(branches(), [], policy=narrow)
    assert len(decision.selected_ids) <= 1


def test_ties_resolve_by_evidence_then_stable_branch_id() -> None:
    """FR-706: identical scores must order deterministically, never by dict order."""
    decision_a = _apply(branches(), [])
    decision_b = _apply(list(reversed(branches())), [])
    assert decision_a.selected_ids == decision_b.selected_ids


def test_stronger_current_evidence_wins_a_tie() -> None:
    ev = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]
    decision = _apply(beam.advance_branches(branches(), ev), ev)
    assert decision.selected_ids[0] == f"B-{H_TRANSFORM}"


def test_policy_is_deterministic_across_repeated_runs() -> None:
    ev = [evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM)]
    advanced = beam.advance_branches(branches(), ev)
    first = _apply(advanced, ev)
    second = _apply(advanced, ev)
    assert first.selected_ids == second.selected_ids
    assert first.pruned == second.pruned
    assert {b: s.total for b, s in first.scores.items()} == {
        b: s.total for b, s in second.scores.items()
    }


# --------------------------------------------------------------------------- #
# FR-508 degradation — the beam and the planner must agree on what is reachable
# --------------------------------------------------------------------------- #
def test_a_failed_tool_is_not_offered_as_an_untried_candidate() -> None:
    """The Commander excludes failed tools when planning; the beam must do the same.

    Otherwise the beam calls a branch expandable, the next round plans nothing for it,
    and the loop spends its rounds on a check that can never happen.
    """
    quality_tools = {"get_quality_results", "get_table_metrics", "compare_source_and_target"}
    blocked = _apply(branches(), [], unavailable=quality_tools)
    assert blocked.thoughts.get(f"B-{H_QUALITY}") is None


def test_a_branch_denied_every_tool_is_recorded_as_blocked_not_exhausted() -> None:
    """Under a server outage a branch has no action history at all.

    Reporting that as "candidate actions exhausted" would hide the outage behind a
    routine disposition, and the escalation package would understate what was missed.
    """
    source_tools = {
        "get_upstream_dependencies",
        "get_processing_watermark",
        "get_pipeline_run_status",
    }
    decision = _apply(branches(), [], unavailable=source_tools)
    source = _branch(decision, H_SOURCE)

    assert source.status == "pruned"
    assert "required tool unavailable" in (source.prune_reason or "")
    assert sorted(source_tools)[0] in (source.prune_reason or "")


def test_a_blocked_branch_is_not_revived_by_diversity_protection() -> None:
    """Diversity protects untested hypotheses, not unreachable ones."""
    source_tools = {
        "get_upstream_dependencies",
        "get_processing_watermark",
        "get_pipeline_run_status",
    }
    decision = _apply(branches(), [], unavailable=source_tools)
    assert f"B-{H_SOURCE}" not in decision.protected_ids
