"""Unit tests for the deterministic stopping evaluator (FR-900/901/902/903)."""

from __future__ import annotations

from datetime import UTC, datetime

from investigator.graph import stopping
from investigator.schemas import Evidence, Hypothesis, RootCauseCategory


def _hyp(hid: str, category: RootCauseCategory, rank: int) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid,
        category=category,
        statement="s",
        discriminating_question="q",
        rank=rank,
    )


def _ev(
    eid: str,
    tool: str,
    supports: list[str],
    contradicts: list[str] | None = None,
    discriminating: bool = True,
    task_id: str = "T1-H1",
) -> Evidence:
    return Evidence(
        evidence_id=eid,
        investigation_id="inv",
        task_id=task_id,
        producing_agent="data_investigator",
        source_kind="current_operational",
        source_name="data_observability",
        tool_name=tool,
        collected_at=datetime(2026, 8, 2, tzinfo=UTC),
        payload={},
        summary="s",
        supports=supports,
        contradicts=contradicts or [],
        freshness_status="current",
        provenance={"discriminating": discriminating},
    )


def test_not_verified_is_not_an_incident() -> None:
    ev = stopping.evaluate(
        verification_status="not_verified",
        hypotheses=[],
        evidence=[],
        calls_used=0,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome == "not_an_incident"


def test_verification_unavailable_is_inconclusive_with_escalation() -> None:
    ev = stopping.evaluate(
        verification_status="verification_unavailable",
        hypotheses=[],
        evidence=[],
        calls_used=0,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome == "inconclusive"
    assert ev.escalation_required is True


def test_diagnosed_requires_all_clauses() -> None:
    hyps = [
        _hyp("H1", RootCauseCategory.TRANSFORMATION_LOGIC, 1),
        _hyp("H2", RootCauseCategory.SOURCE_DATA, 2),
    ]
    evidence = [
        _ev("e1", "compare_source_and_target", ["H1"], contradicts=["H2"]),
        _ev("e2", "get_recent_transformation_changes", ["H1"]),
        # competitor H2 was tested and contradicted:
        _ev("e3", "get_upstream_dependencies", [], contradicts=["H2"], task_id="T1-H2"),
    ]
    ev = stopping.evaluate(
        verification_status="verified",
        hypotheses=hyps,
        evidence=evidence,
        calls_used=3,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome == "diagnosed"
    assert ev.leading_hypothesis_id == "H1"
    assert all(ev.diagnosis_checks.values())


def test_single_support_is_inconclusive() -> None:
    hyps = [_hyp("H1", RootCauseCategory.TRANSFORMATION_LOGIC, 1)]
    evidence = [_ev("e1", "compare_source_and_target", ["H1"])]  # only one independent support
    ev = stopping.evaluate(
        verification_status="verified",
        hypotheses=hyps,
        evidence=evidence,
        calls_used=1,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome == "inconclusive"
    assert ev.diagnosis_checks["two_independent_current_supports"] is False


def test_critical_contradiction_blocks_diagnosis() -> None:
    hyps = [
        _hyp("H1", RootCauseCategory.TRANSFORMATION_LOGIC, 1),
        _hyp("H2", RootCauseCategory.SOURCE_DATA, 2),
    ]
    evidence = [
        _ev("e1", "compare_source_and_target", ["H1"]),
        _ev("e2", "get_recent_transformation_changes", ["H1"]),
        _ev("e3", "get_table_metrics", [], contradicts=["H1"], task_id="T1-H1"),
    ]
    ev = stopping.evaluate(
        verification_status="verified",
        hypotheses=hyps,
        evidence=evidence,
        calls_used=3,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome == "inconclusive"
    assert ev.diagnosis_checks["no_critical_current_contradiction"] is False


def test_agent_never_sets_outcome_only_evaluator() -> None:
    # FR-903: the evaluator is the sole producer of Evaluation.outcome.
    ev = stopping.evaluate(
        verification_status="verified",
        hypotheses=[],
        evidence=[],
        calls_used=0,
        max_calls=8,
        failures=[],
        min_supporting=2,
    )
    assert ev.outcome in {"diagnosed", "inconclusive", "not_an_incident"}
