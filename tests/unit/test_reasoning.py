"""Unit tests for the deterministic reasoning engine (FR-401, FR-1304 determinism)."""

from __future__ import annotations

from datetime import UTC, datetime

from investigator.agents.reasoning import DeterministicReasoning
from investigator.mcp_client.permissions import allowed_tools
from investigator.schemas import Alert, RootCauseCategory


def _alert(symptom: str = "volume_drop") -> Alert:
    return Alert(
        incident_id="INC-1",
        dataset="orders_fact",
        pipeline="orders_daily",
        symptom_type=symptom,  # type: ignore[arg-type]
        observed_value=100,
        expected_value=500,
        window_start=datetime(2026, 8, 1, tzinfo=UTC),
        window_end=datetime(2026, 8, 2, tzinfo=UTC),
        detected_at=datetime(2026, 8, 2, tzinfo=UTC),
        severity="high",
    )


async def test_hypotheses_count_within_bounds() -> None:
    r = DeterministicReasoning()
    for symptom in ("volume_drop", "freshness_delay", "pipeline_failure", "schema_change"):
        hyps = await r.generate_hypotheses(_alert(symptom), max_count=5)
        assert 3 <= len(hyps) <= 5, symptom
        # distinct categories, ordinal ranks 1..n, taxonomy only
        cats = [h.category for h in hyps]
        assert len(set(cats)) == len(cats)
        assert [h.rank for h in hyps] == list(range(1, len(hyps) + 1))
        assert all(isinstance(h.category, RootCauseCategory) for h in hyps)


async def test_hypotheses_are_deterministic() -> None:
    r = DeterministicReasoning()
    a = _alert()
    first = [h.hypothesis_id for h in await r.generate_hypotheses(a, 5)]
    second = [h.hypothesis_id for h in await r.generate_hypotheses(a, 5)]
    assert first == second


def test_task_plans_respect_permissions() -> None:
    r = DeterministicReasoning()
    for category in RootCauseCategory:
        role, tools = r.plan_for_category(category)
        permitted = allowed_tools(role)
        assert set(tools).issubset(permitted), category


async def test_interpret_reconciliation_filter_drop() -> None:
    r = DeterministicReasoning()
    interp = await r.interpret(
        "compare_source_and_target",
        {"source_count": 500, "target_count": 100, "rows_dropped_by_filter": 400},
    )
    assert RootCauseCategory.TRANSFORMATION_LOGIC in interp.supports
    assert RootCauseCategory.SOURCE_DATA in interp.contradicts
    assert interp.discriminating is True


async def test_interpret_run_success_contradicts_orchestration() -> None:
    r = DeterministicReasoning()
    interp = await r.interpret("get_pipeline_run_status", {"state": "success"})
    assert RootCauseCategory.ORCHESTRATION in interp.contradicts


async def test_interpret_no_change_contradicts_transformation() -> None:
    r = DeterministicReasoning()
    interp = await r.interpret("get_recent_transformation_changes", {"changes": []})
    assert RootCauseCategory.TRANSFORMATION_LOGIC in interp.contradicts
