"""Parallel specialist dispatch (FR-203, FR-204, FR-205).

These exercise ``dispatch_specialists`` directly with stub specialist subgraphs so that
concurrency, budget admission, failure isolation, and merge order can be observed
without depending on fixture timing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from investigator.config import load_config
from investigator.graph import parent_graph
from investigator.schemas.evidence import Evidence
from investigator.schemas.finding import AgentFinding
from investigator.schemas.task import InvestigationTask

from ..search.fixtures import H_ORCH, H_SOURCE, H_TRANSFORM, hypotheses, orders_alert

NOW = datetime(2026, 8, 27, tzinfo=UTC)


class _Proxy:
    role = "data_investigator"


class _Client:
    blocked_attempts = 0

    def proxy(self, role):  # noqa: ARG002 - stub
        return _Proxy()


def _task(hypothesis_id: str, agent: str, tools: list[str], priority: int) -> InvestigationTask:
    return InvestigationTask(
        task_id=f"T1-{hypothesis_id}",
        round_number=1,
        assigned_agent=agent,  # type: ignore[arg-type]
        hypothesis_ids=[hypothesis_id],
        question="q",
        expected_discriminating_value="v",
        allowed_tool_names=tools,
        priority=priority,
        status="pending",
    )


def _state(tasks, *, calls_used: int = 0, max_calls: int = 8) -> dict:
    return {
        "investigation_id": "inv_test",
        "trace_id": "trace_test",
        "round_number": 1,
        "alert": orders_alert(),
        "hypotheses": hypotheses(),
        "tasks": tasks,
        "budgets": {
            "max_operational_calls": max_calls,
            "max_rounds": 4,
            "calls_used": calls_used,
        },
        "failures": [],
    }


def _config() -> dict:
    return {"configurable": {"client": _Client(), "app_config": load_config()}}


def _evidence(task_id: str, tool: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev_{task_id}_{tool}",
        investigation_id="inv_test",
        task_id=task_id,
        producing_agent="stub",
        source_kind="current_operational",
        source_name="stub_server",
        tool_name=tool,
        observed_at=NOW,
        collected_at=NOW,
        summary=f"{tool} observation",
        supports=[],
        contradicts=[],
        freshness_status="current",
        provenance={"request_id": "req_stub", "discriminating": True},
    )


class _StubSubgraph:
    """Records concurrency and consumes its private budget slice."""

    def __init__(self, tracker: dict, *, delay: float = 0.0, raises: bool = False) -> None:
        self._tracker = tracker
        self._delay = delay
        self._raises = raises

    async def ainvoke(self, state, config):
        task = state["task"]
        budget = config["configurable"]["budget"]
        self._tracker["live"] += 1
        self._tracker["peak"] = max(self._tracker["peak"], self._tracker["live"])
        try:
            await asyncio.sleep(self._delay)
            if self._raises:
                raise RuntimeError(f"{task.task_id} exploded")
            tools = task.allowed_tool_names[: budget["max"]]
            budget["calls_used"] = len(tools)
            self._tracker["completed"].append(task.task_id)
            return {
                "evidence": [_evidence(task.task_id, t) for t in tools],
                "finding": AgentFinding(
                    task_id=task.task_id,
                    agent_name="stub",
                    conclusion="done",
                    evidence_ids=[f"ev_{task.task_id}_{t}" for t in tools],
                    completion_status="complete",
                ),
                "failures": [],
            }
        finally:
            self._tracker["live"] -= 1


@pytest.fixture
def tracker() -> dict:
    return {"live": 0, "peak": 0, "completed": []}


def _install(monkeypatch, mapping: dict) -> None:
    monkeypatch.setattr(parent_graph, "_SPECIALIST_SUBGRAPHS", mapping)


async def test_independent_specialists_run_concurrently(monkeypatch, tracker) -> None:
    """FR-203: two independent tasks overlap rather than queueing."""
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker, delay=0.05),
            "pipeline_investigator": _StubSubgraph(tracker, delay=0.05),
        },
    )
    tasks = [
        _task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0),
        _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1),
        _task(H_ORCH, "pipeline_investigator", ["get_pipeline_run_status"], 2),
    ]
    await parent_graph.dispatch_specialists(_state(tasks), _config())
    assert tracker["peak"] > 1, "specialists were serialised"


async def test_results_merge_by_task_order_not_completion_order(monkeypatch, tracker) -> None:
    """FR-203: merged by stable IDs, so a slow specialist does not reorder the evidence."""
    slow = _StubSubgraph(tracker, delay=0.06)
    fast = _StubSubgraph(tracker, delay=0.0)
    _install(monkeypatch, {"data_investigator": slow, "pipeline_investigator": fast})

    tasks = [
        _task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0),  # slow, first
        _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1),  # fast, second
    ]
    result = await parent_graph.dispatch_specialists(_state(tasks), _config())

    # The fast task finished first ...
    assert tracker["completed"][0] == f"T1-{H_SOURCE}"
    # ... but the merged evidence still follows task priority.
    assert [e.task_id for e in result["evidence"]] == [f"T1-{H_TRANSFORM}", f"T1-{H_SOURCE}"]
    assert [f.task_id for f in result["findings"]] == [f"T1-{H_TRANSFORM}", f"T1-{H_SOURCE}"]


async def test_one_specialist_failing_does_not_lose_the_others_evidence(
    monkeypatch, tracker
) -> None:
    """FR-204: the failure is recorded, completed evidence survives, the round continues."""
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker),
            "pipeline_investigator": _StubSubgraph(tracker, raises=True),
        },
    )
    tasks = [
        _task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0),
        _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1),
    ]
    result = await parent_graph.dispatch_specialists(_state(tasks), _config())

    assert [e.task_id for e in result["evidence"]] == [f"T1-{H_TRANSFORM}"]
    assert any("exploded" in f for f in result["failures"])
    statuses = {t.task_id: t.status for t in result["tasks"]}
    assert statuses[f"T1-{H_TRANSFORM}"] == "completed"
    assert statuses[f"T1-{H_SOURCE}"] == "failed"


async def test_a_failed_specialist_does_not_consume_budget(monkeypatch, tracker) -> None:
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker),
            "pipeline_investigator": _StubSubgraph(tracker, raises=True),
        },
    )
    tasks = [
        _task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0),
        _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1),
    ]
    result = await parent_graph.dispatch_specialists(_state(tasks), _config())
    assert result["budgets"]["calls_used"] == 1  # only the successful task's call


async def test_parallel_group_never_exceeds_the_remaining_budget(monkeypatch, tracker) -> None:
    """FR-203: combined expected calls of the group must fit the global budget."""
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker),
            "pipeline_investigator": _StubSubgraph(tracker),
        },
    )
    tasks = [
        _task(
            H_TRANSFORM,
            "data_investigator",
            ["compare_source_and_target", "get_table_metrics"],
            0,
        ),
        _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1),
        _task(H_ORCH, "pipeline_investigator", ["get_pipeline_run_status"], 2),
    ]
    # Only three calls left: the first task takes two, the second one, the third defers.
    result = await parent_graph.dispatch_specialists(_state(tasks, calls_used=5), _config())

    assert result["budgets"]["calls_used"] == 8
    dispatched = {t.task_id for t in result["tasks"]}
    assert f"T1-{H_ORCH}" not in dispatched


async def test_no_dispatch_when_the_budget_is_already_spent(monkeypatch, tracker) -> None:
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker),
            "pipeline_investigator": _StubSubgraph(tracker),
        },
    )
    tasks = [_task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0)]
    result = await parent_graph.dispatch_specialists(_state(tasks, calls_used=8), _config())
    assert result["evidence"] == []
    assert result["tasks"] == []
    assert tracker["peak"] == 0


async def test_only_the_current_rounds_pending_tasks_are_dispatched(monkeypatch, tracker) -> None:
    """FR-205: a completed task is never silently re-run when a later round dispatches."""
    _install(
        monkeypatch,
        {
            "data_investigator": _StubSubgraph(tracker),
            "pipeline_investigator": _StubSubgraph(tracker),
        },
    )
    done = _task(H_TRANSFORM, "data_investigator", ["compare_source_and_target"], 0).model_copy(
        update={"status": "completed"}
    )
    pending = _task(H_SOURCE, "pipeline_investigator", ["get_upstream_dependencies"], 1)
    result = await parent_graph.dispatch_specialists(_state([done, pending]), _config())

    assert tracker["completed"] == [f"T1-{H_SOURCE}"]
    assert [t.task_id for t in result["tasks"]] == [f"T1-{H_SOURCE}"]
