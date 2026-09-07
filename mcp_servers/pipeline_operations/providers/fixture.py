"""Deterministic fixture-backed Pipeline Operations provider (FR-505, AD-006).

Reads ``pipeline_operations.json`` from the active scenario. Implements the provider
protocol so the MCP server is agnostic to the backend.
"""

from __future__ import annotations

from ...fixture_base import FixtureProvider
from ..contracts import (
    ExecutionLogs,
    PipelineRunStatus,
    ProcessingWatermark,
    TaskFailures,
    UpstreamDependencies,
)

FIXTURE_FILE = "pipeline_operations.json"


class FixturePipelineOperationsProvider:
    """Concrete provider serving deterministic scenario data."""

    def __init__(self) -> None:
        self._fx = FixtureProvider(FIXTURE_FILE)

    @property
    def scenario_id(self) -> str:
        return self._fx.scenario_id

    async def get_pipeline_run_status(
        self, pipeline: str, run_id: str | None
    ) -> PipelineRunStatus:
        record = self._fx.lookup("pipeline_run_status", pipeline)
        return PipelineRunStatus.model_validate({"pipeline": pipeline, "run_id": run_id, **record})

    async def get_task_failures(self, pipeline: str, run_id: str | None) -> TaskFailures:
        record = self._fx.lookup("task_failures", pipeline)
        return TaskFailures.model_validate({"pipeline": pipeline, "run_id": run_id, **record})

    async def get_execution_logs(
        self, pipeline: str, run_id: str | None, max_events: int
    ) -> ExecutionLogs:
        record = self._fx.lookup("execution_logs", pipeline)
        events = list(record.get("events", []))
        truncated = len(events) > max_events
        return ExecutionLogs.model_validate(
            {
                "pipeline": pipeline,
                "run_id": run_id,
                "events": events[:max_events],
                "truncated": truncated,
            }
        )

    async def get_upstream_dependencies(self, pipeline: str) -> UpstreamDependencies:
        record = self._fx.lookup("upstream_dependencies", pipeline)
        return UpstreamDependencies.model_validate({"pipeline": pipeline, **record})

    async def get_processing_watermark(
        self, pipeline: str, dataset: str | None
    ) -> ProcessingWatermark:
        record = self._fx.lookup("processing_watermark", pipeline)
        return ProcessingWatermark.model_validate(
            {"pipeline": pipeline, "dataset": dataset, **record}
        )
