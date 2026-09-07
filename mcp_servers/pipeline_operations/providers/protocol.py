"""Pipeline Operations provider protocol (FR-505).

Server handlers depend on this protocol, never on a concrete fixture reader. A real
read-only Airflow/Dagster provider may later implement it without changing the MCP
contract (AD-006). Every method is read-only (FR-504).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import (
    ExecutionLogs,
    PipelineRunStatus,
    ProcessingWatermark,
    TaskFailures,
    UpstreamDependencies,
)


@runtime_checkable
class PipelineOperationsProvider(Protocol):
    async def get_pipeline_run_status(
        self, pipeline: str, run_id: str | None
    ) -> PipelineRunStatus: ...

    async def get_task_failures(self, pipeline: str, run_id: str | None) -> TaskFailures: ...

    async def get_execution_logs(
        self, pipeline: str, run_id: str | None, max_events: int
    ) -> ExecutionLogs: ...

    async def get_upstream_dependencies(self, pipeline: str) -> UpstreamDependencies: ...

    async def get_processing_watermark(
        self, pipeline: str, dataset: str | None
    ) -> ProcessingWatermark: ...
