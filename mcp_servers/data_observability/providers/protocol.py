"""Data Observability provider protocol (FR-505).

Server handlers depend on this protocol, never on a concrete fixture reader. A real
read-only warehouse/quality provider may later implement it without changing the MCP
contract (AD-006). Every method is read-only (FR-504).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import (
    QualityResults,
    RecentTransformationChanges,
    Reconciliation,
    SchemaChanges,
    TableMetrics,
)


@runtime_checkable
class DataObservabilityProvider(Protocol):
    async def get_table_metrics(self, dataset: str) -> TableMetrics: ...

    async def get_quality_results(self, dataset: str) -> QualityResults: ...

    async def get_schema_changes(self, dataset: str) -> SchemaChanges: ...

    async def compare_source_and_target(
        self, source_dataset: str, target_dataset: str
    ) -> Reconciliation: ...

    async def get_recent_transformation_changes(
        self, pipeline: str, dataset: str | None, max_changes: int
    ) -> RecentTransformationChanges: ...
