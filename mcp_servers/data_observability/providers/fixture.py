"""Deterministic fixture-backed Data Observability provider (FR-505, AD-006).

Reads ``data_observability.json`` from the active scenario. Implements the provider
protocol so the MCP server is agnostic to the backend.
"""

from __future__ import annotations

from ...fixture_base import FixtureProvider
from ..contracts import (
    QualityResults,
    RecentTransformationChanges,
    Reconciliation,
    SchemaChanges,
    TableMetrics,
)

FIXTURE_FILE = "data_observability.json"


class FixtureDataObservabilityProvider:
    """Concrete provider serving deterministic scenario data."""

    def __init__(self) -> None:
        self._fx = FixtureProvider(FIXTURE_FILE)

    @property
    def scenario_id(self) -> str:
        return self._fx.scenario_id

    async def get_table_metrics(self, dataset: str) -> TableMetrics:
        record = self._fx.lookup("table_metrics", dataset)
        return TableMetrics.model_validate({"dataset": dataset, **record})

    async def get_quality_results(self, dataset: str) -> QualityResults:
        record = self._fx.lookup("quality_results", dataset)
        return QualityResults.model_validate({"dataset": dataset, **record})

    async def get_schema_changes(self, dataset: str) -> SchemaChanges:
        record = self._fx.lookup("schema_changes", dataset)
        return SchemaChanges.model_validate({"dataset": dataset, **record})

    async def compare_source_and_target(
        self, source_dataset: str, target_dataset: str
    ) -> Reconciliation:
        key = f"{source_dataset}::{target_dataset}"
        record = self._fx.lookup("reconciliation", key)
        return Reconciliation.model_validate(
            {"source_dataset": source_dataset, "target_dataset": target_dataset, **record}
        )

    async def get_recent_transformation_changes(
        self, pipeline: str, dataset: str | None, max_changes: int
    ) -> RecentTransformationChanges:
        record = self._fx.lookup("transformation_changes", pipeline)
        changes = list(record.get("changes", []))
        truncated = len(changes) > max_changes
        return RecentTransformationChanges.model_validate(
            {
                "pipeline": pipeline,
                "dataset": dataset,
                "changes": changes[:max_changes],
                "truncated": truncated,
            }
        )
