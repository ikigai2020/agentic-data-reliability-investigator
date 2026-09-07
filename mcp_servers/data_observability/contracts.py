"""Typed tool input/output contracts for the Data Observability MCP server (FR-501).

Output models describe the ``data`` payload carried inside the shared ``ToolResult``
envelope (FR-503). All tools are read-only (FR-504).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SERVER_NAME = "data-observability-mcp"
SOURCE_SYSTEM = "data_observability"

QualityStatus = Literal["pass", "fail", "error"]
SchemaChangeType = Literal["added", "removed", "type_changed", "nullability_changed"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class TableMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str


class QualityResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str


class SchemaChangesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str


class ReconciliationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_dataset: str
    target_dataset: str


class TransformationChangesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    dataset: str | None = None
    max_changes: int = Field(default=20, ge=1, le=200)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
class TableMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    row_count: int | None = None
    expected_row_count: int | None = None
    freshness_seconds: float | None = None
    last_loaded_at: datetime | None = None
    null_rate: dict[str, float] = Field(default_factory=dict)
    duplicate_count: int | None = None
    distribution: dict[str, float] = Field(default_factory=dict)


class QualityTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_name: str
    column: str | None = None
    status: QualityStatus
    observed_value: float | int | None = None
    threshold: float | int | None = None


class QualityResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    results: list[QualityTestResult] = Field(default_factory=list)


class SchemaChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: str
    change_type: SchemaChangeType
    before: str | None = None
    after: str | None = None


class SchemaChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    from_version: str | None = None
    to_version: str | None = None
    changes: list[SchemaChange] = Field(default_factory=list)
    breaking: bool = False


class Reconciliation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_dataset: str
    target_dataset: str
    source_count: int | None = None
    target_count: int | None = None
    difference: int | None = None
    difference_pct: float | None = None
    rows_dropped_by_filter: int | None = None
    notes: str | None = None


class TransformationChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    change_id: str
    author: str | None = None
    committed_at: datetime | None = None
    summary: str
    affected_columns: list[str] = Field(default_factory=list)
    diff_summary: str | None = None


class RecentTransformationChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    dataset: str | None = None
    changes: list[TransformationChange] = Field(default_factory=list)
    truncated: bool = False
