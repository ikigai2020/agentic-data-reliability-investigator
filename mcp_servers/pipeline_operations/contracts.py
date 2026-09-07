"""Typed tool input/output contracts for the Pipeline Operations MCP server (FR-500).

Output models describe the ``data`` payload carried inside the shared ``ToolResult``
envelope (FR-503). All tools are read-only (FR-504); no model here can express a
mutation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SERVER_NAME = "pipeline-operations-mcp"
SOURCE_SYSTEM = "pipeline_operations"

RunState = Literal["success", "failed", "running", "skipped", "no_run"]
TaskState = Literal["failed", "skipped", "retried", "blocked"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class PipelineRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    run_id: str | None = None


class ExecutionLogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    run_id: str | None = None
    max_events: int = Field(default=50, ge=1, le=500)


class WatermarkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    dataset: str | None = None


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
class PipelineRunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    run_id: str | None
    schedule: str | None = None
    scheduled_start: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    duration_seconds: float | None = None
    state: RunState
    is_terminal: bool


class TaskFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: str
    state: TaskState
    attempts: int = 1
    error_type: str | None = None
    message: str | None = None


class TaskFailures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    run_id: str | None
    failures: list[TaskFailure] = Field(default_factory=list)


class LogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    task: str | None = None
    message: str


class ExecutionLogs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    run_id: str | None
    events: list[LogEvent] = Field(default_factory=list)
    truncated: bool = False


class UpstreamDependency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    state: RunState
    last_success: datetime | None = None
    expected_by: datetime | None = None
    is_late: bool = False


class UpstreamDependencies(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    dependencies: list[UpstreamDependency] = Field(default_factory=list)


class ProcessingWatermark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline: str
    dataset: str | None = None
    watermark: datetime | None = None
    latest_event_time: datetime | None = None
    lag_seconds: float | None = None
