"""Pipeline Operations MCP server (FR-500, FR-502, FR-504).

Separately runnable read-only server exposing pipeline execution tools. Run with:

    uv run python -m mcp_servers.pipeline_operations.server

Which scenario it serves is read from the environment (see ``fixture_base``).
Every tool is read-only; no mutating tool exists (FR-504).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from investigator.config import load_config
from investigator.schemas.results import ToolResult

from ..provider_errors import ProviderError
from ..tool_runtime import execute
from .contracts import (
    SERVER_NAME,
    SOURCE_SYSTEM,
    ExecutionLogsInput,
    PipelineRunInput,
    WatermarkInput,
)
from .providers.fixture import FIXTURE_FILE, FixturePipelineOperationsProvider

_CFG = load_config()
_TIMEOUT = _CFG.mcp.tool_timeout_seconds

provider = FixturePipelineOperationsProvider()
SCENARIO_ID = provider.scenario_id

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "READ-ONLY pipeline operations observability. Exposes run status, task "
        "failures, execution logs, upstream dependencies, and processing watermarks. "
        "No tool modifies, restarts, or backfills anything."
    ),
)


@mcp.tool(
    description="READ-ONLY: return pipeline run, schedule, duration, and terminal state."
)
async def get_pipeline_run_status(pipeline: str, run_id: str | None = None) -> ToolResult:
    return await execute(
        tool_name="get_pipeline_run_status",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=PipelineRunInput,
        raw_inputs={"pipeline": pipeline, "run_id": run_id},
        provider_call=lambda i: provider.get_pipeline_run_status(i.pipeline, i.run_id),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(
    description="READ-ONLY: return failed, skipped, retried, or blocked tasks for a run."
)
async def get_task_failures(pipeline: str, run_id: str | None = None) -> ToolResult:
    return await execute(
        tool_name="get_task_failures",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=PipelineRunInput,
        raw_inputs={"pipeline": pipeline, "run_id": run_id},
        provider_call=lambda i: provider.get_task_failures(i.pipeline, i.run_id),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(description="READ-ONLY: return bounded structured execution log events.")
async def get_execution_logs(
    pipeline: str, run_id: str | None = None, max_events: int = 50
) -> ToolResult:
    return await execute(
        tool_name="get_execution_logs",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=ExecutionLogsInput,
        raw_inputs={"pipeline": pipeline, "run_id": run_id, "max_events": max_events},
        provider_call=lambda i: provider.get_execution_logs(i.pipeline, i.run_id, i.max_events),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(description="READ-ONLY: return upstream dependency state and timing.")
async def get_upstream_dependencies(pipeline: str) -> ToolResult:
    return await execute(
        tool_name="get_upstream_dependencies",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=PipelineRunInput,
        raw_inputs={"pipeline": pipeline},
        provider_call=lambda i: provider.get_upstream_dependencies(i.pipeline),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(
    description="READ-ONLY: return the latest completed event or processing watermark."
)
async def get_processing_watermark(pipeline: str, dataset: str | None = None) -> ToolResult:
    return await execute(
        tool_name="get_processing_watermark",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=WatermarkInput,
        raw_inputs={"pipeline": pipeline, "dataset": dataset},
        provider_call=lambda i: provider.get_processing_watermark(i.pipeline, i.dataset),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


# --------------------------------------------------------------------------- #
# Resources (FR-502): context, not current operational evidence.
# --------------------------------------------------------------------------- #
@mcp.resource("runbook://pipelines/{pipeline_name}")
def runbook(pipeline_name: str) -> str:
    """Return runbook context for a pipeline (approved corpus arrives in Milestone 2)."""
    from ..fixture_base import FixtureProvider

    try:
        fx = FixtureProvider("resources.json")
        content = fx.section("runbooks").get(pipeline_name)
    except ProviderError:
        content = None
    if content:
        return str(content)
    return f"[no approved runbook for pipeline '{pipeline_name}' in this scenario]"


if __name__ == "__main__":
    mcp.run(transport="stdio")
