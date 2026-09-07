"""Data Observability MCP server (FR-501, FR-502, FR-504).

Separately runnable read-only server exposing data-quality/metric tools. Run with:

    uv run python -m mcp_servers.data_observability.server

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
    QualityResultsInput,
    ReconciliationInput,
    SchemaChangesInput,
    TableMetricsInput,
    TransformationChangesInput,
)
from .providers.fixture import FIXTURE_FILE, FixtureDataObservabilityProvider

_CFG = load_config()
_TIMEOUT = _CFG.mcp.tool_timeout_seconds

provider = FixtureDataObservabilityProvider()
SCENARIO_ID = provider.scenario_id

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "READ-ONLY data observability. Exposes table metrics, quality-test results, "
        "schema changes, source/target reconciliation, and recent transformation "
        "changes. No tool writes, repairs, or backfills data."
    ),
)


@mcp.tool(
    description="READ-ONLY: return count, freshness, null, duplicate, and distribution metrics."
)
async def get_table_metrics(dataset: str) -> ToolResult:
    return await execute(
        tool_name="get_table_metrics",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=TableMetricsInput,
        raw_inputs={"dataset": dataset},
        provider_call=lambda i: provider.get_table_metrics(i.dataset),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(description="READ-ONLY: return declared quality-test outcomes for a dataset.")
async def get_quality_results(dataset: str) -> ToolResult:
    return await execute(
        tool_name="get_quality_results",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=QualityResultsInput,
        raw_inputs={"dataset": dataset},
        provider_call=lambda i: provider.get_quality_results(i.dataset),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(description="READ-ONLY: compare schema versions and contracts for a dataset.")
async def get_schema_changes(dataset: str) -> ToolResult:
    return await execute(
        tool_name="get_schema_changes",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=SchemaChangesInput,
        raw_inputs={"dataset": dataset},
        provider_call=lambda i: provider.get_schema_changes(i.dataset),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(
    description="READ-ONLY: return reconciliation metrics across a source->target lineage edge."
)
async def compare_source_and_target(source_dataset: str, target_dataset: str) -> ToolResult:
    return await execute(
        tool_name="compare_source_and_target",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=ReconciliationInput,
        raw_inputs={"source_dataset": source_dataset, "target_dataset": target_dataset},
        provider_call=lambda i: provider.compare_source_and_target(
            i.source_dataset, i.target_dataset
        ),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


@mcp.tool(description="READ-ONLY: return bounded recent transformation metadata or diffs.")
async def get_recent_transformation_changes(
    pipeline: str, dataset: str | None = None, max_changes: int = 20
) -> ToolResult:
    return await execute(
        tool_name="get_recent_transformation_changes",
        source_system=SOURCE_SYSTEM,
        scenario_id=SCENARIO_ID,
        input_model=TransformationChangesInput,
        raw_inputs={"pipeline": pipeline, "dataset": dataset, "max_changes": max_changes},
        provider_call=lambda i: provider.get_recent_transformation_changes(
            i.pipeline, i.dataset, i.max_changes
        ),
        timeout_seconds=_TIMEOUT,
        fixture_file=FIXTURE_FILE,
    )


# --------------------------------------------------------------------------- #
# Resources (FR-502): context, not current operational evidence.
# --------------------------------------------------------------------------- #
def _read_resource(section: str, key: str, empty_note: str) -> str:
    from ..fixture_base import FixtureProvider

    try:
        fx = FixtureProvider("resources.json")
        content = fx.section(section).get(key)
    except ProviderError:
        content = None
    return str(content) if content else empty_note


@mcp.resource("schema://datasets/{dataset_name}")
def dataset_schema(dataset_name: str) -> str:
    """Return documented schema context for a dataset (context only, not evidence)."""
    return _read_resource(
        "schemas", dataset_name, f"[no schema doc for dataset '{dataset_name}' in this scenario]"
    )


@mcp.resource("incident://history/{incident_id}")
def incident_history(incident_id: str) -> str:
    """Return prior-incident context (confirmed corpus + trust gate arrive in Milestone 2)."""
    return _read_resource(
        "incidents",
        incident_id,
        f"[no confirmed historical incident '{incident_id}'; retrieval deferred to M2]",
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
