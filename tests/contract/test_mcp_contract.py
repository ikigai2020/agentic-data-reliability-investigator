"""MCP contract tests (FR-509).

Proves: both servers start independently; discovery returns expected tools; schemas
validate; bad input yields structured errors; timeouts/outages are handled; and
providers can be swapped without changing contracts (envelope stability).
"""

from __future__ import annotations

import pytest

from investigator.mcp_client.client import InvestigatorMCPClient
from investigator.mcp_client.permissions import (
    OBSERVABILITY_SERVER,
    OBSERVABILITY_TOOLS,
    PIPELINE_SERVER,
    PIPELINE_TOOLS,
)
from investigator.schemas.results import ToolResult

from ..conftest import DIAGNOSED

PIPELINE_MODULE = "mcp_servers.pipeline_operations.server"
OBS_MODULE = "mcp_servers.data_observability.server"

pytestmark = pytest.mark.asyncio


async def test_pipeline_server_starts_independently() -> None:
    async with InvestigatorMCPClient(
        DIAGNOSED, server_modules={PIPELINE_SERVER: PIPELINE_MODULE}
    ) as client:
        assert not client.unavailable_servers
        assert set(client.discovered_tools[PIPELINE_SERVER]) == set(PIPELINE_TOOLS)
        assert OBSERVABILITY_SERVER not in client.discovered_tools


async def test_observability_server_starts_independently() -> None:
    async with InvestigatorMCPClient(
        DIAGNOSED, server_modules={OBSERVABILITY_SERVER: OBS_MODULE}
    ) as client:
        assert not client.unavailable_servers
        assert set(client.discovered_tools[OBSERVABILITY_SERVER]) == set(OBSERVABILITY_TOOLS)


async def test_discovery_returns_both_servers() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        tools = client.discovered_tools
        assert set(tools[PIPELINE_SERVER]) == set(PIPELINE_TOOLS)
        assert set(tools[OBSERVABILITY_SERVER]) == set(OBSERVABILITY_TOOLS)


async def test_tool_schemas_validate() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        # Every discovered tool advertises a JSON-schema object with properties (FR-509).
        for server, tools in client._tools.items():  # noqa: SLF001 - test introspection
            for name, schema in tools.items():
                assert isinstance(schema, dict), (server, name)
                assert schema.get("type") == "object"
                assert "properties" in schema


async def test_result_envelope_is_stable() -> None:
    # FR-503: uniform envelope regardless of tool -> provider swap safe (AD-006).
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        di = client.proxy("data_investigator")
        result = await di.call("get_table_metrics", {"dataset": "orders_fact"})
    assert isinstance(result, ToolResult)
    for field in (
        "request_id",
        "tool_name",
        "source_system",
        "scenario_id",
        "collected_at",
        "status",
        "data",
        "provenance",
        "warnings",
    ):
        assert hasattr(result, field)
    assert result.status == "ok"
    assert result.provenance["read_only"] is True


async def test_bad_input_yields_structured_error() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        pi = client.proxy("pipeline_investigator")
        # max_events exceeds the contract bound (le=500) -> structured invalid_input.
        result = await pi.call(
            "get_execution_logs", {"pipeline": "orders_daily", "max_events": 99999}
        )
    assert result.status == "invalid_input"
    assert result.error_message


async def test_no_data_is_distinct_from_error() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        di = client.proxy("data_investigator")
        result = await di.call("get_quality_results", {"dataset": "does_not_exist"})
    assert result.status == "no_data"
    assert result.is_evidence is True  # observed absence, not a failure (FR-508)


async def test_tool_not_found() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        result = await client.call_tool(OBSERVABILITY_SERVER, "get_nonexistent_tool", {})
    assert result.status == "tool_not_found"


async def test_server_unavailable_handled() -> None:
    # Point one server at a bogus module -> discovery records it unavailable, calls
    # return server_unavailable, and the other server is unaffected (FR-508, FR-204).
    async with InvestigatorMCPClient(
        DIAGNOSED,
        server_modules={
            PIPELINE_SERVER: "mcp_servers.pipeline_operations.does_not_exist",
            OBSERVABILITY_SERVER: OBS_MODULE,
        },
    ) as client:
        assert PIPELINE_SERVER in client.unavailable_servers
        result = await client.call_tool(
            PIPELINE_SERVER, "get_pipeline_run_status", {"pipeline": "x"}
        )
        assert result.status == "server_unavailable"
        ok = await client.call_tool(
            OBSERVABILITY_SERVER, "get_table_metrics", {"dataset": "orders_fact"}
        )
        assert ok.status == "ok"


async def test_provider_unavailable_handled() -> None:
    async with InvestigatorMCPClient(
        DIAGNOSED, extra_env={"INVESTIGATOR_FORCE_PROVIDER_UNAVAILABLE": "1"}
    ) as client:
        di = client.proxy("data_investigator")
        result = await di.call("get_table_metrics", {"dataset": "orders_fact"})
    assert result.status == "provider_unavailable"
    assert result.is_evidence is False  # FR-508: not counted against a hypothesis
