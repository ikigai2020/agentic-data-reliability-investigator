"""Permission enforcement + discovery integration tests (FR-506, §18 acceptance).

Acceptance thresholds: unauthorized tool calls = 0%. The permissioned proxy refuses
out-of-role calls *before* dispatch, so servers never receive them.
"""

from __future__ import annotations

from investigator.mcp_client.client import InvestigatorMCPClient
from investigator.mcp_client.permissions import (
    OBSERVABILITY_TOOLS,
    PIPELINE_TOOLS,
    allowed_tools,
    is_allowed,
)

from ..conftest import DIAGNOSED


def test_permission_matrix_unit() -> None:
    # Commander and Critic have no operational tools (FR-506).
    assert allowed_tools("commander") == frozenset()
    assert allowed_tools("critic") == frozenset()
    # Specialists are confined to their own server.
    assert allowed_tools("pipeline_investigator") == PIPELINE_TOOLS
    assert allowed_tools("data_investigator") == OBSERVABILITY_TOOLS
    # Cross-domain is denied.
    assert is_allowed("data_investigator", "get_pipeline_run_status") is False
    assert is_allowed("pipeline_investigator", "get_table_metrics") is False


async def test_in_role_calls_allowed() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        pi = client.proxy("pipeline_investigator")
        r = await pi.call("get_pipeline_run_status", {"pipeline": "orders_daily"})
        assert r.status == "ok"

        di = client.proxy("data_investigator")
        r2 = await di.call("get_table_metrics", {"dataset": "orders_fact"})
        assert r2.status == "ok"


async def test_out_of_role_calls_denied() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        di = client.proxy("data_investigator")
        denied = await di.call("get_pipeline_run_status", {"pipeline": "orders_daily"})
        assert denied.status == "permission_denied"

        pi = client.proxy("pipeline_investigator")
        denied2 = await pi.call("get_table_metrics", {"dataset": "orders_fact"})
        assert denied2.status == "permission_denied"


async def test_commander_and_critic_have_no_tools() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        for role in ("commander", "critic"):
            proxy = client.proxy(role)  # type: ignore[arg-type]
            for tool in list(PIPELINE_TOOLS) + list(OBSERVABILITY_TOOLS):
                result = await proxy.call(tool, {})
                assert result.status == "permission_denied", (role, tool)
