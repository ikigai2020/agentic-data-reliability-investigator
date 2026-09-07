"""Runtime enforcement + risk-gated release integration tests (v2.1: FR-1106, AD-007).

These exercise the report-release gate end to end and the FR-1106 requirement that
blocked tool attempts are measured even when no unsafe action occurs.
"""

from __future__ import annotations

import pytest

from investigator.app import run_investigation
from investigator.mcp_client.client import InvestigatorMCPClient

from ..conftest import DIAGNOSED

pytestmark = pytest.mark.asyncio


async def test_blocked_attempts_are_measured() -> None:
    # FR-1106: an out-of-role call is refused pre-dispatch and counted on the client.
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        assert client.blocked_attempts == 0
        di = client.proxy("data_investigator")
        denied = await di.call("get_pipeline_run_status", {"pipeline": "orders_daily"})
        assert denied.status == "permission_denied"
        assert client.blocked_attempts == 1
        # A second out-of-role attempt increments the safety metric again.
        await di.call("get_upstream_dependencies", {"pipeline": "orders_daily"})
        assert client.blocked_attempts == 2


async def test_diagnosed_low_impact_releases() -> None:
    state = await run_investigation(DIAGNOSED)
    assert state["outcome"] == "diagnosed"
    assert state["risk_tier"] in {"low", "medium"}
    assert state["release_status"] == "released"
    assert state["report"]["release_status"] == "released"
    assert state["report"]["risk"]["tier"] == state["risk_tier"]
    assert "blocked_tool_attempts" in state["report"]["guardrail_and_budget_events"]


async def test_high_impact_holds_diagnosis_for_human_review() -> None:
    # AD-007: even a technically sufficient diagnosis is held for human authority when
    # the deterministic risk tier is high.
    state = await run_investigation(DIAGNOSED, risk_overrides={"high_impact": True})
    assert state["outcome"] == "diagnosed"  # technical result unchanged
    assert state["risk_tier"] == "high"
    assert state["release_status"] == "human_review_required"
    assert state["escalation_required"] is True
    assert state["escalation_package"] is not None


async def test_prohibited_blocks_release() -> None:
    state = await run_investigation(DIAGNOSED, risk_overrides={"blocked": True})
    assert state["risk_tier"] == "prohibited"
    assert state["release_status"] == "blocked"
