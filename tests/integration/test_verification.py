"""Integration tests for MCP-based verification (FR-803)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from investigator.graph.verification import verify_alert
from investigator.mcp_client.client import InvestigatorMCPClient
from investigator.schemas import Alert

from ..conftest import DIAGNOSED

pytestmark = pytest.mark.asyncio


def _alert(**overrides) -> Alert:
    base = dict(
        incident_id="INC-1001",
        dataset="orders_fact",
        pipeline="orders_daily",
        symptom_type="volume_drop",
        observed_value=12000,
        expected_value=50000,
        window_start=datetime(2026, 8, 15, tzinfo=UTC),
        window_end=datetime(2026, 8, 16, tzinfo=UTC),
        detected_at=datetime(2026, 8, 16, 2, tzinfo=UTC),
        severity="high",
    )
    base.update(overrides)
    return Alert(**base)  # type: ignore[arg-type]


async def test_volume_drop_is_verified() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        status, detail, evidence = await verify_alert(
            _alert(), client.proxy, tolerance=0.05, investigation_id="inv"
        )
    assert status == "verified"
    assert detail["tool_name"] == "get_table_metrics"
    assert evidence is not None
    assert evidence.source_kind == "current_operational"


async def test_absent_condition_is_not_verified() -> None:
    # orders_fact_source has 50000 rows == expected -> no drop -> not_verified.
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        status, _detail, _ev = await verify_alert(
            _alert(dataset="orders_fact_source"),
            client.proxy,
            tolerance=0.05,
            investigation_id="inv",
        )
    assert status == "not_verified"


async def test_missing_data_is_verification_unavailable() -> None:
    async with InvestigatorMCPClient(DIAGNOSED) as client:
        status, detail, evidence = await verify_alert(
            _alert(dataset="ghost_dataset"),
            client.proxy,
            tolerance=0.05,
            investigation_id="inv",
        )
    assert status == "verification_unavailable"
    assert evidence is None
