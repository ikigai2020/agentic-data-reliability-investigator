"""Malformed or underspecified alert input (FR-300, FR-801, FR-1301 #10).

The alert is untrusted data. A bad one must stop at the door with a typed outcome, not
propagate into hypothesis generation where a model would be asked to make sense of it.
"""

from __future__ import annotations

from investigator.config import load_config
from investigator.graph.parent_graph import parse_alert
from investigator.graph.routing import route_after_parse
from investigator.reporting.report import build_report

VALID = {
    "incident_id": "INC-9001",
    "dataset": "orders_fact",
    "pipeline": "orders_daily",
    "symptom_type": "volume_drop",
    "window_start": "2026-08-15T00:00:00Z",
    "window_end": "2026-08-16T00:00:00Z",
    "detected_at": "2026-08-16T02:00:00Z",
    "severity": "high",
}


def _config() -> dict:
    return {"configurable": {"app_config": load_config()}}


async def _parse(raw: dict) -> dict:
    state = {"raw_alert": raw, "investigation_id": "inv_test", "trace_id": "trace_test"}
    return await parse_alert(state, _config())  # type: ignore[arg-type]


async def test_a_valid_alert_parses_and_proceeds() -> None:
    update = await _parse(VALID)
    assert update["alert"] is not None
    assert route_after_parse(update) == "ok"  # type: ignore[arg-type]


async def test_an_incomplete_alert_stops_at_invalid_input() -> None:
    update = await _parse({k: v for k, v in VALID.items() if k != "detected_at"})

    assert update["alert"] is None
    assert update["outcome"] == "invalid_input"
    assert update["confidence"] == "not_applicable"
    assert route_after_parse(update) == "invalid"  # type: ignore[arg-type]


async def test_an_unknown_symptom_is_not_coerced_into_the_nearest_one() -> None:
    update = await _parse({**VALID, "symptom_type": "everything_looks_weird"})
    assert update["outcome"] == "invalid_input"
    assert "symptom_type" in update["parse_error"]


async def test_an_invalid_alert_produces_a_report_that_names_no_cause() -> None:
    """The invalid path still ends in a structured report — an empty one, on purpose."""
    update = await _parse({**VALID, "window_end": "2026-08-01T00:00:00Z"})
    report = build_report({**update, "investigation_id": "inv_test", "evidence": []})

    assert report["outcome"] == "invalid_input"
    assert report["leading_hypothesis"] is None
    assert report["supporting_evidence_ids"] == []
    assert report["alert"] is None
