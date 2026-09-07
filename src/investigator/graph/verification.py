"""MCP-based incident verification (FR-803, FR-901).

Before hypothesis generation the deterministic controller verifies the alert symptom
through a suitable MCP tool (plan ambiguity #1 resolves the symptom->tool map). The
comparison is deterministic; the model is not involved.

Verification is executed by the controller using the *appropriate specialist-scoped
proxy* (the Commander has no tools, FR-506). Outcomes: ``verified``, ``not_verified``,
``verification_unavailable``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ..agents.reasoning import DeterministicReasoning
from ..schemas.alert import Alert
from ..schemas.enums import VerificationStatus
from ..schemas.evidence import Evidence
from ..schemas.results import ToolResult

# symptom -> (specialist role that owns the verification tool, tool name)
VERIFICATION_TOOL: dict[str, tuple[str, str]] = {
    "volume_drop": ("data_investigator", "get_table_metrics"),
    "volume_spike": ("data_investigator", "get_table_metrics"),
    "null_spike": ("data_investigator", "get_table_metrics"),
    "duplicate_processing": ("data_investigator", "get_table_metrics"),
    "uniqueness_failure": ("data_investigator", "get_quality_results"),
    "referential_integrity_failure": ("data_investigator", "get_quality_results"),
    "freshness_delay": ("data_investigator", "get_table_metrics"),
    "schema_change": ("data_investigator", "get_schema_changes"),
    "pipeline_failure": ("pipeline_investigator", "get_pipeline_run_status"),
}

ProxyFor = Callable[[str], Any]  # role -> object with async .call(tool, args)


def _now() -> datetime:
    return datetime.now(UTC)


def _condition_holds(symptom: str, data: dict, alert: Alert, tol: float) -> tuple[bool, str]:
    """Deterministically decide whether the alert condition is present in ``data``."""
    if symptom in {"volume_drop", "volume_spike"}:
        rc = data.get("row_count")
        expected = data.get("expected_row_count")
        if expected is None:
            expected = alert.expected_value
        if rc is None or expected is None:
            return False, "row_count or expected unavailable"
        if symptom == "volume_drop":
            ok = rc < expected * (1 - tol)
            return ok, f"row_count={rc} vs expected={expected} (drop)"
        ok = rc > expected * (1 + tol)
        return ok, f"row_count={rc} vs expected={expected} (spike)"

    if symptom == "freshness_delay":
        fs = data.get("freshness_seconds")
        if fs is None:
            return False, "freshness unavailable"
        threshold = alert.expected_value if alert.expected_value is not None else 3600
        ok = fs > threshold * (1 + tol)
        return ok, f"freshness_seconds={fs} vs threshold={threshold}"

    if symptom == "null_spike":
        null_rate = data.get("null_rate", {})
        worst = max(null_rate.values(), default=0.0)
        ok = worst > 0.2
        return ok, f"max null_rate={worst}"

    if symptom == "duplicate_processing":
        dup = data.get("duplicate_count") or 0
        return dup > 0, f"duplicate_count={dup}"

    if symptom in {"uniqueness_failure", "referential_integrity_failure"}:
        results = data.get("results", [])
        failing = [r for r in results if r.get("status") in {"fail", "error"}]
        return bool(failing), f"{len(failing)} failing quality test(s)"

    if symptom == "schema_change":
        changes = data.get("changes", [])
        return bool(changes), f"{len(changes)} schema change(s)"

    if symptom == "pipeline_failure":
        state = data.get("state")
        return state in {"failed", "no_run", "skipped"}, f"run state={state}"

    return False, f"no verification rule for {symptom}"


async def verify_alert(
    alert: Alert,
    proxy_for: ProxyFor,
    *,
    tolerance: float,
    investigation_id: str,
) -> tuple[VerificationStatus, dict[str, Any], Evidence | None]:
    """Verify the alert symptom via MCP and return (status, detail, evidence)."""
    reasoning = DeterministicReasoning()
    mapping = VERIFICATION_TOOL.get(alert.symptom_type)
    if mapping is None:
        return "verification_unavailable", {"reason": "no tool mapping"}, None

    role, tool_name = mapping
    proxy = proxy_for(role)
    args = reasoning.tool_arguments(tool_name, alert)
    result: ToolResult = await proxy.call(tool_name, args)

    detail: dict[str, Any] = {
        "tool_name": tool_name,
        "role": role,
        "status": result.status,
        "request_id": result.request_id,
    }

    if result.status not in {"ok"}:
        # no_data / unavailable / timeout: symptom cannot be confirmed (FR-508, FR-803).
        detail["reason"] = result.error_message or result.status
        return "verification_unavailable", detail, None

    holds, explanation = _condition_holds(alert.symptom_type, result.data, alert, tolerance)
    detail["explanation"] = explanation
    status: VerificationStatus = "verified" if holds else "not_verified"

    evidence = Evidence(
        evidence_id=f"ev_verification_{alert.incident_id}",
        investigation_id=investigation_id,
        task_id="verification",
        producing_agent="controller",
        source_kind="current_operational",
        source_name=result.source_system,
        tool_name=tool_name,
        observed_at=result.observed_at,
        collected_at=result.collected_at or _now(),
        payload=result.data,
        summary=f"verification: {explanation} -> {status}",
        supports=[],
        contradicts=[],
        freshness_status="current",
        provenance={**result.provenance, "request_id": result.request_id, "phase": "verification"},
    )
    return status, detail, evidence
