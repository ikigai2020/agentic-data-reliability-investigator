"""End-to-end scenario tests (§22 M1 exit, FR-1300/1301, §23 demonstrations).

Runs the full parent LangGraph over the fixture scenarios and asserts the deterministic
outcomes, grounding, evidence traceability, and escalation behaviour declared in each
scenario's meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from investigator.app import run_investigation
from investigator.config import load_config
from investigator.reporting.report import validate_grounding

from ..conftest import DIAGNOSED, INCONCLUSIVE

pytestmark = pytest.mark.asyncio


def _meta(scenario_id: str) -> dict:
    path = Path(load_config().scenarios_dir) / scenario_id / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_diagnosed_scenario_reaches_diagnosis() -> None:
    meta = _meta(DIAGNOSED)
    state = await run_investigation(DIAGNOSED)

    assert state["verification_status"] == meta["expected_verification"]
    assert state["outcome"] == meta["expected_outcome"] == "diagnosed"

    report = state["report"]
    leading = report["leading_hypothesis"]
    assert leading["category"] == meta["expected_root_cause_category"]

    # All six FR-900 clauses satisfied.
    checks = report["guardrail_and_budget_events"]["diagnosis_checks"]
    assert all(checks.values()), checks

    # >= 2 independent supporting current observations, all traceable to real evidence.
    supporting = report["supporting_evidence_ids"]
    assert len(supporting) >= 2
    evidence_ids = {e.evidence_id for e in state["evidence"]}
    assert set(supporting).issubset(evidence_ids)

    # Required discriminating tools participated.
    called = set(report["tools"]["called"])
    assert set(meta["required_evidence_tools"]).issubset(called)

    # Grounding holds (§18: 100% grounding for delivered reports).
    grounded, issues = validate_grounding(report, state["evidence"])
    assert grounded, issues

    # Risk-gated release (v2.1 FR-1105): read-only, low-impact -> releasable diagnosis.
    assert state["risk_tier"] in {"low", "medium"}
    assert state["release_status"] == meta["expected_release_status"] == "released"

    # No escalation for a clean, releasable diagnosis.
    assert not state.get("escalation_package")


async def test_diagnosed_is_deterministic() -> None:
    a = await run_investigation(DIAGNOSED)
    b = await run_investigation(DIAGNOSED)
    assert a["outcome"] == b["outcome"] == "diagnosed"
    assert (
        a["report"]["leading_hypothesis"]["category"]
        == b["report"]["leading_hypothesis"]["category"]
    )
    assert a["report"]["supporting_evidence_ids"] == b["report"]["supporting_evidence_ids"]


async def test_inconclusive_scenario_abstains_and_escalates() -> None:
    meta = _meta(INCONCLUSIVE)
    state = await run_investigation(INCONCLUSIVE)

    assert state["outcome"] == meta["expected_outcome"] == "inconclusive"
    assert state["outcome"] not in meta["forbidden_outcomes"]
    assert state["confidence"] == "not_applicable"

    # Escalation package prepared (FR-1103).
    assert state["escalation_required"] is True
    pkg = state["escalation_package"]
    assert pkg is not None
    assert pkg["recommended_next_checks"]
    assert "stop_reason" in pkg

    # No fabricated diagnosis: leading hypothesis has < 2 independent supports.
    checks = state["report"]["guardrail_and_budget_events"]["diagnosis_checks"]
    assert checks["two_independent_current_supports"] is False

    # Risk gate present; a low-impact abstention still releases as an abstention.
    assert state["risk_tier"] in {"low", "medium"}
    assert state["release_status"] == "released"


async def test_reports_persisted_to_outputs() -> None:
    state = await run_investigation(DIAGNOSED)
    out_path = Path(load_config().outputs_dir) / f"{state['investigation_id']}.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["report"]["outcome"] == "diagnosed"
