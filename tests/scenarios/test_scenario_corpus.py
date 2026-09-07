"""The M4.3 scenario corpus (FR-1301, FR-1307, §23 demonstrations).

Two scenarios that each fail differently, and the coverage check over the corpus as a
whole. The point of both is a behaviour the happy path cannot show: declining to find a
fault that is not there, and refusing memory that would have pointed the wrong way.

The remaining FR-1301 scenarios are deferred — see TODO.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from investigator.app import run_investigation
from investigator.config import load_config
from investigator.evaluation.labels import load_labels

from ..conftest import MISLEADING_MEMORY, NOT_AN_INCIDENT

# The FR-1301 scenarios this release ships. Adding one means adding its index here, so a
# scenario cannot arrive without declaring which requirement it covers.
SHIPPED_FR1301_INDICES = {1, 2, 3, 4, 5, 6}


def _meta(scenario_id: str) -> dict:
    path = Path(load_config().scenarios_dir) / scenario_id / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_shipped_scenario_is_labelled_and_indexed() -> None:
    labels = load_labels(load_config().scenarios_dir)
    assert {label.fr1301_index for label in labels} == SHIPPED_FR1301_INDICES
    # Every scenario carries the FR-1300 fields the harness scores against.
    for label in labels:
        assert label.expected_outcome
        assert label.expected_verification
        assert label.severity in {"low", "medium", "high", "critical"}


# --------------------------------------------------------------------------- #
# #4 — a legitimate business change is not an incident
# --------------------------------------------------------------------------- #
async def test_legitimate_volume_change_is_not_an_incident() -> None:
    meta = _meta(NOT_AN_INCIDENT)
    state = await run_investigation(NOT_AN_INCIDENT)

    assert state["verification_status"] == meta["expected_verification"] == "not_verified"
    assert state["outcome"] == "not_an_incident"
    assert state["confidence"] == "not_applicable"
    assert state["release_status"] == "released"
    assert not state.get("escalation_package")


async def test_declining_costs_nothing_and_names_no_cause() -> None:
    """FR-901 stops before hypothesis generation: no cause is named, no budget spent."""
    state = await run_investigation(NOT_AN_INCIDENT)

    assert state["report"]["leading_hypothesis"] is None
    assert state["report"]["supporting_evidence_ids"] == []
    assert state.get("hypotheses", []) == []
    assert state["budgets"]["calls_used"] == 0


# --------------------------------------------------------------------------- #
# #5 — misleading memory from another pipeline (§23.3)
# --------------------------------------------------------------------------- #
async def test_misleading_memory_is_rejected_on_metadata() -> None:
    state = await run_investigation(MISLEADING_MEMORY)
    decisions = {d.doc_id: d for d in state["trust_decisions"]}

    trap = decisions["INC-HIST-0055"]
    assert not trap.accepted
    # Rejected on *metadata*, having cleared relevance, type, confirmation and staleness —
    # so the rejection cannot be explained away as the document simply not matching.
    assert trap.gate_results == {
        "relevance": True,
        "permitted_type": True,
        "confirmed": True,
        "metadata_match": False,
        "not_stale": True,
        "no_conflict": True,
    }
    assert not any(d.accepted for d in state["trust_decisions"])


async def test_the_trap_document_is_the_most_relevant_candidate() -> None:
    """A trap nothing retrieves proves nothing. It has to be the top hit and still lose."""
    state = await run_investigation(MISLEADING_MEMORY)
    ranked = sorted(state["trust_decisions"], key=lambda d: d.score, reverse=True)
    assert ranked[0].doc_id == "INC-HIST-0055"


async def test_rejected_memory_leaves_hypothesis_order_untouched() -> None:
    state = await run_investigation(MISLEADING_MEMORY)
    influence = state["retrieval_influence"]

    assert influence.kind == "rejected"
    assert influence.order_before == influence.order_after


async def test_the_diagnosis_rests_on_current_evidence_alone() -> None:
    meta = _meta(MISLEADING_MEMORY)
    state = await run_investigation(MISLEADING_MEMORY)

    assert state["outcome"] == "diagnosed"
    leading = state["report"]["leading_hypothesis"]
    # The trap blamed schema_contract; current evidence says data_quality.
    assert leading["category"] == meta["expected_root_cause_category"] == "data_quality"

    by_id = {e.evidence_id: e for e in state["evidence"]}
    supporting = state["report"]["supporting_evidence_ids"]
    assert len(supporting) >= 2
    assert all(by_id[e].source_kind == "current_operational" for e in supporting)
