"""Closed-loop improvement backlog (FR-1207, NFR-010)."""

from __future__ import annotations

import pytest

from investigator.governance import BacklogRefused, BacklogStore, from_evaluation


@pytest.fixture
def store(tmp_path) -> BacklogStore:
    return BacklogStore(tmp_path / "backlog")


def _propose(store: BacklogStore, **kw):
    defaults = dict(
        title="stopping evaluator accepts a single-tool support",
        change_kind="policy",
        evidence_source="evaluation",
        failure_evidence=["scenario=inconclusive_missing_evidence", "error=false_diagnosis"],
        regression_test="tests/scenarios/test_scenarios.py::test_abstains",
    )
    defaults.update(kw)
    return store.propose(**defaults)  # type: ignore[arg-type]


def test_a_proposal_needs_observed_failure_evidence(store: BacklogStore) -> None:
    """"The prompt feels weak" is not a backlog item."""
    with pytest.raises(BacklogRefused, match="observed failure evidence"):
        _propose(store, failure_evidence=[])


def test_a_proposal_needs_a_regression_test(store: BacklogStore) -> None:
    """A fix without one is how the same bug comes back."""
    with pytest.raises(BacklogRefused, match="regression test"):
        _propose(store, regression_test="")


def test_a_production_trace_needs_a_human_review_behind_it(store: BacklogStore) -> None:
    """NFR-010: changes are never learned from unreviewed production traces."""
    with pytest.raises(BacklogRefused, match="unreviewed production traces"):
        _propose(store, evidence_source="production_trace", review_id=None)


def test_a_reviewed_production_trace_is_accepted(store: BacklogStore) -> None:
    item = _propose(store, evidence_source="production_trace", review_id="rev_abc123")
    assert item.review_id == "rev_abc123"


def test_a_well_formed_proposal_is_recorded(store: BacklogStore) -> None:
    item = _propose(store, rationale="two supports came from one tool")
    assert item.item_id.startswith("imp_")
    assert item.status == "proposed"
    assert item.failure_evidence
    assert item.regression_test

    reloaded = store.all()
    assert len(reloaded) == 1
    assert reloaded[0].item_id == item.item_id


def test_status_can_advance(store: BacklogStore) -> None:
    item = _propose(store)
    assert store.set_status(item.item_id, "accepted").status == "accepted"
    assert store.set_status(item.item_id, "shipped").status == "shipped"


def test_an_unknown_item_cannot_be_advanced(store: BacklogStore) -> None:
    with pytest.raises(KeyError):
        store.set_status("imp_nope", "shipped")


def test_evaluation_failures_become_drafts_not_items() -> None:
    """The system does not get to file its own fixes — a human triages them."""
    report = {
        "matrix": [
            {
                "scenario_id": "s1",
                "correct": False,
                "errors": ["false_confident_diagnosis"],
                "expected_outcome": "inconclusive",
                "outcome": "diagnosed",
            },
            {"scenario_id": "s2", "correct": True, "errors": []},
        ]
    }
    drafts = from_evaluation(report, regression_test="tests/evaluation/test_acceptance.py")

    assert len(drafts) == 1
    assert drafts[0]["title"].startswith("s1")
    assert any("false_confident_diagnosis" in e for e in drafts[0]["failure_evidence"])
    assert drafts[0]["regression_test"]
    assert "needs human triage" in drafts[0]["rationale"]


def test_drafts_still_have_to_clear_the_gate(store: BacklogStore) -> None:
    """A draft is a proposal, not an exemption."""
    drafts = from_evaluation(
        {"matrix": [{"scenario_id": "s1", "correct": False, "errors": ["x"]}]},
        regression_test="",
    )
    with pytest.raises(BacklogRefused):
        store.propose(**drafts[0])  # type: ignore[arg-type]


def test_an_empty_backlog_is_not_an_error(store: BacklogStore) -> None:
    assert store.all() == []
