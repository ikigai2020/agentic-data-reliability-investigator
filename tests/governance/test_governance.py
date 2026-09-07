"""Human review capture and memory promotion (FR-1108, FR-1109, NFR-010)."""

from __future__ import annotations

import json

import pytest

from investigator.governance import PromotionRefused, ReviewStore, promote, supersede
from investigator.retrieval.corpus import Corpus
from investigator.retrieval.trust_gate import evaluate as trust_evaluate
from investigator.schemas.enums import RootCauseCategory
from investigator.schemas.retrieval import CorpusDocument, RetrievedItem

from ..search.fixtures import orders_alert

REPORT = {
    "investigation_id": "inv_test123",
    "outcome": "diagnosed",
    "alert": {
        "dataset": "orders_fact",
        "pipeline": "orders_daily",
        "symptom_type": "volume_drop",
    },
    "evidence_index": [
        {
            "evidence_id": "ev_1",
            "source_kind": "current_operational",
            "summary": "400 rows dropped at the transform stage",
        },
        {
            "evidence_id": "ev_mem_1",
            "source_kind": "retrieved_incident",
            "summary": "a historical incident that must not be copied forward",
        },
    ],
}


@pytest.fixture
def store(tmp_path) -> ReviewStore:
    return ReviewStore(tmp_path / "reviews")


def _confirm(store: ReviewStore, investigation_id: str = "inv_test123", **kw):
    defaults = dict(
        reviewer_id="alice@example.com",
        reviewer_role="data-platform-oncall",
        decision="confirmed",
        confirmed_root_cause="transformation_logic",
    )
    defaults.update(kw)
    return store.record(investigation_id, **defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FR-1108 — an escalation is open until someone answers
# --------------------------------------------------------------------------- #
def test_an_opened_escalation_is_pending_not_absent(store: ReviewStore) -> None:
    """"Nobody answered yet" must be distinguishable from "nobody was asked"."""
    store.open_escalation("inv_test123", reason="insufficient evidence")
    record = store.get("inv_test123")

    assert record is not None
    assert record.decision == "pending"
    assert record.is_complete is False
    assert store.disposition_stats().queue_drain_rate == 0.0


def test_opening_the_same_escalation_twice_keeps_the_first_record(store: ReviewStore) -> None:
    first = store.open_escalation("inv_test123")
    second = store.open_escalation("inv_test123")
    assert first.review_id == second.review_id


def test_recording_a_decision_completes_the_escalation(store: ReviewStore) -> None:
    opened = store.open_escalation("inv_test123")
    decision = _confirm(store, override_reason="reconciliation was conclusive")

    assert decision.review_id == opened.review_id  # same review, now answered
    assert decision.is_complete is True
    assert store.disposition_stats().queue_drain_rate == 1.0


def test_a_decision_records_who_decided_and_why(store: ReviewStore) -> None:
    decision = _confirm(
        store,
        accepted_claims=["filter regression"],
        rejected_claims=["upstream shortfall"],
        override_reason="the source reconciled cleanly",
        remediation_outcome="filter reverted",
    )
    assert decision.reviewer_id == "alice@example.com"
    assert decision.reviewer_role == "data-platform-oncall"
    assert decision.accepted_claims == ["filter regression"]
    assert decision.rejected_claims == ["upstream shortfall"]
    assert decision.override_reason
    assert decision.remediation_outcome


def test_disposition_stats_count_open_and_closed(store: ReviewStore) -> None:
    store.open_escalation("inv_a")
    store.open_escalation("inv_b")
    _confirm(store, "inv_a")

    stats = store.disposition_stats()
    assert (stats.total, stats.complete, stats.pending) == (2, 1, 1)
    assert stats.queue_drain_rate == 0.5


# --------------------------------------------------------------------------- #
# FR-1109 / NFR-010 — only a reviewed, attributed outcome becomes precedent
# --------------------------------------------------------------------------- #
def test_an_unreviewed_outcome_is_never_precedent(tmp_path) -> None:
    with pytest.raises(PromotionRefused, match="no human review"):
        promote(REPORT, None, incidents_dir=tmp_path)


def test_an_open_escalation_is_not_a_decision(store: ReviewStore, tmp_path) -> None:
    store.open_escalation("inv_test123")
    with pytest.raises(PromotionRefused, match="still 'pending'"):
        promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)


def test_a_rejected_review_is_not_precedent(store: ReviewStore, tmp_path) -> None:
    _confirm(store, decision="rejected")
    with pytest.raises(PromotionRefused, match="only a confirmed outcome"):
        promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)


def test_a_confirmation_without_a_reviewer_is_refused(store: ReviewStore, tmp_path) -> None:
    _confirm(store, reviewer_id="")
    with pytest.raises(PromotionRefused, match="attributable reviewer"):
        promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)


def test_an_invalid_alert_never_becomes_memory(store: ReviewStore, tmp_path) -> None:
    _confirm(store)
    bad = {**REPORT, "outcome": "invalid_input"}
    with pytest.raises(PromotionRefused, match="never validated"):
        promote(bad, store.get("inv_test123"), incidents_dir=tmp_path)


def test_a_confirmed_outcome_is_promoted_with_full_attribution(
    store: ReviewStore, tmp_path
) -> None:
    _confirm(store, remediation_outcome="filter reverted")
    result = promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)

    assert result.created is True
    doc = result.document
    assert doc.confirmation_status == "confirmed"
    assert doc.resolution_status == "resolved"
    assert doc.category is RootCauseCategory.TRANSFORMATION_LOGIC
    assert doc.pipeline == "orders_daily"
    assert doc.provenance["reviewer_id"] == "alice@example.com"
    assert doc.provenance["promoted_from_investigation"] == "inv_test123"
    assert doc.provenance["remediation_outcome"] == "filter reverted"

    # Round-trips through the corpus contract.
    saved = CorpusDocument.model_validate(json.loads(result.path.read_text()))
    assert saved.doc_id == doc.doc_id


def test_promoted_memory_never_copies_forward_other_memory(
    store: ReviewStore, tmp_path
) -> None:
    """AD-005: a promoted document is built from current observations, not from memory."""
    _confirm(store)
    result = promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)
    assert "must not be copied forward" not in result.document.text
    assert "400 rows dropped" in result.document.text


def test_an_abstention_a_human_resolved_is_the_most_valuable_memory(
    store: ReviewStore, tmp_path
) -> None:
    """The case the system could not solve, with the answer a person supplied.

    Refusing this would mean the system only ever learns what it already knew.
    """
    _confirm(store, confirmed_root_cause="source_data")
    abstained = {
        **REPORT,
        "outcome": "inconclusive",
        "evidence_index": [
            {
                "evidence_id": "ev_1",
                "source_kind": "current_operational",
                "summary": "compare_source_and_target: no data",
            }
        ],
    }
    result = promote(abstained, store.get("inv_test123"), incidents_dir=tmp_path)
    assert result.document.category is RootCauseCategory.SOURCE_DATA
    assert "Human-confirmed root cause: source_data" in result.document.text


def test_promotion_is_idempotent(store: ReviewStore, tmp_path) -> None:
    _confirm(store)
    decision = store.get("inv_test123")
    assert promote(REPORT, decision, incidents_dir=tmp_path).created is True
    assert promote(REPORT, decision, incidents_dir=tmp_path).created is False
    assert len(list(tmp_path.glob("*.json"))) == 1


# --------------------------------------------------------------------------- #
# The promoted document has to behave in the M2 trust gate
# --------------------------------------------------------------------------- #
def test_a_promoted_document_is_accepted_as_precedent(store: ReviewStore, tmp_path) -> None:
    from investigator.config import load_config

    _confirm(store)
    result = promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)

    decisions = trust_evaluate(
        [RetrievedItem(document=result.document, score=0.9, rank=1)],
        alert=orders_alert(),
        hypotheses=[],
        current_evidence=[],
        cfg=load_config().retrieval,
    )
    assert decisions[0].accepted is True, decisions[0].reasons


def test_superseding_a_document_retires_it_without_deleting_it(
    store: ReviewStore, tmp_path
) -> None:
    from investigator.config import load_config

    _confirm(store)
    result = promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)
    supersede(result.document.doc_id, incidents_dir=tmp_path, reason="root cause revised")

    raw = json.loads(result.path.read_text())
    assert raw["resolution_status"] == "superseded"
    assert raw["provenance"]["superseded_reason"] == "root cause revised"
    assert raw["provenance"]["reviewer_id"]  # audit trail intact

    retired = CorpusDocument.model_validate(raw)
    decisions = trust_evaluate(
        [RetrievedItem(document=retired, score=0.9, rank=1)],
        alert=orders_alert(),
        hypotheses=[],
        current_evidence=[],
        cfg=load_config().retrieval,
    )
    assert decisions[0].accepted is False
    assert any("not confirmed precedent" in r for r in decisions[0].reasons)


def test_a_promoted_document_is_loadable_by_the_corpus(store: ReviewStore, tmp_path) -> None:
    _confirm(store)
    promote(REPORT, store.get("inv_test123"), incidents_dir=tmp_path)
    corpus = Corpus.from_dirs(tmp_path)
    assert len(corpus) == 1
