"""Unit tests for the FR-603 trust gate (each gate rejects appropriately)."""

from __future__ import annotations

from investigator.config import load_config
from investigator.retrieval import trust_gate
from investigator.schemas.retrieval import RetrievedItem

from .fixtures import NOW, eval_items, orders_alert, orders_hypotheses

_CFG = load_config().retrieval


def _decisions():
    items = [
        RetrievedItem(document=i.document, score=0.9, rank=r)
        for r, i in enumerate(eval_items(), 1)
    ]
    decisions = trust_gate.evaluate(
        items,
        alert=orders_alert(),
        hypotheses=orders_hypotheses(),
        current_evidence=[],
        cfg=_CFG,
        now=NOW,
    )
    return {d.doc_id: d for d in decisions}


def test_relevant_confirmed_docs_accepted() -> None:
    d = _decisions()
    assert d["REL-1"].accepted is True
    assert d["REL-2"].accepted is True


def test_different_pipeline_rejected_on_metadata() -> None:
    d = _decisions()["DIFFPIPE-1"]
    assert d.accepted is False
    assert d.gate_results["metadata_match"] is False


def test_outdated_superseded_rejected() -> None:
    d = _decisions()["OUTDATED-1"]
    assert d.accepted is False
    # stale (old reviewed_at) and superseded both disqualify it.
    assert d.gate_results["not_stale"] is False or d.gate_results["confirmed"] is False


def test_unconfirmed_rejected_as_precedent() -> None:
    # FR-1109: unconfirmed diagnoses are never retrieved as confirmed precedent.
    d = _decisions()["UNCONF-1"]
    assert d.accepted is False
    assert d.gate_results["confirmed"] is False


def test_low_relevance_rejected() -> None:
    # A confirmed, on-pipeline doc with a below-threshold score is still rejected.
    item = RetrievedItem(
        document=eval_items()[0].document,  # REL-1, confirmed/on-pipeline
        score=0.01,
        rank=1,
    )
    [d] = trust_gate.evaluate(
        [item],
        alert=orders_alert(),
        hypotheses=orders_hypotheses(),
        current_evidence=[],
        cfg=_CFG,
        now=NOW,
    )
    assert d.accepted is False
    assert d.gate_results["relevance"] is False
