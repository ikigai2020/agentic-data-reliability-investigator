"""Unit tests for retrieval influence + memory-as-context evidence (FR-604, AD-005)."""

from __future__ import annotations

from investigator.retrieval.influence import apply_influence, build_memory_evidence
from investigator.schemas.retrieval import RetrievedItem, TrustDecision

from .fixtures import eval_items, orders_hypotheses


def _accepted_item(doc_id: str):
    doc = next(i.document for i in eval_items() if i.document.doc_id == doc_id)
    item = RetrievedItem(document=doc, score=0.9, rank=1)
    decision = TrustDecision(doc_id=doc_id, accepted=True, score=0.9)
    return item, decision


def test_accepted_incident_reorders_hypotheses() -> None:
    hyps = orders_hypotheses()  # rank 1 = transformation_logic
    item, decision = _accepted_item("REL-1")  # confirmed source_data incident
    updated, influence = apply_influence(hyps, [item], [decision])
    assert influence.kind == "reordered"
    order_after = [h.hypothesis_id for h in sorted(updated, key=lambda h: h.rank)]
    # source_data promoted ahead of transformation_logic.
    assert order_after[0].endswith("source_data")
    assert influence.order_before != influence.order_after
    # Promoted hypothesis is marked retrieval-influenced (FR-301 origin).
    promoted = next(h for h in updated if h.hypothesis_id.endswith("source_data"))
    assert promoted.origin == "retrieval_influenced"


def test_all_rejected_yields_rejected_influence() -> None:
    hyps = orders_hypotheses()
    item, _ = _accepted_item("DIFFPIPE-1")
    rejected = TrustDecision(doc_id="DIFFPIPE-1", accepted=False, score=0.9)
    updated, influence = apply_influence(hyps, [item], [rejected])
    assert influence.kind == "rejected"
    assert updated == hyps  # unchanged


def test_memory_evidence_never_supports_a_hypothesis() -> None:
    # AD-005: retrieved memory is historical, carries no support/contradiction.
    item, decision = _accepted_item("REL-1")
    evidence = build_memory_evidence([item], [decision], investigation_id="inv")
    assert len(evidence) == 1
    e = evidence[0]
    assert e.source_kind == "retrieved_incident"
    assert e.freshness_status == "historical"
    assert e.supports == [] and e.contradicts == []
