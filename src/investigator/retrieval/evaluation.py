"""Retrieval evaluation (FR-605, FR-1304).

Measures precision@k, recall@k, and rejection accuracy over a labeled eval set that must
contain clearly-relevant, partially-relevant, keyword-similar-irrelevant, outdated, and
same-symptom/different-pipeline cases (FR-605), plus the influence-on-order effect.

Deterministic and offline: uses the deterministic embedder + the real trust gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import RetrievalConfig
from ..schemas.alert import Alert
from ..schemas.hypothesis import Hypothesis
from ..schemas.retrieval import CorpusDocument
from . import retriever, trust_gate
from .corpus import Corpus


@dataclass
class EvalItem:
    document: CorpusDocument
    is_relevant: bool  # ground-truth relevance for precision/recall
    should_reject: bool  # ground-truth trust-gate decision


@dataclass
class RetrievalMetrics:
    k: int
    precision_at_k: float
    recall_at_k: float
    rejection_accuracy: float
    gate_accuracy: float
    retrieved_ids: list[str] = field(default_factory=list)
    accepted_ids: list[str] = field(default_factory=list)


def evaluate_retrieval(
    alert: Alert,
    hypotheses: list[Hypothesis],
    eval_items: list[EvalItem],
    *,
    cfg: RetrievalConfig,
    now: datetime,
) -> RetrievalMetrics:
    labels = {i.document.doc_id: i for i in eval_items}
    corpus = Corpus([i.document for i in eval_items])
    k = cfg.top_k

    retrieved = retriever.retrieve(alert, hypotheses, corpus, top_k=k)
    retrieved_ids = [r.document.doc_id for r in retrieved]

    relevant_ids = {i.document.doc_id for i in eval_items if i.is_relevant}
    retrieved_relevant = [d for d in retrieved_ids if d in relevant_ids]
    precision = (len(retrieved_relevant) / len(retrieved_ids)) if retrieved_ids else 0.0
    recall = (len(retrieved_relevant) / len(relevant_ids)) if relevant_ids else 0.0

    decisions = trust_gate.evaluate(
        retrieved,
        alert=alert,
        hypotheses=hypotheses,
        current_evidence=[],
        cfg=cfg,
        now=now,
    )
    accepted_ids = [d.doc_id for d in decisions if d.accepted]

    # Rejection accuracy over the retrieved items that *should* be rejected.
    should_reject_retrieved = [d for d in decisions if labels[d.doc_id].should_reject]
    correctly_rejected = [d for d in should_reject_retrieved if not d.accepted]
    rejection_accuracy = (
        len(correctly_rejected) / len(should_reject_retrieved)
        if should_reject_retrieved
        else 1.0
    )

    # Overall gate correctness over all retrieved items.
    correct = sum(1 for d in decisions if d.accepted != labels[d.doc_id].should_reject)
    gate_accuracy = (correct / len(decisions)) if decisions else 1.0

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision,
        recall_at_k=recall,
        rejection_accuracy=rejection_accuracy,
        gate_accuracy=gate_accuracy,
        retrieved_ids=retrieved_ids,
        accepted_ids=accepted_ids,
    )
