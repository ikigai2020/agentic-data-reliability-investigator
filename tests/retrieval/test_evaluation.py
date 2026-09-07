"""FR-605 retrieval evaluation: precision@k, recall@k, rejection accuracy."""

from __future__ import annotations

from investigator.config import load_config
from investigator.retrieval.evaluation import evaluate_retrieval

from .fixtures import NOW, eval_items, orders_alert, orders_hypotheses


def test_retrieval_metrics_meet_thresholds() -> None:
    cfg = load_config().retrieval
    metrics = evaluate_retrieval(
        orders_alert(),
        orders_hypotheses(),
        eval_items(),
        cfg=cfg,
        now=NOW,
    )
    # Relevant orders_daily incidents surface (recall), and irrelevant ones don't crowd
    # them out (precision > 0).
    assert metrics.recall_at_k >= 0.66, metrics
    assert metrics.precision_at_k > 0.0, metrics
    # Every item that should be rejected is rejected by the trust gate (FR-603).
    assert metrics.rejection_accuracy == 1.0, metrics
    # Overall gate correctness is high.
    assert metrics.gate_accuracy >= 0.85, metrics


def test_misleading_and_outdated_never_accepted() -> None:
    cfg = load_config().retrieval
    metrics = evaluate_retrieval(
        orders_alert(), orders_hypotheses(), eval_items(), cfg=cfg, now=NOW
    )
    # The different-pipeline, outdated, unconfirmed, and irrelevant docs must not be
    # accepted as precedent.
    for bad in ("DIFFPIPE-1", "OUTDATED-1", "UNCONF-1", "IRREL-1", "IRREL-2"):
        assert bad not in metrics.accepted_ids, (bad, metrics.accepted_ids)
