"""Shared builders + labeled retrieval eval set for Milestone 2 tests (FR-605)."""

from __future__ import annotations

from datetime import UTC, datetime

from investigator.agents.reasoning import DeterministicReasoning
from investigator.retrieval.evaluation import EvalItem
from investigator.schemas import Alert
from investigator.schemas.hypothesis import Hypothesis
from investigator.schemas.retrieval import CorpusDocument

# Deterministic "now" so staleness is reproducible regardless of the wall clock.
NOW = datetime(2026, 8, 27, tzinfo=UTC)


def orders_alert() -> Alert:
    return Alert(
        incident_id="INC-1001",
        dataset="orders_fact",
        pipeline="orders_daily",
        symptom_type="volume_drop",
        observed_value=12000,
        expected_value=50000,
        window_start=datetime(2026, 8, 15, tzinfo=UTC),
        window_end=datetime(2026, 8, 16, tzinfo=UTC),
        detected_at=datetime(2026, 8, 16, 2, tzinfo=UTC),
        severity="high",
    )


def orders_hypotheses() -> list[Hypothesis]:
    return DeterministicReasoning().generate_hypotheses_sync(orders_alert(), max_count=5)


def _doc(doc_id: str, **kw: object) -> CorpusDocument:
    base = dict(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#0",
        type="incident",
        title=kw.pop("title", "historical incident"),
        text=kw.pop("text", ""),
        environment="prod",
        version="v7",
        resolution_status="resolved",
        confirmation_status="confirmed",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        reviewed_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    base.update(kw)
    return CorpusDocument.model_validate(base)


def eval_items() -> list[EvalItem]:
    """Labeled eval set covering the FR-605 required case classes."""
    return [
        # 1) clearly relevant — same pipeline/dataset/symptom, confirmed, fresh.
        EvalItem(
            document=_doc(
                "REL-1",
                title="orders_daily volume drop from source shortfall",
                text="orders_daily volume_drop orders_fact source_data upstream shortfall",
                pipeline="orders_daily",
                dataset="orders_fact",
                symptom_type="volume_drop",
                category="source_data",
            ),
            is_relevant=True,
            should_reject=False,
        ),
        # 2) partially relevant — same pipeline/dataset, different category, confirmed.
        EvalItem(
            document=_doc(
                "REL-2",
                title="orders_daily volume drop from a transformation filter",
                text="orders_daily volume_drop orders_fact transformation_logic filter regression",
                pipeline="orders_daily",
                dataset="orders_fact",
                symptom_type="volume_drop",
                category="transformation_logic",
            ),
            is_relevant=True,
            should_reject=False,
        ),
        # 3) outdated — topically relevant but stale/superseded -> gate must reject.
        EvalItem(
            document=_doc(
                "OUTDATED-1",
                title="legacy orders_daily volume drop (pre-v7)",
                text="orders_daily volume_drop orders_fact transformation_logic legacy",
                pipeline="orders_daily",
                dataset="orders_fact",
                symptom_type="volume_drop",
                category="transformation_logic",
                resolution_status="superseded",
                reviewed_at=datetime(2024, 1, 1, tzinfo=UTC),
                created_at=datetime(2023, 11, 1, tzinfo=UTC),
            ),
            is_relevant=True,
            should_reject=True,
        ),
        # 4) same symptom, different pipeline -> gate must reject (misleading).
        EvalItem(
            document=_doc(
                "DIFFPIPE-1",
                title="payments_daily volume drop from orchestration failure",
                text="payments_daily volume_drop payments_fact orchestration scheduler failed",
                pipeline="payments_daily",
                dataset="payments_fact",
                symptom_type="volume_drop",
                category="orchestration",
            ),
            is_relevant=False,
            should_reject=True,
        ),
        # 5) keyword-similar but irrelevant (different domain + symptom) -> reject.
        EvalItem(
            document=_doc(
                "IRREL-1",
                title="marketing_daily freshness delay",
                text="marketing_daily freshness_delay campaign_fact late arrival",
                pipeline="marketing_daily",
                dataset="campaign_fact",
                symptom_type="freshness_delay",
                category="orchestration",
            ),
            is_relevant=False,
            should_reject=True,
        ),
        # 6) unconfirmed draft on the right pipeline -> reject (FR-1109 precedent control).
        EvalItem(
            document=_doc(
                "UNCONF-1",
                title="orders_daily volume drop (unconfirmed draft)",
                text="orders_daily volume_drop orders_fact data_quality draft not reviewed",
                pipeline="orders_daily",
                dataset="orders_fact",
                symptom_type="volume_drop",
                category="data_quality",
                confirmation_status="unconfirmed",
            ),
            is_relevant=False,
            should_reject=True,
        ),
        # 7) unrelated domain, low overlap -> not retrieved as relevant, reject.
        EvalItem(
            document=_doc(
                "IRREL-2",
                title="inventory_hourly null spike",
                text="inventory_hourly null_spike stock_levels data_quality nulls",
                pipeline="inventory_hourly",
                dataset="stock_levels",
                symptom_type="null_spike",
                category="data_quality",
            ),
            is_relevant=False,
            should_reject=True,
        ),
    ]
