"""Retrieval routing and query construction (FR-602).

Retrieval runs only when prior incidents/runbooks/schema context could improve
hypothesis prioritization or action selection. The query is built deterministically from
the verified alert and the current hypotheses.
"""

from __future__ import annotations

from ..schemas.alert import Alert
from ..schemas.hypothesis import Hypothesis
from ..schemas.retrieval import RetrievedItem
from .corpus import Corpus


def should_retrieve(
    alert: Alert | None, hypotheses: list[Hypothesis], corpus: Corpus, *, enabled: bool
) -> tuple[bool, str]:
    """FR-602 routing decision: retrieve only when it could improve planning."""
    if not enabled:
        return False, "retrieval disabled by config"
    if alert is None:
        return False, "no verified alert"
    if len(corpus) == 0:
        return False, "empty corpus"
    if not hypotheses:
        return False, "no hypotheses to prioritize"
    return True, "prior incidents/runbooks may improve hypothesis prioritization"


def build_query(alert: Alert, hypotheses: list[Hypothesis]) -> str:
    """Compose a retrieval query from the alert and current hypothesis categories."""
    categories = " ".join(sorted({h.category.value for h in hypotheses}))
    return f"{alert.symptom_type} {alert.dataset} {alert.pipeline} {categories}"


def retrieve(
    alert: Alert, hypotheses: list[Hypothesis], corpus: Corpus, *, top_k: int
) -> list[RetrievedItem]:
    return corpus.search(build_query(alert, hypotheses), top_k)
