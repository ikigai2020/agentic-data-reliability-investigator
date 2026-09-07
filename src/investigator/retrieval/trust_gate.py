"""Retrieval trust gate (FR-603, AD-005, FR-1109).

A retrieved item may influence planning only when it clears every gate:
  * relevance      — cosine score >= calibrated threshold,
  * permitted_type — type is allowed,
  * confirmed      — confirmation status is confirmed/approved and not superseded/rejected
                     (unconfirmed/unresolved diagnoses are never precedent, FR-1109),
  * metadata_match — pipeline/dataset/environment match sufficiently,
  * not_stale      — within the staleness window and not expired,
  * no_conflict    — does not conflict with current operational evidence (AD-005).

Deterministic; no model involvement. Failing closed is the default (a missing signal is
treated as a gate failure).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..config import RetrievalConfig
from ..schemas.alert import Alert
from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis
from ..schemas.retrieval import CorpusDocument, RetrievedItem, TrustDecision

_TRUSTED_CONFIRMATION = frozenset({"confirmed", "approved"})
_DEFAULT_ENVIRONMENT = "prod"


def _doc_reference_date(doc: CorpusDocument) -> datetime | None:
    return doc.reviewed_at or doc.created_at


def _is_stale(doc: CorpusDocument, now: datetime, max_days: int) -> bool:
    if doc.expires_at is not None and now > doc.expires_at:
        return True
    ref = _doc_reference_date(doc)
    if ref is None:
        return True  # fail closed: undatable memory is treated as stale
    return (now - ref) > timedelta(days=max_days)


def _metadata_matches(doc: CorpusDocument, alert: Alert, expected_env: str) -> tuple[bool, str]:
    if doc.pipeline is not None and doc.pipeline != alert.pipeline:
        return False, f"pipeline mismatch (doc={doc.pipeline}, alert={alert.pipeline})"
    if doc.dataset is not None and doc.dataset != alert.dataset:
        return False, f"dataset mismatch (doc={doc.dataset}, alert={alert.dataset})"
    if doc.environment is not None and doc.environment != expected_env:
        return False, f"environment mismatch (doc={doc.environment}, expected={expected_env})"
    return True, "pipeline/dataset/environment match"


def _conflicts_with_current(
    doc: CorpusDocument, hypotheses: list[Hypothesis], current_evidence: list[Evidence]
) -> bool:
    """True if current operational evidence contradicts the doc's confirmed category."""
    if doc.category is None:
        return False
    hyp_ids = {h.hypothesis_id for h in hypotheses if h.category == doc.category}
    if not hyp_ids:
        return False
    contradicted = any(
        e.source_kind == "current_operational" and (set(e.contradicts) & hyp_ids)
        for e in current_evidence
    )
    supported = any(
        e.source_kind == "current_operational" and (set(e.supports) & hyp_ids)
        for e in current_evidence
    )
    return contradicted and not supported


def evaluate(
    items: list[RetrievedItem],
    *,
    alert: Alert,
    hypotheses: list[Hypothesis],
    current_evidence: list[Evidence],
    cfg: RetrievalConfig,
    now: datetime | None = None,
    expected_environment: str = _DEFAULT_ENVIRONMENT,
) -> list[TrustDecision]:
    """Apply the FR-603 trust gate to each retrieved item."""
    now = now or datetime.now(UTC)
    decisions: list[TrustDecision] = []
    for item in items:
        doc = item.document
        reasons: list[str] = []

        relevance_ok = item.score >= cfg.relevance_threshold
        if not relevance_ok:
            reasons.append(
                f"below relevance threshold ({item.score:.3f} < {cfg.relevance_threshold})"
            )

        type_ok = doc.type in cfg.permitted_types
        if not type_ok:
            reasons.append(f"type '{doc.type}' not permitted")

        confirmed_ok = (
            doc.confirmation_status in _TRUSTED_CONFIRMATION
            and doc.resolution_status != "superseded"
        )
        if not confirmed_ok:
            reasons.append(
                f"not confirmed precedent (confirmation={doc.confirmation_status},"
                f" resolution={doc.resolution_status})"
            )

        metadata_ok, meta_reason = _metadata_matches(doc, alert, expected_environment)
        if not metadata_ok:
            reasons.append(meta_reason)

        not_stale = not _is_stale(doc, now, cfg.max_staleness_days)
        if not not_stale:
            reasons.append("stale or expired")

        no_conflict = not _conflicts_with_current(doc, hypotheses, current_evidence)
        if not no_conflict:
            reasons.append("conflicts with current operational evidence")

        gate_results = {
            "relevance": relevance_ok,
            "permitted_type": type_ok,
            "confirmed": confirmed_ok,
            "metadata_match": metadata_ok,
            "not_stale": not_stale,
            "no_conflict": no_conflict,
        }
        accepted = all(gate_results.values())
        if accepted:
            reasons.append("all trust gates passed")
        decisions.append(
            TrustDecision(
                doc_id=doc.doc_id,
                accepted=accepted,
                score=item.score,
                gate_results=gate_results,
                reasons=reasons,
            )
        )
    return decisions
