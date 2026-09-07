"""Memory promotion — the write side of retrieval (FR-1109, NFR-010).

Milestone 2 built the read-side control: only ``confirmed``/``approved``, non-superseded
documents may influence planning. This is the other half — what it takes for an
investigation to *become* one of those documents.

The gate is the *human*, not the machine. A diagnosis the system reached on its own is
not precedent however confident it was; a human-confirmed, attributed cause is —
including on an investigation the system abstained from. That case is deliberately
allowed, and is the most valuable memory the system can acquire: the incident it could
not solve, with the answer a person eventually supplied. Blocking it would mean the
system only ever learns what it already knew.

Promotion is also idempotent and reversible: re-promoting updates in place, and
:func:`supersede` retires a document without deleting the audit trail — a superseded
document stops being precedent immediately, because the M2 trust gate already rejects it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas.retrieval import CorpusDocument
from ..schemas.review import HumanDecision


class PromotionRefused(RuntimeError):
    """Raised when an outcome does not meet the bar to become precedent."""


@dataclass(frozen=True)
class PromotionResult:
    document: CorpusDocument
    path: Path
    created: bool


def _now() -> datetime:
    return datetime.now(UTC)


def check_promotable(report: dict[str, Any], decision: HumanDecision | None) -> None:
    """Raise :class:`PromotionRefused` unless this outcome may become precedent.

    Every clause here exists to stop the system teaching itself from its own unreviewed
    output, which is the failure NFR-010 is written against.
    """
    if decision is None:
        raise PromotionRefused(
            "no human review on record — an unreviewed outcome is never precedent (NFR-010)"
        )
    if not decision.is_complete:
        raise PromotionRefused(
            f"review is still '{decision.decision}' — an open escalation is not a decision"
        )
    if decision.decision != "confirmed":
        raise PromotionRefused(
            f"review decision is '{decision.decision}'; only a confirmed outcome is precedent"
        )
    if decision.confirmed_root_cause is None:
        raise PromotionRefused("review confirmed no root cause, so there is nothing to record")
    if not decision.reviewer_id:
        raise PromotionRefused("review has no attributable reviewer (NFR-010)")
    if report.get("outcome") == "invalid_input":
        raise PromotionRefused("the alert never validated, so there is no incident to record")
    if not (report.get("alert") or {}).get("dataset"):
        raise PromotionRefused("no dataset on the alert — nothing to key the memory to")


def build_document(report: dict[str, Any], decision: HumanDecision) -> CorpusDocument:
    """Turn a confirmed investigation into a corpus document.

    The text is built from the *human-confirmed* cause and the current evidence summaries,
    not from the model's prose — what gets remembered is what a person signed off on. When
    the system abstained, the evidence summaries are thin and the confirmed category
    carries the document; that is still the right thing to remember.
    """
    alert = report.get("alert") or {}
    category = decision.confirmed_root_cause
    assert category is not None  # guaranteed by check_promotable

    current_summaries = [
        item["summary"]
        for item in report.get("evidence_index", [])
        if item.get("source_kind") == "current_operational" and item.get("summary")
    ]
    doc_id = f"INC-{report.get('investigation_id', 'unknown')}"

    return CorpusDocument(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#0",
        type="incident",
        title=f"{alert.get('symptom_type', 'incident')} on {alert.get('dataset', 'unknown')}"
        f" — {category.value}",
        text=" ".join(
            [
                f"{alert.get('symptom_type', '')} on {alert.get('dataset', '')}",
                f"in pipeline {alert.get('pipeline', '')}.",
                f"Human-confirmed root cause: {category.value}.",
                *current_summaries,
            ]
        ).strip(),
        pipeline=alert.get("pipeline"),
        dataset=alert.get("dataset"),
        symptom_type=alert.get("symptom_type"),
        category=category,
        environment="prod",
        version=None,
        resolution_status="resolved",
        confirmation_status="confirmed",
        created_at=_now(),
        reviewed_at=decision.reviewed_at,
        expires_at=None,
        provenance={
            "promoted_from_investigation": report.get("investigation_id"),
            "reviewer_id": decision.reviewer_id,
            "reviewer_role": decision.reviewer_role,
            "review_id": decision.review_id,
            "remediation_outcome": decision.remediation_outcome,
        },
    )


def promote(
    report: dict[str, Any],
    decision: HumanDecision | None,
    *,
    incidents_dir: str | Path,
) -> PromotionResult:
    """Promote a human-confirmed investigation into the retrieval corpus (FR-1109)."""
    check_promotable(report, decision)
    assert decision is not None
    document = build_document(report, decision)

    directory = Path(incidents_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document.doc_id}.json"
    created = not path.exists()
    path.write_text(
        json.dumps(document.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
    )
    return PromotionResult(document=document, path=path, created=created)


def supersede(doc_id: str, *, incidents_dir: str | Path, reason: str = "") -> Path:
    """Retire a promoted document without deleting it.

    The M2 trust gate rejects ``superseded`` documents, so this takes effect immediately
    while leaving the record intact for audit (FR-1208).
    """
    path = Path(incidents_dir) / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no promoted document '{doc_id}' in {incidents_dir}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["resolution_status"] = "superseded"
    provenance = dict(raw.get("provenance") or {})
    provenance["superseded_at"] = _now().isoformat()
    if reason:
        provenance["superseded_reason"] = reason
    raw["provenance"] = provenance
    path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    return path
