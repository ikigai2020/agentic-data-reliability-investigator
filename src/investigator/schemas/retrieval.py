"""Retrieval and long-term memory contracts (Milestone 2, §11).

Retrieved memory is context, never current operational evidence (AD-005, FR-604). These
contracts carry the corpus documents, their trust decisions, and the recorded influence
of retrieval on the investigation.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ConfirmationStatus,
    DocumentType,
    InfluenceKind,
    ResolutionStatus,
    RootCauseCategory,
    SymptomType,
)


class CorpusDocument(BaseModel):
    """A confirmed/approved memory item with metadata (FR-600, FR-601).

    Current metrics, logs, or run state are never represented here (FR-600).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    type: DocumentType
    title: str
    text: str
    pipeline: str | None = None
    dataset: str | None = None
    category: RootCauseCategory | None = None  # confirmed root cause (incidents)
    symptom_type: SymptomType | None = None
    environment: str | None = None
    version: str | None = None
    resolution_status: ResolutionStatus | None = None
    confirmation_status: ConfirmationStatus = "unconfirmed"
    created_at: datetime | None = None
    reviewed_at: datetime | None = None
    expires_at: datetime | None = None
    # NFR-010: who confirmed this and where it came from. Empty for hand-authored corpus
    # documents; populated by the FR-1109 promotion workflow for anything the system
    # learned from its own reviewed investigations.
    provenance: dict = Field(default_factory=dict)


class RetrievedItem(BaseModel):
    """A corpus document surfaced by the retriever with its relevance score."""

    model_config = ConfigDict(extra="forbid")

    document: CorpusDocument
    score: float  # cosine relevance in [0, 1]
    rank: int


class TrustDecision(BaseModel):
    """Outcome of the FR-603 trust gate for one retrieved item."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    accepted: bool
    score: float
    gate_results: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class RetrievalInfluence(BaseModel):
    """Record of what retrieval did to the investigation (FR-604).

    Retrieval may reorder hypotheses or suggest a check, but shall never populate
    ``current_operational`` evidence or establish a diagnosis (AD-005).
    """

    model_config = ConfigDict(extra="forbid")

    kind: InfluenceKind = "not_run"
    detail: str = ""
    promoted_hypotheses: list[str] = Field(default_factory=list)
    accepted_doc_ids: list[str] = Field(default_factory=list)
    rejected_doc_ids: list[str] = Field(default_factory=list)
    order_before: list[str] = Field(default_factory=list)
    order_after: list[str] = Field(default_factory=list)
