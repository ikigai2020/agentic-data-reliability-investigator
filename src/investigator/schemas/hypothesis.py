"""Hypothesis contract (FR-301).

Initial hypotheses carry no fabricated probabilities (FR-301, coding rule 9).
``rank`` is ordinal; numeric scoring arrives with beam search in Milestone 3.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import HypothesisOrigin, HypothesisStatus, RootCauseCategory


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    category: RootCauseCategory
    statement: str
    discriminating_question: str
    rank: int
    status: HypothesisStatus = "active"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    origin: HypothesisOrigin = "initial_generation"
