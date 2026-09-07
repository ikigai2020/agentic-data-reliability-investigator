"""Critic review contract (FR-306).

Included for contract completeness. The Evidence Critic *agent* is Milestone 3;
in Milestone 1 no critic node produces these, but the deterministic stopping
evaluator emits an equivalent ``stop_recommendation`` semantics internally.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import StopRecommendation


class CriticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    round_number: int
    branch_scores: dict[str, float] = Field(default_factory=dict)
    strongest_hypothesis_id: str | None = None
    strongest_competitor_id: str | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    pruning_recommendations: list[str] = Field(default_factory=list)
    reopening_recommendations: list[str] = Field(default_factory=list)
    stop_recommendation: StopRecommendation = "continue"
    rationale_summary: str = ""
