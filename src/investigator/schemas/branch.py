"""Investigation branch contract (FR-305).

Included for contract completeness. Beam-search population of branches (scores,
pruning, reopening) is Milestone 3; in Milestone 1 a single branch per hypothesis
may be tracked but is not scored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import BranchStatus


class InvestigationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: str
    hypothesis_id: str
    depth: int = 0
    parent_branch_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    score: float | None = None
    status: BranchStatus = "active"
    prune_reason: str | None = None
