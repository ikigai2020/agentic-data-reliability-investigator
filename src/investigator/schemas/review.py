"""Human review contract (FR-1108, NFR-010).

A human decision is an *auditable input to improvement*, never an automatically trusted
training example (NFR-010). It records who decided, when, what they accepted and
rejected, and why — so a later corpus promotion or policy change can be traced back to an
attributable person, not to an anonymous thumbs-up.

An escalation is not complete until its disposition is recorded or explicitly marked
``pending`` (FR-1108), which is why ``pending`` is a first-class decision rather than the
absence of one.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import HumanDecisionKind, RootCauseCategory


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    investigation_id: str
    reviewer_id: str
    reviewer_role: str
    reviewed_at: datetime
    decision: HumanDecisionKind
    accepted_claims: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    override_reason: str | None = None
    requested_next_check: str | None = None
    confirmed_root_cause: RootCauseCategory | None = None
    remediation_outcome: str | None = None
    notes: str = ""

    @property
    def is_complete(self) -> bool:
        """FR-1108: an escalation stays open until a non-pending disposition is recorded."""
        return self.decision != "pending"

    @property
    def record_is_complete(self) -> bool:
        """FR-1305: a *completed* review must carry the fields that make it auditable.

        Distinct from :attr:`is_complete`, which asks whether anyone has answered yet. A
        pending review is not an incomplete record — it is an open question.
        """
        if not self.is_complete:
            return True  # nothing to be incomplete about yet
        return bool(self.reviewer_id and self.reviewer_role and self.reviewed_at)

    @property
    def is_promotable(self) -> bool:
        """NFR-010: only a confirmed, attributed outcome may become precedent (FR-1109)."""
        return (
            self.decision == "confirmed"
            and self.confirmed_root_cause is not None
            and bool(self.reviewer_id)
        )
