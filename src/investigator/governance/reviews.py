"""Human review store (FR-1108, FR-1305).

Escalations are written here when an investigation asks for human authority, and closed
here when a person answers. The store is deliberately boring — one JSON file per review
under ``data/reviews/`` — because its value is auditability, not throughput.

The completeness metric FR-1305 requires ("human-review disposition records complete:
100% for completed reviews") is computed from this store by :func:`disposition_stats`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas.review import HumanDecision


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class DispositionStats:
    """Two different questions, deliberately not merged into one number.

    *Queue drain* asks how many escalations anyone has answered — a workload signal that
    legitimately sits below 100% while people are still working.

    *Record completeness* asks whether the reviews that **were** answered carry the fields
    FR-1305 requires. That one must be 100%: an answered review missing its reviewer is a
    governance failure, whereas an unanswered one is just a queue.
    """

    total: int
    complete: int
    pending: int
    well_formed: int = 0

    @property
    def queue_drain_rate(self) -> float:
        return 1.0 if self.total == 0 else self.complete / self.total

    @property
    def record_completeness(self) -> float:
        """FR-1305: complete records, measured over *completed* reviews only."""
        return 1.0 if self.complete == 0 else self.well_formed / self.complete


class ReviewStore:
    """File-backed store of human decisions, keyed by investigation."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, investigation_id: str) -> Path:
        return self.directory / f"{investigation_id}.json"

    # --- writing --------------------------------------------------------- #
    def open_escalation(self, investigation_id: str, *, reason: str = "") -> HumanDecision:
        """Record that a human was asked, before anyone has answered (FR-1108).

        Without this an unanswered escalation is indistinguishable from one that was never
        raised, and the completeness metric would silently read 100%.
        """
        existing = self.get(investigation_id)
        if existing is not None:
            return existing
        decision = HumanDecision(
            review_id=f"rev_{uuid.uuid4().hex[:12]}",
            investigation_id=investigation_id,
            reviewer_id="",
            reviewer_role="",
            reviewed_at=_now(),
            decision="pending",
            notes=reason,
        )
        self.save(decision)
        return decision

    def record(
        self,
        investigation_id: str,
        *,
        reviewer_id: str,
        reviewer_role: str,
        decision: str,
        accepted_claims: list[str] | None = None,
        rejected_claims: list[str] | None = None,
        override_reason: str | None = None,
        requested_next_check: str | None = None,
        confirmed_root_cause: str | None = None,
        remediation_outcome: str | None = None,
        notes: str = "",
    ) -> HumanDecision:
        """Close an escalation with an attributed disposition."""
        existing = self.get(investigation_id)
        record = HumanDecision(
            review_id=existing.review_id if existing else f"rev_{uuid.uuid4().hex[:12]}",
            investigation_id=investigation_id,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            reviewed_at=_now(),
            decision=decision,  # type: ignore[arg-type]
            accepted_claims=accepted_claims or [],
            rejected_claims=rejected_claims or [],
            override_reason=override_reason,
            requested_next_check=requested_next_check,
            confirmed_root_cause=confirmed_root_cause,  # type: ignore[arg-type]
            remediation_outcome=remediation_outcome,
            notes=notes,
        )
        self.save(record)
        return record

    def save(self, decision: HumanDecision) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(decision.investigation_id)
        path.write_text(
            json.dumps(decision.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        return path

    # --- reading --------------------------------------------------------- #
    def get(self, investigation_id: str) -> HumanDecision | None:
        path = self._path(investigation_id)
        if not path.exists():
            return None
        return HumanDecision.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def all(self) -> list[HumanDecision]:
        if not self.directory.exists():
            return []
        decisions: list[HumanDecision] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                decisions.append(
                    HumanDecision.model_validate(json.loads(path.read_text(encoding="utf-8")))
                )
            except Exception:  # noqa: BLE001 - a malformed record must not hide the rest
                continue
        return decisions

    def disposition_stats(self) -> DispositionStats:
        records = self.all()
        answered = [r for r in records if r.is_complete]
        return DispositionStats(
            total=len(records),
            complete=len(answered),
            pending=len(records) - len(answered),
            well_formed=sum(1 for r in answered if r.record_is_complete),
        )

    def summary(self, investigation_id: str) -> dict[str, Any]:
        """Review state for the report and the UI."""
        decision = self.get(investigation_id)
        if decision is None:
            return {"reviewed": False, "decision": None, "complete": False}
        return {
            "reviewed": True,
            "decision": decision.decision,
            "complete": decision.is_complete,
            "reviewer_role": decision.reviewer_role or None,
            "reviewed_at": decision.reviewed_at.isoformat(),
            "confirmed_root_cause": (
                decision.confirmed_root_cause.value if decision.confirmed_root_cause else None
            ),
        }
