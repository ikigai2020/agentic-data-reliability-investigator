"""Closed-loop improvement backlog (FR-1207, NFR-010).

FR-1207 is a governance control disguised as a to-do list. Its two clauses do the work:

* **Each proposed change shall link to observed failure evidence and a regression test.**
  So an item without both is refused. "The prompt feels weak" is not a backlog item; "run
  inv_abc123 produced a false-confident diagnosis, covered by
  `tests/scenarios/test_x.py::test_y`" is.
* **Changes shall not be learned directly from unreviewed production traces.** So an item
  sourced from a run must name a completed human review, exactly as memory promotion does
  (NFR-010). The system does not get to decide what about itself needs fixing on the
  strength of its own unreviewed output.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ChangeKind = Literal["prompt", "routing", "guardrail", "tool", "policy", "corpus"]
BacklogStatus = Literal["proposed", "accepted", "rejected", "shipped"]

# Where a proposal came from. `human_review` and `safety_event` are attributable;
# `production_trace` deliberately is not, and is refused without a review to stand behind it.
EvidenceSource = Literal[
    "human_review", "safety_event", "monitoring_alert", "evaluation", "production_trace"
]


class BacklogRefused(RuntimeError):
    """Raised when a proposal does not meet the FR-1207 bar."""


@dataclass
class BacklogItem:
    item_id: str
    title: str
    change_kind: ChangeKind
    evidence_source: EvidenceSource
    failure_evidence: list[str]
    regression_test: str
    created_at: datetime
    status: BacklogStatus = "proposed"
    investigation_id: str | None = None
    review_id: str | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


def _check(
    *,
    failure_evidence: list[str],
    regression_test: str,
    evidence_source: EvidenceSource,
    review_id: str | None,
) -> None:
    if not failure_evidence:
        raise BacklogRefused(
            "a proposal must link to observed failure evidence — an opinion is not a defect"
        )
    if not regression_test:
        raise BacklogRefused(
            "a proposal must name the regression test that would catch this again; "
            "a fix without one is how the same bug returns"
        )
    if evidence_source == "production_trace" and not review_id:
        raise BacklogRefused(
            "changes may not be learned from unreviewed production traces (FR-1207, "
            "NFR-010) — cite the human review that examined this trace"
        )


class BacklogStore:
    """File-backed improvement backlog."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def propose(
        self,
        *,
        title: str,
        change_kind: ChangeKind,
        evidence_source: EvidenceSource,
        failure_evidence: list[str],
        regression_test: str,
        investigation_id: str | None = None,
        review_id: str | None = None,
        rationale: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BacklogItem:
        """Record a proposed change, refusing anything that fails the FR-1207 bar."""
        _check(
            failure_evidence=failure_evidence,
            regression_test=regression_test,
            evidence_source=evidence_source,
            review_id=review_id,
        )
        item = BacklogItem(
            item_id=f"imp_{uuid.uuid4().hex[:10]}",
            title=title,
            change_kind=change_kind,
            evidence_source=evidence_source,
            failure_evidence=list(failure_evidence),
            regression_test=regression_test,
            created_at=datetime.now(UTC),
            investigation_id=investigation_id,
            review_id=review_id,
            rationale=rationale,
            metadata=dict(metadata or {}),
        )
        self.save(item)
        return item

    def save(self, item: BacklogItem) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{item.item_id}.json"
        path.write_text(json.dumps(item.as_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def all(self) -> list[BacklogItem]:
        if not self.directory.exists():
            return []
        items: list[BacklogItem] = []
        for path in sorted(self.directory.glob("imp_*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["created_at"] = datetime.fromisoformat(raw["created_at"])
                items.append(BacklogItem(**raw))
            except Exception:  # noqa: BLE001 - one bad record must not hide the rest
                continue
        return items

    def set_status(self, item_id: str, status: BacklogStatus) -> BacklogItem:
        for item in self.all():
            if item.item_id == item_id:
                item.status = status
                self.save(item)
                return item
        raise KeyError(f"no backlog item '{item_id}'")


def from_evaluation(report: dict[str, Any], *, regression_test: str) -> list[dict[str, Any]]:
    """Turn failing evaluation rows into draft proposals.

    Drafts, not items: each still needs a human to accept it, because the system proposing
    its own fixes from its own output is precisely what NFR-010 restricts.
    """
    drafts: list[dict[str, Any]] = []
    for row in report.get("matrix", []):
        if row.get("correct"):
            continue
        drafts.append(
            {
                "title": f"{row['scenario_id']}: {', '.join(row.get('errors', [])) or 'incorrect'}",
                "change_kind": "policy",
                "evidence_source": "evaluation",
                "failure_evidence": [
                    f"scenario={row['scenario_id']}",
                    f"expected={row.get('expected_outcome')} actual={row.get('outcome')}",
                    *(f"error={e}" for e in row.get("errors", [])),
                ],
                "regression_test": regression_test,
                "rationale": "generated from a failing evaluation row; needs human triage",
            }
        )
    return drafts
