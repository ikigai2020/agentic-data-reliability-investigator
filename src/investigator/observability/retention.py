"""Trace protection and retention (FR-1208, NFR-009).

Three separate questions, deliberately not conflated:

* **How long is each class of record kept?** Alerts, evidence payloads, prompts and model
  outputs, reviewer data, and evaluation records have different useful lives and different
  sensitivity. Reviewer data outlives traces because it is the audit trail; raw payloads
  expire soonest because they carry the most incidental content.
* **What survives expiry?** Deleting a trace must not orphan an audit. Expiry redacts the
  *content* while preserving identifiers and evidence references, so a report remains
  checkable after its payloads are gone.
* **Who may read it?** Least privilege is expressed as a class-to-role map, so a retention
  policy and an access policy stay one artifact rather than drifting apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

RecordClass = Literal[
    "alert",
    "evidence_payload",
    "prompt_and_output",
    "trace",
    "reviewer_data",
    "evaluation",
]

# Identifiers that must survive redaction so an audit can still be reconstructed.
AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "investigation_id",
        "trace_id",
        "scenario_id",
        "evidence_id",
        "review_id",
        "request_id",
        "task_id",
        "tool_name",
        "source_kind",
        "outcome",
        "release_status",
    }
)

REDACTED = "***expired***"


@dataclass(frozen=True)
class RetentionRule:
    record_class: RecordClass
    retain_days: int
    readable_by: tuple[str, ...]
    rationale: str


DEFAULT_POLICY: tuple[RetentionRule, ...] = (
    RetentionRule(
        "evidence_payload",
        retain_days=30,
        readable_by=("data-platform-oncall", "data-platform-lead"),
        rationale="raw tool output carries the most incidental content and ages fastest",
    ),
    RetentionRule(
        "prompt_and_output",
        retain_days=30,
        readable_by=("data-platform-lead", "security-oncall"),
        rationale="model inputs may echo untrusted alert and log content",
    ),
    RetentionRule(
        "alert",
        retain_days=90,
        readable_by=("data-platform-oncall", "data-platform-lead"),
        rationale="needed to reproduce an investigation within a quarter",
    ),
    RetentionRule(
        "trace",
        retain_days=90,
        readable_by=("data-platform-oncall", "data-platform-lead", "security-oncall"),
        rationale="operational debugging window",
    ),
    RetentionRule(
        "evaluation",
        retain_days=365,
        readable_by=("data-platform-lead",),
        rationale="regression comparisons need a year of history",
    ),
    RetentionRule(
        "reviewer_data",
        retain_days=365,
        readable_by=("data-platform-lead",),
        rationale="the human audit trail outlives the traces it justifies",
    ),
)


def policy_for(record_class: RecordClass, policy=DEFAULT_POLICY) -> RetentionRule:
    for rule in policy:
        if rule.record_class == record_class:
            return rule
    raise KeyError(f"no retention rule for '{record_class}'")


def may_read(role: str, record_class: RecordClass, policy=DEFAULT_POLICY) -> bool:
    """Least privilege (FR-1208): absence of a grant is a denial, not an oversight."""
    return role in policy_for(record_class, policy).readable_by


def is_expired(
    written_at: datetime,
    record_class: RecordClass,
    *,
    now: datetime | None = None,
    policy=DEFAULT_POLICY,
) -> bool:
    now = now or datetime.now(UTC)
    rule = policy_for(record_class, policy)
    return (now - written_at) > timedelta(days=rule.retain_days)


def redact_expired(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip content while keeping identifiers and evidence references.

    This is what makes retention safe to enforce: a report whose payloads have expired is
    still auditable — you can see *that* an observation supported a diagnosis, and which
    tool produced it, without retaining what it said.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in AUDIT_KEYS:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = redact_expired(value)
        elif isinstance(value, list):
            out[key] = [redact_expired(v) if isinstance(v, dict) else REDACTED for v in value]
        else:
            out[key] = REDACTED
    return out


def sweep(
    directory: str | Path,
    record_class: RecordClass,
    *,
    now: datetime | None = None,
    policy=DEFAULT_POLICY,
    dry_run: bool = True,
) -> list[Path]:
    """Redact expired records in place, returning the paths affected.

    Redaction rather than deletion, because deleting a persisted run would break the
    evidence references a completed review points at.
    """
    now = now or datetime.now(UTC)
    root = Path(directory)
    if not root.exists():
        return []

    affected: list[Path] = []
    for path in sorted(root.glob("*.json")):
        written_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if not is_expired(written_at, record_class, now=now, policy=policy):
            continue
        affected.append(path)
        if dry_run:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        redacted = redact_expired(raw)
        redacted["_retention"] = {
            "record_class": record_class,
            "expired_at": now.isoformat(),
            "policy_days": policy_for(record_class, policy).retain_days,
        }
        path.write_text(json.dumps(redacted, indent=2, default=str), encoding="utf-8")
    return affected
