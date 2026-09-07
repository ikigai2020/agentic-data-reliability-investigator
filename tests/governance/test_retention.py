"""Trace protection and retention (FR-1208)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from investigator.observability.retention import (
    DEFAULT_POLICY,
    REDACTED,
    is_expired,
    may_read,
    policy_for,
    redact_expired,
    sweep,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def test_every_record_class_has_a_stated_rationale_and_readers() -> None:
    for rule in DEFAULT_POLICY:
        assert rule.retain_days > 0, rule.record_class
        assert rule.readable_by, rule.record_class
        assert rule.rationale, rule.record_class


def test_the_audit_trail_outlives_the_traces_it_justifies() -> None:
    assert (
        policy_for("reviewer_data").retain_days > policy_for("trace").retain_days
    )


def test_raw_payloads_expire_soonest() -> None:
    """They carry the most incidental content and the least lasting value."""
    shortest = min(DEFAULT_POLICY, key=lambda r: r.retain_days)
    assert shortest.record_class in {"evidence_payload", "prompt_and_output"}


def test_access_is_denied_by_absence_not_by_rule() -> None:
    assert may_read("data-platform-lead", "reviewer_data") is True
    assert may_read("data-platform-oncall", "reviewer_data") is False
    assert may_read("intern", "evidence_payload") is False


def test_expiry_follows_the_class_not_the_file() -> None:
    written = NOW - timedelta(days=45)
    assert is_expired(written, "evidence_payload", now=NOW) is True  # 30-day class
    assert is_expired(written, "trace", now=NOW) is False  # 90-day class


def test_redaction_keeps_identifiers_and_drops_content() -> None:
    """A report whose payloads expired must still be auditable."""
    record = {
        "investigation_id": "inv_1",
        "outcome": "diagnosed",
        "report": {
            "stop_reason": "leading hypothesis meets all criteria",
            "evidence_index": [
                {
                    "evidence_id": "ev_1",
                    "tool_name": "compare_source_and_target",
                    "source_kind": "current_operational",
                    "summary": "400 rows dropped, customer_email present",
                }
            ],
        },
    }
    redacted = redact_expired(record)

    # Identifiers survive: you can still see that ev_1 came from that tool.
    assert redacted["investigation_id"] == "inv_1"
    assert redacted["outcome"] == "diagnosed"
    item = redacted["report"]["evidence_index"][0]
    assert item["evidence_id"] == "ev_1"
    assert item["tool_name"] == "compare_source_and_target"
    # Content does not.
    assert item["summary"] == REDACTED
    assert redacted["report"]["stop_reason"] == REDACTED
    assert "customer_email" not in json.dumps(redacted)


def test_a_sweep_defaults_to_reporting_not_deleting(tmp_path) -> None:
    path = tmp_path / "inv_old.json"
    path.write_text(json.dumps({"investigation_id": "inv_old", "summary": "secret"}))
    old = (NOW - timedelta(days=200)).timestamp()
    import os

    os.utime(path, (old, old))

    affected = sweep(tmp_path, "evidence_payload", now=NOW)  # dry run
    assert affected == [path]
    assert "secret" in path.read_text()  # untouched


def test_a_sweep_redacts_rather_than_deletes(tmp_path) -> None:
    path = tmp_path / "inv_old.json"
    path.write_text(json.dumps({"investigation_id": "inv_old", "summary": "secret"}))
    old = (NOW - timedelta(days=200)).timestamp()
    import os

    os.utime(path, (old, old))

    sweep(tmp_path, "evidence_payload", now=NOW, dry_run=False)
    raw = json.loads(path.read_text())

    assert path.exists()  # the file survives, so evidence references do not dangle
    assert raw["investigation_id"] == "inv_old"
    assert raw["summary"] == REDACTED
    assert raw["_retention"]["record_class"] == "evidence_payload"


def test_a_recent_record_is_left_alone(tmp_path) -> None:
    path = tmp_path / "inv_new.json"
    path.write_text(json.dumps({"investigation_id": "inv_new", "summary": "recent"}))
    assert sweep(tmp_path, "evidence_payload", now=NOW, dry_run=False) == []
    assert "recent" in path.read_text()


def test_sweeping_a_missing_directory_is_not_an_error(tmp_path) -> None:
    assert sweep(tmp_path / "nope", "trace") == []
