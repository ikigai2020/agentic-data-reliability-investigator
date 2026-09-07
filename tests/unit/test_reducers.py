"""Unit tests for state reducers (FR-205, FR-307 dedupe/append by stable ID)."""

from __future__ import annotations

from investigator.graph.reducers import dedupe_by, extend_unique, upsert_by
from investigator.schemas import Hypothesis, RootCauseCategory


def _h(hid: str, status: str = "active") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid,
        category=RootCauseCategory.SOURCE_DATA,
        statement="s",
        discriminating_question="q",
        rank=1,
        status=status,  # type: ignore[arg-type]
    )


def test_dedupe_by_first_write_wins() -> None:
    reduce = dedupe_by("evidence_id")
    existing = [{"evidence_id": "a"}, {"evidence_id": "b"}]
    update = [{"evidence_id": "b"}, {"evidence_id": "c"}]  # b is a replay/dup
    merged = reduce(existing, update)
    assert [m["evidence_id"] for m in merged] == ["a", "b", "c"]


def test_upsert_by_replaces_status() -> None:
    reduce = upsert_by("hypothesis_id")
    existing = [_h("H1", "active")]
    update = [_h("H1", "rejected"), _h("H2", "active")]
    merged = reduce(existing, update)
    by_id = {h.hypothesis_id: h for h in merged}
    assert by_id["H1"].status == "rejected"
    assert by_id["H2"].status == "active"


def test_extend_unique_skips_duplicates() -> None:
    assert extend_unique(["x"], ["x", "y"]) == ["x", "y"]


def test_checkpoint_replay_no_duplication() -> None:
    # Simulate replaying the same evidence batch (FR-205 idempotency, §18 acceptance).
    reduce = dedupe_by("evidence_id")
    batch = [{"evidence_id": "e1"}, {"evidence_id": "e2"}]
    state = reduce([], batch)
    state = reduce(state, batch)  # replay
    assert len(state) == 2
