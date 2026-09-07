"""Run journal contracts (M5.1).

The persisted report says where an investigation *ended*. The journal says how it
**moved**: which node ran, in which round, what it changed, how long it took, and which
model calls it made. That distinction is the whole point — a report can be reread, but
only a journal can be replayed.

Stage records are deliberately summaries rather than full state snapshots. A snapshot per
node would multiply the evidence payloads by the number of nodes and make the file
unreadable for the one thing it exists to answer: what changed here, and why.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JOURNAL_VERSION = "1"


class StageRecord(BaseModel):
    """One parent-graph node execution."""

    model_config = ConfigDict(extra="forbid")

    # Position in the run, assigned from the journal's own length so that replaying a
    # checkpointed node produces the same id and the reducer drops the duplicate
    # (FR-205) rather than recording the stage twice.
    stage_id: str
    sequence: int
    node: str
    round_number: int
    started_at: datetime
    duration_ms: int
    # One line a human can read without opening the delta.
    summary: str
    # Compacted view of the state keys this node changed.
    delta: dict[str, Any] = Field(default_factory=dict)
    # Completions made while this node ran, attributed to the stage that made them.
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)


class RunJournal(BaseModel):
    """The ordered story of one investigation."""

    model_config = ConfigDict(extra="forbid")

    journal_version: str = JOURNAL_VERSION
    investigation_id: str
    trace_id: str | None = None
    scenario_id: str | None = None
    # FR-1206: the journal is attributable to the components that produced it, or it is
    # an anecdote about a system that no longer exists.
    versions: dict[str, Any] = Field(default_factory=dict)
    # Where this run can be found in LangSmith, when it was traced (M5.2). ``None`` for
    # an untraced run, which is the default — a journal is never a promise of a trace.
    tracing: dict[str, Any] | None = None
    outcome: str | None = None
    release_status: str | None = None
    rounds_used: int = 0
    stages: list[StageRecord] = Field(default_factory=list)

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        """Every completion in the run, in the order the stages made them."""
        return [call for stage in self.stages for call in stage.llm_calls]

    @property
    def total_duration_ms(self) -> int:
        return sum(stage.duration_ms for stage in self.stages)

    def rounds(self) -> list[int]:
        """Distinct round numbers, in order — the axis a round slider walks."""
        seen: list[int] = []
        for stage in self.stages:
            if stage.round_number not in seen:
                seen.append(stage.round_number)
        return sorted(seen)

    def for_round(self, round_number: int) -> list[StageRecord]:
        return [s for s in self.stages if s.round_number == round_number]
