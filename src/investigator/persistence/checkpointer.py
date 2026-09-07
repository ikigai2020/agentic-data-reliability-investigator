"""Checkpointing and resume (FR-804, FR-205).

LangGraph persists a checkpoint after every node when a checkpointer is attached, which
covers the FR-804 checkpoint points (verification, hypothesis generation, retrieval trust
decisions, specialist completion, critic review, stop evaluation) and then some.

Resume safety comes from the reducers, not from this module: ``dedupe_by`` is
first-write-wins on ``evidence_id`` and ``upsert_by`` is keyed on stable hypothesis,
task, and branch IDs, so replaying a checkpoint can never duplicate accepted evidence
(FR-205). Budgets live in state and are restored with it.

SQLite is the durable backend; an in-memory saver is available for tests that only need
resume semantics without a file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Nodes after which FR-804 requires a checkpoint. LangGraph checkpoints after every
# node, so this tuple documents the requirement and is asserted against in the tests.
REQUIRED_CHECKPOINT_NODES: tuple[str, ...] = (
    "verify_incident",
    "commander_generate_hypotheses",
    "apply_retrieval_trust_gate",
    "dispatch_specialists",
    "critic_review",
    "evaluate_stop",
)


def thread_config(investigation_id: str) -> dict:
    """The ``configurable`` fragment identifying one resumable investigation thread."""
    return {"thread_id": investigation_id}


@asynccontextmanager
async def sqlite_checkpointer(path: str | Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a durable SQLite checkpointer, creating the parent directory if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(target)) as saver:
        yield saver


def memory_checkpointer() -> MemorySaver:
    """A process-local checkpointer with the same resume semantics, for tests."""
    return MemorySaver()
