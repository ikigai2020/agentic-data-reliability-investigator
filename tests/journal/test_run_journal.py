"""The run journal (M5.1).

The persisted report says where an investigation ended. This says how it moved. Every
assertion here is about a question the report cannot answer: which node ran, in which
round, what it changed, and what it cost.
"""

from __future__ import annotations

import json

from investigator.app import run_investigation, write_journal
from investigator.config import load_config
from investigator.graph.journal import build_journal, compact_delta, summarize
from investigator.graph.parent_graph import build_parent_graph
from investigator.persistence import memory_checkpointer
from investigator.schemas.journal import RunJournal, StageRecord

from ..conftest import DIAGNOSED, INCONCLUSIVE


def _graph_nodes() -> set[str]:
    return {
        n for n in build_parent_graph().get_graph().nodes if n not in {"__start__", "__end__"}
    }


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
async def test_every_graph_node_records_a_stage() -> None:
    """Journaling is applied at assembly, so a node cannot quietly opt out of it."""
    state = await run_investigation(DIAGNOSED)
    journaled = {stage.node for stage in state["journal"]}
    assert _graph_nodes().issubset(journaled)


async def test_stages_are_ordered_and_uniquely_identified() -> None:
    state = await run_investigation(DIAGNOSED)
    stages = state["journal"]

    sequences = [s.sequence for s in stages]
    assert sequences == sorted(sequences) == list(range(len(stages)))
    assert len({s.stage_id for s in stages}) == len(stages)


async def test_the_round_a_stage_belongs_to_is_recorded() -> None:
    """The round is the axis a replay walks, so planning opens the round it plans for."""
    state = await run_investigation(DIAGNOSED)
    journal = build_journal(state)

    setup = [s for s in journal.stages if s.node == "verify_incident"]
    planning = [s for s in journal.stages if s.node == "commander_plan_round"]
    assert setup[0].round_number == 0
    assert planning[0].round_number == 1
    assert journal.rounds() == [0, 1]
    assert all(s.round_number == 1 for s in journal.for_round(1))


async def test_a_multi_round_run_records_each_round_separately() -> None:
    state = await run_investigation(INCONCLUSIVE)
    journal = build_journal(state)

    planning = [s for s in journal.stages if s.node == "commander_plan_round"]
    assert len(planning) >= 2
    assert [s.round_number for s in planning] == sorted({s.round_number for s in planning})


# --------------------------------------------------------------------------- #
# What a stage says
# --------------------------------------------------------------------------- #
async def test_every_stage_carries_a_readable_summary() -> None:
    """A summary that just lists changed keys is the fallback, not the product."""
    state = await run_investigation(DIAGNOSED)
    for stage in state["journal"]:
        assert stage.summary
        assert not stage.summary.startswith(f"{stage.node}:"), stage.summary


async def test_the_decisive_stages_say_what_they_decided() -> None:
    state = await run_investigation(DIAGNOSED)
    by_node = {s.node: s for s in state["journal"]}

    assert "verified" in by_node["verify_incident"].summary
    assert "hypotheses" in by_node["commander_generate_hypotheses"].summary
    assert "selected" in by_node["apply_branch_policy"].summary
    assert "diagnosed" in by_node["evaluate_stop"].summary
    assert "released" in by_node["classify_risk"].summary


async def test_branch_dispositions_are_replayable_from_the_delta() -> None:
    """The beam table is a view over this: every branch, its score, and why it was cut."""
    state = await run_investigation(DIAGNOSED)
    policy = next(s for s in state["journal"] if s.node == "apply_branch_policy")

    branches = policy.delta["branches"]
    assert branches
    assert {"id", "status", "score", "prune_reason"} <= set(branches[0])
    assert any(b["status"] == "pruned" and b["prune_reason"] for b in branches)


async def test_the_trust_gate_decision_is_replayable_from_the_delta() -> None:
    """The Memory tab needs the per-gate grid, not just the verdict."""
    state = await run_investigation(DIAGNOSED)
    gate = next(s for s in state["journal"] if s.node == "apply_retrieval_trust_gate")

    decisions = gate.delta["trust_decisions"]
    assert decisions
    assert len(decisions[0]["gates"]) == 6
    assert decisions[0]["reasons"]


def test_bulky_state_is_not_repeated_in_every_stage() -> None:
    """The report is held in full elsewhere; repeating it per node would bury the delta."""
    delta = compact_delta({"report": {"huge": "x" * 10_000}, "outcome": "diagnosed"})
    assert delta["report"] == "<omitted: held in full elsewhere>"
    assert delta["outcome"] == "diagnosed"


def test_a_broken_summary_never_breaks_a_run() -> None:
    """Observability is not allowed to be the thing that fails an investigation."""
    assert summarize("verify_incident", {}, {"verification_detail": None})
    assert compact_delta({"hypotheses": ["not a hypothesis"]})["hypotheses"]


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
class _Ledger:
    """Minimal stand-in for an LLM-backed engine's drainable call ledger."""

    def __init__(self) -> None:
        self.calls = [
            type("Call", (), {"as_log_fields": lambda self: {"purpose": "interpret",
                                                             "model": "m", "latency_ms": 12,
                                                             "total_tokens": 340}})()
        ]

    def drain_calls(self):
        drained, self.calls = self.calls, []
        return drained


async def test_model_calls_are_attributed_to_the_stage_that_made_them() -> None:
    from investigator.graph.journal import journaled

    async def node(state, config):
        return {"outcome": "diagnosed"}

    engine = _Ledger()
    update = await journaled("evaluate_stop", node)(
        {"journal": [], "round_number": 2}, {"configurable": {"reasoning": engine}}
    )
    stage = update["journal"][0]

    assert stage.llm_calls[0]["purpose"] == "interpret"
    assert stage.round_number == 2
    assert engine.calls == [], "the ledger should have been drained by the wrapper"


def test_journal_totals_read_across_stages() -> None:
    def _stage(seq: int, ms: int, calls: list) -> StageRecord:
        return StageRecord(
            stage_id=f"{seq:03d}-n",
            sequence=seq,
            node="n",
            round_number=1,
            started_at="2026-09-05T00:00:00Z",
            duration_ms=ms,
            summary="s",
            llm_calls=calls,
        )

    journal = RunJournal(
        investigation_id="inv_x",
        stages=[_stage(0, 10, [{"total_tokens": 5}]), _stage(1, 32, [{"total_tokens": 7}])],
    )
    assert journal.total_duration_ms == 42
    assert len(journal.llm_calls) == 2


# --------------------------------------------------------------------------- #
# Persistence and resume
# --------------------------------------------------------------------------- #
async def test_the_journal_is_written_beside_the_report() -> None:
    cfg = load_config()
    state = await run_investigation(DIAGNOSED)
    path = cfg.journal_dir / f"{state['investigation_id']}.json"

    assert path.exists()
    journal = RunJournal.model_validate_json(path.read_text(encoding="utf-8"))
    assert journal.investigation_id == state["investigation_id"]
    assert journal.scenario_id == DIAGNOSED
    assert journal.outcome == "diagnosed"
    # FR-1206: a journal nobody can attribute is an anecdote about a system that has moved on.
    assert journal.versions["policy_digest"]
    assert journal.versions["engine"] == "deterministic"


async def test_the_persisting_stage_appears_in_its_own_journal() -> None:
    """Written after the graph returns, so the stage that writes the report is in it."""
    state = await run_investigation(DIAGNOSED)
    journal = RunJournal.model_validate_json(
        (load_config().journal_dir / f"{state['investigation_id']}.json").read_text()
    )
    assert journal.stages[-1].node == "persist_result"


async def test_a_journal_survives_a_json_round_trip() -> None:
    state = await run_investigation(DIAGNOSED)
    journal = build_journal(state)
    restored = RunJournal.model_validate(json.loads(journal.model_dump_json()))

    assert [s.stage_id for s in restored.stages] == [s.stage_id for s in journal.stages]
    assert restored.stages[0].delta == journal.stages[0].delta


async def test_resuming_a_run_does_not_record_a_stage_twice() -> None:
    """FR-205: the journal is reducer-merged, like every other accumulated collection."""
    saver = memory_checkpointer()
    first = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-journal")
    resumed = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-journal")

    ids = [s.stage_id for s in resumed["journal"]]
    assert len(ids) == len(set(ids)), "a stage was recorded twice across the resume"
    assert {s.stage_id for s in first["journal"]}.issubset(ids)


async def test_writing_a_journal_for_a_run_without_one_is_a_no_op() -> None:
    assert write_journal({"investigation_id": "inv_empty"}, load_config()) is None


# --------------------------------------------------------------------------- #
# Tracing (M5.2) — the journal is the link back to a LangSmith run
# --------------------------------------------------------------------------- #
async def test_an_untraced_run_records_no_trace(monkeypatch) -> None:
    """A journal is never a promise of a trace: absent, not an empty object."""
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    load_config.cache_clear()

    state = await run_investigation(DIAGNOSED)
    journal = RunJournal.model_validate_json(
        (load_config().journal_dir / f"{state['investigation_id']}.json").read_text()
    )
    assert journal.tracing is None
    load_config.cache_clear()


async def test_a_captured_trace_is_written_into_the_journal(tmp_path) -> None:
    """The Traces view deep-links from here, so the link has to survive persistence."""
    from investigator.observability.tracing import TraceLink

    state = await run_investigation(DIAGNOSED)
    cfg = load_config()
    write_journal(state, cfg, trace_link=TraceLink(project="p", run_id="abc-123", url="u"))

    journal = RunJournal.model_validate_json(
        (cfg.journal_dir / f"{state['investigation_id']}.json").read_text()
    )
    assert journal.tracing == {
        "provider": "langsmith",
        "project": "p",
        "run_id": "abc-123",
        "url": "u",
    }
