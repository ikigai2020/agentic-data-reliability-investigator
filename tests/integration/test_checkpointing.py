"""Checkpointing and resume (FR-804, FR-205)."""

from __future__ import annotations

from investigator.app import _load_raw_alert, run_investigation
from investigator.config import load_config
from investigator.graph.parent_graph import build_parent_graph
from investigator.mcp_client.client import InvestigatorMCPClient
from investigator.persistence import (
    REQUIRED_CHECKPOINT_NODES,
    memory_checkpointer,
    sqlite_checkpointer,
    thread_config,
)

from ..conftest import DIAGNOSED


def _raw_alert(cfg):
    return _load_raw_alert(cfg, DIAGNOSED)


def test_every_required_checkpoint_point_is_a_real_graph_node() -> None:
    """FR-804 names the points; they must exist in the compiled graph, not just in prose."""
    nodes = set(build_parent_graph().get_graph().nodes)
    assert set(REQUIRED_CHECKPOINT_NODES).issubset(nodes)


async def test_checkpointed_run_produces_the_same_outcome() -> None:
    saver = memory_checkpointer()
    checkpointed = await run_investigation(
        DIAGNOSED, checkpointer=saver, thread_id="t-parity"
    )
    plain = await run_investigation(DIAGNOSED)

    assert checkpointed["outcome"] == plain["outcome"] == "diagnosed"
    assert (
        checkpointed["report"]["supporting_evidence_ids"]
        == plain["report"]["supporting_evidence_ids"]
    )


async def test_state_is_recoverable_from_the_checkpointer() -> None:
    saver = memory_checkpointer()
    state = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-recover")

    graph = build_parent_graph(checkpointer=saver)
    restored = await graph.aget_state({"configurable": thread_config("t-recover")})

    assert restored.values["investigation_id"] == state["investigation_id"]
    assert restored.values["outcome"] == "diagnosed"
    assert len(restored.values["evidence"]) == len(state["evidence"])
    assert restored.values["budgets"]["calls_used"] == state["budgets"]["calls_used"]


async def test_interrupted_run_resumes_without_duplicating_work() -> None:
    """FR-804/FR-205: the real resume path — pause mid-run, continue, no double-counting.

    The run is interrupted after the specialists have already spent operational calls, so
    a resume that re-ran them would be visible as duplicated evidence or an inflated
    budget.
    """
    cfg = load_config()
    saver = memory_checkpointer()
    graph = build_parent_graph(checkpointer=saver, interrupt_before=["critic_review"])
    run_config = {
        "configurable": {"app_config": cfg, "client": None, **thread_config("t-interrupt")}
    }

    async with InvestigatorMCPClient(DIAGNOSED) as client:
        run_config["configurable"]["client"] = client
        paused = await graph.ainvoke(
            {"raw_alert": _raw_alert(cfg), "scenario_id": DIAGNOSED, "risk_policy_inputs": {}},
            config=run_config,
        )
        # Interrupted before the Critic: evidence collected, no outcome decided yet.
        assert paused.get("outcome") is None
        assert paused["evidence"]
        calls_at_pause = paused["budgets"]["calls_used"]
        evidence_at_pause = [e.evidence_id for e in paused["evidence"]]

        resumed = await graph.ainvoke(None, config=run_config)

    ids = [e.evidence_id for e in resumed["evidence"]]
    assert len(ids) == len(set(ids)), "evidence duplicated across the resume"
    assert set(evidence_at_pause).issubset(ids), "evidence lost across the resume"
    assert resumed["budgets"]["calls_used"] >= calls_at_pause
    assert resumed["outcome"] == "diagnosed"

    # The resumed run matches an uninterrupted one exactly.
    plain = await run_investigation(DIAGNOSED)
    assert sorted(ids) == sorted(e.evidence_id for e in plain["evidence"])
    assert resumed["budgets"]["calls_used"] == plain["budgets"]["calls_used"]


async def test_reinvoking_a_finished_thread_never_duplicates_evidence() -> None:
    """Re-running a completed thread continues it; the reducers still keep IDs unique.

    This is not resume — LangGraph starts a fresh superstep over the retained state — but
    it is the easiest way to accidentally double-count evidence, so it is pinned here.
    """
    saver = memory_checkpointer()
    first = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-rerun")
    second = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-rerun")

    second_ids = [e.evidence_id for e in second["evidence"]]
    assert len(second_ids) == len(set(second_ids)), "evidence duplicated on re-invocation"
    assert {e.evidence_id for e in first["evidence"]}.issubset(second_ids)
    assert second["outcome"] == first["outcome"] == "diagnosed"


async def test_separate_threads_do_not_share_state() -> None:
    saver = memory_checkpointer()
    a = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-a")
    b = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-b")
    assert a["investigation_id"] != b["investigation_id"]


async def test_sqlite_checkpointer_persists_across_graph_instances(tmp_path) -> None:
    """The durable backend must survive rebuilding the graph, not just the process."""
    db = tmp_path / "checkpoints" / "investigations.sqlite"
    async with sqlite_checkpointer(db) as saver:
        state = await run_investigation(DIAGNOSED, checkpointer=saver, thread_id="t-sqlite")
    assert db.exists()

    async with sqlite_checkpointer(db) as saver:
        graph = build_parent_graph(checkpointer=saver)
        restored = await graph.aget_state({"configurable": thread_config("t-sqlite")})

    assert restored.values["investigation_id"] == state["investigation_id"]
    assert restored.values["outcome"] == "diagnosed"
    assert [e.evidence_id for e in restored.values["evidence"]] == [
        e.evidence_id for e in state["evidence"]
    ]
