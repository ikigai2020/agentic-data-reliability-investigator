"""UI view models (M5.3).

The UI's logic lives in pure functions over a persisted run, so it is tested here rather
than by clicking it. What these assert is that each tab shows the thing it exists to
show — a permission boundary, a rejected document, an advisory recommendation beside the
authoritative one — and that a missing journal degrades instead of crashing.
"""

from __future__ import annotations

from investigator.app import run_investigation
from investigator.mcp_client.permissions import OBSERVABILITY_TOOLS, PIPELINE_TOOLS
from investigator.ui import loader, views

from ..conftest import DIAGNOSED, MISLEADING_MEMORY


async def _run(scenario: str = DIAGNOSED) -> loader.RunView:
    state = await run_investigation(scenario)
    return loader.load_run(state["investigation_id"])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
async def test_a_run_loads_from_its_two_persisted_halves() -> None:
    state = await run_investigation(DIAGNOSED)
    run = loader.load_run(state["investigation_id"])

    assert run.report["outcome"] == "diagnosed"
    assert run.has_journal
    assert run.scenario_id == DIAGNOSED


async def test_recent_runs_are_listed_newest_first() -> None:
    await run_investigation(DIAGNOSED)
    runs = loader.list_runs(limit=5)

    assert runs
    assert runs == sorted(runs, key=lambda r: r.modified_at, reverse=True)
    assert all(r.has_journal for r in runs)


def test_a_run_with_no_journal_degrades_instead_of_crashing() -> None:
    """Runs persisted before journals existed must still open in the UI."""
    run = loader.load_run("inv_does_not_exist")

    assert run.has_journal is False
    assert views.header(run)["outcome"] is None
    assert views.beam_rounds(run) == []
    assert views.evidence_rows(run) == []
    assert views.memory_view(run)["decisions"] == []


# --------------------------------------------------------------------------- #
# Header and landing tab
# --------------------------------------------------------------------------- #
async def test_the_verdict_travels_in_the_header() -> None:
    head = views.header(await _run())

    assert head["outcome"] == "diagnosed"
    assert head["confidence"] == "moderate"
    assert head["risk_tier"] in {"low", "medium", "high"}
    assert head["release_status"] == "released"
    assert head["calls_used"] <= head["calls_budget"]


async def test_the_landing_tab_shows_the_verdict_and_what_supports_it() -> None:
    data = views.investigation(await _run())

    assert data["alert"]["incident_id"]
    assert data["verification"]["status"] == "verified"
    assert data["leading_hypothesis"]["category"] == "transformation_logic"
    assert len(data["supporting_evidence"]) >= 2
    assert all(e["source_kind"] == "current_operational" for e in data["supporting_evidence"])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def test_the_graph_is_read_from_the_compiled_graph_not_drawn_by_hand() -> None:
    """A hand-drawn picture drifts from the code; this one cannot."""
    from investigator.graph.parent_graph import build_parent_graph

    run = await _run()
    dot = views.orchestration_dot(run)

    for node in build_parent_graph().get_graph().nodes:
        if node not in {"__start__", "__end__"}:
            assert f'"{node}"' in dot, node
    assert dot.startswith("digraph investigation {")


async def test_the_round_slider_narrows_the_highlighted_path() -> None:
    run = await _run()
    timeline = views.stage_timeline(run, round_number=1)

    assert timeline
    assert any(s["in_round"] for s in timeline)
    assert any(not s["in_round"] for s in timeline), "round 0 stages should fall outside"
    assert [s["sequence"] for s in timeline] == sorted(s["sequence"] for s in timeline)


# --------------------------------------------------------------------------- #
# Hypotheses and beam
# --------------------------------------------------------------------------- #
async def test_every_branch_shows_its_itemized_score_and_disposition() -> None:
    rounds = views.beam_rounds(await _run())

    assert rounds
    branches = rounds[0]["branches"]
    assert branches
    for row in branches:
        assert row["disposition"]
        assert row["total"] is not None
        assert set(row["components"]) == {key for key, _, _ in views.RUBRIC_COMPONENTS}
    assert any(row["reasons"] for row in branches), "the rubric must carry its reasons"


async def test_a_pruned_branch_shows_why_it_was_cut() -> None:
    rounds = views.beam_rounds(await _run())
    pruned = [r for round_ in rounds for r in round_["branches"] if r["status"] == "pruned"]

    assert pruned
    assert all(row["prune_reason"] for row in pruned)


# --------------------------------------------------------------------------- #
# Agents — the permission boundary, made visible
# --------------------------------------------------------------------------- #
async def test_the_two_agents_that_hold_no_tools_are_shown_holding_none() -> None:
    cards = {card["agent"]: card for card in views.agent_cards(await _run())}

    assert cards["incident_commander"]["holds_no_tools"] is True
    assert cards["evidence_critic"]["holds_no_tools"] is True
    assert cards["incident_commander"]["calls_made"] == 0
    assert cards["evidence_critic"]["calls_made"] == 0


async def test_each_specialist_shows_only_its_own_server_and_tools() -> None:
    cards = {card["agent"]: card for card in views.agent_cards(await _run())}

    assert set(cards["pipeline_investigator"]["permitted_tools"]) == set(PIPELINE_TOOLS)
    assert set(cards["data_investigator"]["permitted_tools"]) == set(OBSERVABILITY_TOOLS)
    assert cards["pipeline_investigator"]["servers"] == ["pipeline_operations"]
    assert cards["data_investigator"]["servers"] == ["data_observability"]
    assert cards["pipeline_investigator"]["calls_made"] > 0


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
async def test_current_and_historical_evidence_are_distinguishable() -> None:
    run = await _run()
    rows = views.evidence_rows(run)

    assert any(r["is_current"] for r in rows)
    assert any(not r["is_current"] for r in rows), "this scenario accepts memory as context"
    assert all(r["is_current"] for r in views.evidence_rows(run, current_only=True))


async def test_only_a_live_run_can_show_raw_payloads() -> None:
    """Payloads are deliberately never written to disk (FR-1208), so a replay has none."""
    state = await run_investigation(DIAGNOSED)

    replayed = views.evidence_rows(loader.load_run(state["investigation_id"]))
    assert all(row["payload"] is None for row in replayed)

    live = views.evidence_rows(loader.view_from_state(state))
    assert any(row["payload"] for row in live)
    assert any(row["provenance"] for row in live)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
async def test_the_trust_gate_grid_shows_all_six_gates_per_document() -> None:
    memory = views.memory_view(await _run(MISLEADING_MEMORY))

    assert len(memory["gates"]) == 6
    assert memory["decisions"]
    for decision in memory["decisions"]:
        assert set(decision["gate_results"]) == set(memory["gates"])


async def test_a_rejected_document_shows_which_gate_refused_it() -> None:
    memory = views.memory_view(await _run(MISLEADING_MEMORY))
    trap = next(d for d in memory["decisions"] if d["doc_id"] == "INC-HIST-0055")

    assert trap["accepted"] is False
    assert trap["failed_gates"] == ["metadata_match"]
    assert trap["reasons"]
    assert memory["reordered"] is False, "rejected memory must not have reordered anything"


async def test_accepted_memory_shows_the_order_it_changed() -> None:
    memory = views.memory_view(await _run(DIAGNOSED))

    assert memory["order_before"] and memory["order_after"]
    assert memory["reordered"] is True
    assert memory["influence"]["kind"] == "reordered"


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
async def test_the_critic_is_shown_beside_the_authority_not_instead_of_it() -> None:
    rounds = views.critic_rounds(await _run())

    assert rounds
    for review in rounds:
        assert review["recommendation"]
        assert review["deterministic_outcome"]
        assert isinstance(review["disagrees"], bool)


async def test_the_critics_actual_findings_are_available_not_just_counts() -> None:
    rounds = views.critic_rounds(await _run())
    findings = [item for r in rounds for item in r["evidence_gaps"] + r["pruning_recommendations"]]

    assert findings
    assert all(isinstance(item, str) and item for item in findings)


async def test_recorded_disagreements_surface() -> None:
    run = await _run()
    assert isinstance(views.disagreements(run), list)


# --------------------------------------------------------------------------- #
# Traces and safety
# --------------------------------------------------------------------------- #
async def test_a_deterministic_run_reports_no_model_calls_and_no_trace() -> None:
    data = views.traces(await _run())

    assert data["llm_calls"] == []
    assert data["tracing"] is None
    assert data["versions"]["engine"] == "deterministic"
    assert data["stages"]


async def test_the_safety_view_carries_the_release_decision_and_its_reasons() -> None:
    data = views.safety(await _run())

    assert data["risk_tier"] in {"low", "medium", "high", "prohibited"}
    assert data["risk_reasons"]
    assert data["released"] is True
    assert data["blocked_tool_attempts"] == 0
    assert data["grounded"] is True
    assert data["diagnosis_checks"]


def test_acceptance_gates_render_as_pass_fail_rows() -> None:
    payload = {
        "acceptance": {
            "gates": [
                {"name": "grounding", "metric": "grounding_rate", "value": 1.0, "limit": 1.0,
                 "passed": True, "requirement": "FR-1305: 100%"}
            ]
        }
    }
    rows = views.acceptance_snapshot(payload)
    assert rows == [
        {"name": "grounding", "metric": "grounding_rate", "value": 1.0, "limit": 1.0,
         "passed": True, "requirement": "FR-1305: 100%"}
    ]
