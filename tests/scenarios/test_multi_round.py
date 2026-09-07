"""Milestone 3 exit criteria (§22).

"All four components satisfy the agent definition; competing branches are explored;
diagnosis and abstention tests pass."
"""

from __future__ import annotations

from investigator.agents.commander import graph as commander
from investigator.agents.data_investigator import graph as data_investigator
from investigator.agents.evidence_critic import graph as evidence_critic
from investigator.agents.pipeline_investigator import graph as pipeline_investigator
from investigator.app import run_investigation
from investigator.graph.parent_graph import build_parent_graph
from investigator.mcp_client.permissions import allowed_tools
from investigator.search.beam import LIVE_STATUSES

from ..conftest import DIAGNOSED, INCONCLUSIVE

# FR-800 required parent nodes, plus classify_risk from the v2.1 release gate.
FR800_NODES = {
    "initialize_investigation",
    "parse_alert",
    "verify_incident",
    "commander_generate_hypotheses",
    "retrieve_context",
    "apply_retrieval_trust_gate",
    "commander_plan_round",
    "dispatch_specialists",
    "merge_specialist_results",
    "critic_review",
    "apply_branch_policy",
    "evaluate_stop",
    "commander_compose_report",
    "validate_grounding",
    "prepare_human_escalation",
    "persist_result",
}


# --------------------------------------------------------------------------- #
# §6.1 — all four components are agents, not renamed nodes
# --------------------------------------------------------------------------- #
def test_all_four_components_have_a_role_prompt_and_a_tool_boundary() -> None:
    components = [
        (commander, frozenset()),
        (pipeline_investigator, allowed_tools("pipeline_investigator")),
        (data_investigator, allowed_tools("data_investigator")),
        (evidence_critic, frozenset()),
    ]
    for module, expected_tools in components:
        assert module.PROMPT and "[missing prompt" not in module.PROMPT, module.AGENT_NAME
        assert module.PROMPT_ID.endswith(":v1"), module.AGENT_NAME
        assert allowed_tools(module.ROLE) == expected_tools, module.AGENT_NAME

    # The two coordinating agents hold no operational tools at all.
    assert allowed_tools(commander.ROLE) == frozenset()
    assert allowed_tools(evidence_critic.ROLE) == frozenset()

    # The specialists' tool sets are disjoint — neither can reach the other's server.
    assert allowed_tools("pipeline_investigator").isdisjoint(allowed_tools("data_investigator"))


def test_each_specialist_and_the_critic_have_their_own_working_loop() -> None:
    """§6.1 requires local working state and a reasoning-and-action loop, not one node."""
    for build in (pipeline_investigator.build, data_investigator.build, evidence_critic.build):
        nodes = set(build().get_graph().nodes)
        assert len(nodes - {"__start__", "__end__"}) >= 5, nodes


def test_parent_graph_has_the_full_fr800_node_set() -> None:
    nodes = set(build_parent_graph().get_graph().nodes)
    assert FR800_NODES.issubset(nodes)
    assert "classify_risk" in nodes


# --------------------------------------------------------------------------- #
# Competing branches are explored
# --------------------------------------------------------------------------- #
async def test_competing_branches_are_explored_in_parallel() -> None:
    state = await run_investigation(DIAGNOSED)
    branches = state["branches"]

    # One branch per hypothesis, several actually investigated.
    assert len(branches) == len(state["hypotheses"])
    investigated = [b for b in branches if b.action_history]
    assert len(investigated) >= 2, "only one branch was ever explored"

    # More than one root-cause category was tested with current evidence.
    tested_categories = {
        h.category
        for h in state["hypotheses"]
        if any(
            h.hypothesis_id in e.supports or h.hypothesis_id in e.contradicts
            for e in state["evidence"]
        )
    }
    assert len(tested_categories) >= 2


async def test_every_branch_is_scored_and_dispositioned_with_a_reason() -> None:
    state = await run_investigation(DIAGNOSED)
    for branch in state["branches"]:
        assert branch.score is not None, branch.branch_id
        assert branch.status in LIVE_STATUSES | {"pruned", "closed"}
        if branch.status == "pruned":
            assert branch.prune_reason, branch.branch_id


async def test_a_falsified_competitor_is_pruned_with_that_reason() -> None:
    """Reconciliation refutes the source-data branch, and the record says so."""
    state = await run_investigation(DIAGNOSED)
    source = next(b for b in state["branches"] if b.hypothesis_id.endswith("source_data"))
    assert source.status == "pruned"
    assert "falsified" in (source.prune_reason or "")


async def test_critic_reviews_every_round() -> None:
    state = await run_investigation(DIAGNOSED)
    reviews = state["critic_reviews"]
    assert reviews
    assert len(reviews) == state["rounds_used"]
    review = reviews[-1]
    assert review.branch_scores
    assert review.strongest_hypothesis_id
    assert review.rationale_summary


async def test_critic_recommendations_never_set_the_outcome() -> None:
    """FR-903: the Critic advises; only the deterministic evaluator decides."""
    state = await run_investigation(INCONCLUSIVE)
    last = state["critic_reviews"][-1]
    assert last.stop_recommendation == "continue"  # the Critic wanted to keep going ...
    assert state["outcome"] == "inconclusive"  # ... the budget gate ended it anyway


async def test_a_disagreement_is_recorded_not_silently_resolved() -> None:
    """FR-1107 (recording half): overruling the Critic has to leave a trace."""
    state = await run_investigation(INCONCLUSIVE)
    disagreements = state["critic_disagreements"]
    assert disagreements, "the Critic was overruled with no record of it"
    assert any("deterministic evaluator returned 'inconclusive'" in d for d in disagreements)
    assert (
        state["report"]["guardrail_and_budget_events"]["critic_disagreements"] == disagreements
    )


async def test_the_report_carries_the_branch_audit_trail() -> None:
    state = await run_investigation(DIAGNOSED)
    branches = state["report"]["branches"]
    assert len(branches) == len(state["branches"])
    for entry in branches:
        assert entry["score"] is not None
        assert entry["status"]
        if entry["status"] == "pruned":
            assert entry["prune_reason"]


# --------------------------------------------------------------------------- #
# Multi-round continuation (FR-801)
# --------------------------------------------------------------------------- #
async def test_unresolved_investigation_runs_another_round() -> None:
    state = await run_investigation(INCONCLUSIVE)
    assert state["rounds_used"] >= 2
    assert {t.round_number for t in state["tasks"]} >= {1, 2}
    # Round two only re-checks branches the beam selected, never the whole field again.
    round_two = [t for t in state["tasks"] if t.round_number == 2]
    assert 0 < len(round_two) < len(state["hypotheses"])


async def test_a_later_round_never_repeats_a_tool_on_the_same_branch() -> None:
    """The loop has to make progress; re-running a check would burn budget for nothing."""
    state = await run_investigation(INCONCLUSIVE)
    for branch in state["branches"]:
        tools = [a for a in branch.action_history if not a.startswith(("reopened:", "diversity"))]
        assert len(tools) == len(set(tools)), branch.branch_id


async def test_the_loop_stops_and_says_why() -> None:
    state = await run_investigation(INCONCLUSIVE)
    assert state["should_continue"] is False
    assert state["rounds_used"] <= state["budgets"]["max_rounds"]
    assert state["budgets"]["calls_used"] <= state["budgets"]["max_operational_calls"]
    assert state["stop_reason"]


async def test_diagnosis_stops_immediately_without_spending_the_budget() -> None:
    state = await run_investigation(DIAGNOSED)
    assert state["outcome"] == "diagnosed"
    assert state["rounds_used"] == 1
    assert state["budgets"]["calls_used"] < state["budgets"]["max_operational_calls"]


# --------------------------------------------------------------------------- #
# Diagnosis and abstention still hold (M1/M2 exit criteria under M3 search)
# --------------------------------------------------------------------------- #
async def test_multi_round_search_is_deterministic() -> None:
    a = await run_investigation(INCONCLUSIVE)
    b = await run_investigation(INCONCLUSIVE)
    assert a["rounds_used"] == b["rounds_used"]
    assert a["outcome"] == b["outcome"]
    assert [x.branch_id for x in a["branches"]] == [x.branch_id for x in b["branches"]]
    assert [x.score for x in a["branches"]] == [x.score for x in b["branches"]]
    assert [x.status for x in a["branches"]] == [x.status for x in b["branches"]]


async def test_abstention_still_hands_a_human_open_questions() -> None:
    """Pruning for budget must not erase the checks a human should pick up (FR-1103)."""
    state = await run_investigation(INCONCLUSIVE)
    assert state["outcome"] == "inconclusive"
    assert state["escalation_required"] is True
    assert state["escalation_package"]["recommended_next_checks"]


async def test_memory_still_never_counts_as_proof_under_beam_search() -> None:
    """AD-005 holds through scoring: no retrieved evidence supports the diagnosis."""
    state = await run_investigation(DIAGNOSED)
    supporting = set(state["report"]["supporting_evidence_ids"])
    for e in state["evidence"]:
        if e.source_kind != "current_operational":
            assert e.evidence_id not in supporting
            assert not e.supports and not e.contradicts
