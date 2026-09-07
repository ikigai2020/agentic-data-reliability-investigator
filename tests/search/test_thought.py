"""Thought representation and candidate generation (FR-700, FR-702)."""

from __future__ import annotations

from investigator.agents.reasoning import CATEGORY_PLAN, DeterministicReasoning
from investigator.mcp_client.permissions import allowed_tools
from investigator.schemas.enums import RootCauseCategory
from investigator.search import thought

from .fixtures import hypotheses


def test_thought_carries_only_structured_fields() -> None:
    """FR-700: a thought is typed and bounded — no free-form hidden reasoning."""
    hyp = hypotheses()[0]
    proposals = thought.propose(hyp, "B-1", tried=set(), limit=2)
    assert proposals
    t = proposals[0]
    assert t.hypothesis_id == hyp.hypothesis_id
    assert t.proposed_action in CATEGORY_PLAN[hyp.category][1]
    assert t.expected_supportive_observation
    assert t.expected_weakening_observation
    assert 0 <= t.discriminating_value <= 2
    assert t.estimated_cost >= 0
    # The contract forbids extra fields, so nothing unstructured can be smuggled in.
    assert set(t.model_dump()) == {
        "thought_id",
        "branch_id",
        "hypothesis_id",
        "proposed_action",
        "expected_supportive_observation",
        "expected_weakening_observation",
        "discriminating_value",
        "estimated_cost",
    }


def test_candidate_actions_are_capped_and_skip_tried_tools() -> None:
    """FR-702: at most two candidate actions per branch, never repeating a tested one."""
    hyp = hypotheses()[0]
    first = thought.candidate_tools(hyp.category, tried=set(), limit=2)
    assert len(first) == 2

    after = thought.candidate_tools(hyp.category, tried=set(first), limit=2)
    assert set(after).isdisjoint(first)


def test_candidates_never_exceed_the_specialists_permissions() -> None:
    """The search may not propose an action the Commander would have to drop (FR-506)."""
    for category, (role, _plan) in CATEGORY_PLAN.items():
        permitted = allowed_tools(role)  # type: ignore[arg-type]
        proposed = thought.candidate_tools(category, tried=set(), limit=10)
        assert set(proposed).issubset(permitted), category


def test_falsifying_tool_outranks_a_merely_corroborating_one() -> None:
    """A tool that can refute a hypothesis is worth more than one that can only agree."""
    assert (
        thought.discriminating_value(
            RootCauseCategory.TRANSFORMATION_LOGIC, "compare_source_and_target"
        )
        == 2
    )
    assert (
        thought.discriminating_value(RootCauseCategory.INFRASTRUCTURE, "get_execution_logs") == 1
    )
    assert thought.discriminating_value(RootCauseCategory.SCHEMA_CONTRACT, "get_task_failures") == 0


def test_discriminating_tables_match_the_reasoning_rules() -> None:
    """The FR-703 inputs must not drift from the interpretation rules they describe.

    Every tool the tables claim can support or falsify a category must be a tool the
    reasoning engine actually knows how to interpret.
    """
    reasoning = DeterministicReasoning()
    for table in (thought.FALSIFYING_TOOLS, thought.CORROBORATING_TOOLS):
        for category, tools in table.items():
            for tool in tools:
                assert hasattr(reasoning, f"_interpret_{tool}"), (category, tool)

    # A tool that can falsify a category is always worth the full two points, whether or
    # not it can also corroborate: `get_pipeline_run_status` refutes an infrastructure
    # cause when the run succeeded, but a failure never specifically implicates
    # infrastructure — purely falsifying tools are legitimate and still discriminating.
    for category, falsifying in thought.FALSIFYING_TOOLS.items():
        for tool in falsifying:
            assert thought.discriminating_value(category, tool) == 2, (category, tool)

    # Every category the taxonomy defines is covered by both tables.
    assert set(thought.FALSIFYING_TOOLS) == set(RootCauseCategory)
    assert set(thought.CORROBORATING_TOOLS) == set(RootCauseCategory)


def test_proposals_are_ordered_most_discriminating_first() -> None:
    hyp = next(h for h in hypotheses() if h.category is RootCauseCategory.ORCHESTRATION)
    proposals = thought.propose(hyp, "B-1", tried=set(), limit=3)
    values = [t.discriminating_value for t in proposals]
    assert values == sorted(values, reverse=True)
