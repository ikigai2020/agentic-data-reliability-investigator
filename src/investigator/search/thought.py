"""Thought representation and candidate-action generation (FR-700, FR-702).

A *thought* is a structured proposed step: which hypothesis it tests, the action it
proposes, what observation would support the hypothesis, what observation would weaken
it, how discriminating the action is, and what it costs. Every field is typed and
bounded — the system never persists unrestricted hidden reasoning (FR-700).

Candidate generation is deterministic and offline. The two tables below mirror the
interpretation rules in :mod:`investigator.agents.reasoning`: a tool that can *contradict*
a category can falsify that hypothesis and is therefore directly discriminating (value 2);
a tool that can only corroborate it is worth 1. ``tests/search/test_thought.py`` asserts
both tables stay consistent with the reasoning module.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from ..agents.reasoning import CATEGORY_PLAN
from ..mcp_client.permissions import allowed_tools
from ..schemas.enums import AgentRole, RootCauseCategory
from ..schemas.hypothesis import Hypothesis

C = RootCauseCategory

# Tools whose interpretation can *contradict* the category — i.e. can falsify it.
FALSIFYING_TOOLS: dict[RootCauseCategory, frozenset[str]] = {
    C.ORCHESTRATION: frozenset({"get_pipeline_run_status", "get_task_failures"}),
    C.INFRASTRUCTURE: frozenset({"get_pipeline_run_status", "get_task_failures"}),
    C.PROCESSING_STATE: frozenset({"get_pipeline_run_status", "get_processing_watermark"}),
    C.SOURCE_DATA: frozenset({"get_upstream_dependencies", "compare_source_and_target"}),
    C.TRANSFORMATION_LOGIC: frozenset(
        {"compare_source_and_target", "get_recent_transformation_changes"}
    ),
    C.DATA_QUALITY: frozenset({"get_quality_results"}),
    C.SCHEMA_CONTRACT: frozenset({"get_schema_changes"}),
    C.LEGITIMATE_BUSINESS_CHANGE: frozenset(),
}

# Tools whose interpretation can *support* the category.
CORROBORATING_TOOLS: dict[RootCauseCategory, frozenset[str]] = {
    C.ORCHESTRATION: frozenset(
        {
            "get_pipeline_run_status",
            "get_task_failures",
            "get_execution_logs",
            "get_upstream_dependencies",
        }
    ),
    C.INFRASTRUCTURE: frozenset({"get_task_failures", "get_execution_logs"}),
    C.PROCESSING_STATE: frozenset(
        {"get_pipeline_run_status", "get_processing_watermark", "get_table_metrics"}
    ),
    C.SOURCE_DATA: frozenset(
        {
            "get_upstream_dependencies",
            "get_processing_watermark",
            "get_table_metrics",
            "compare_source_and_target",
        }
    ),
    C.TRANSFORMATION_LOGIC: frozenset(
        {"compare_source_and_target", "get_recent_transformation_changes"}
    ),
    C.DATA_QUALITY: frozenset({"get_quality_results", "get_table_metrics"}),
    C.SCHEMA_CONTRACT: frozenset({"get_schema_changes"}),
    C.LEGITIMATE_BUSINESS_CHANGE: frozenset(
        {"get_recent_transformation_changes", "get_table_metrics"}
    ),
}

# One operational MCP round-trip per action (FR-702 counts calls, not wall time).
ACTION_COST = 1


class Thought(BaseModel):
    """A structured proposed step (FR-700). No free-form hidden reasoning is stored."""

    model_config = ConfigDict(extra="forbid")

    thought_id: str
    branch_id: str
    hypothesis_id: str
    proposed_action: str
    expected_supportive_observation: str
    expected_weakening_observation: str
    discriminating_value: int = Field(ge=0, le=2)
    estimated_cost: int = Field(ge=0)


def discriminating_value(category: RootCauseCategory, tool_name: str) -> int:
    """0 = unrelated, 1 = can only corroborate, 2 = can falsify (directly discriminating)."""
    if tool_name in FALSIFYING_TOOLS.get(category, frozenset()):
        return 2
    if tool_name in CORROBORATING_TOOLS.get(category, frozenset()):
        return 1
    return 0


def candidate_tools(category: RootCauseCategory, tried: set[str], limit: int) -> list[str]:
    """Next untried permitted tools for a category, in plan order, capped at ``limit``.

    ``limit`` implements FR-702's "candidate actions per branch: maximum 2". The
    permission filter is the same one the Commander applies when planning (FR-506), so
    the search never proposes an action the planner would silently drop — a branch the
    beam believes is expandable is always one the next round can actually dispatch.
    """
    role, plan = CATEGORY_PLAN[category]
    permitted = allowed_tools(cast(AgentRole, role))
    return [t for t in plan if t not in tried and t in permitted][:limit]


def propose(
    hypothesis: Hypothesis,
    branch_id: str,
    tried: set[str],
    limit: int,
) -> list[Thought]:
    """Generate the branch's candidate thoughts, most discriminating first."""
    thoughts: list[Thought] = []
    for tool in candidate_tools(hypothesis.category, tried, limit):
        thoughts.append(
            Thought(
                thought_id=f"{branch_id}:{tool}",
                branch_id=branch_id,
                hypothesis_id=hypothesis.hypothesis_id,
                proposed_action=tool,
                expected_supportive_observation=(
                    f"{tool} returns an observation consistent with: {hypothesis.statement}"
                ),
                expected_weakening_observation=(
                    f"{tool} returns an observation inconsistent with: {hypothesis.statement}"
                ),
                discriminating_value=discriminating_value(hypothesis.category, tool),
                estimated_cost=ACTION_COST,
            )
        )
    # Stable order: higher discriminating value first, then plan order (already applied).
    return sorted(thoughts, key=lambda t: (-t.discriminating_value, t.thought_id))
