"""Incident Commander agent (FR-100).

The Commander coordinates; it has no operational MCP tools. Its reasoning steps run as
parent-graph nodes (FR-800), so this module exposes the Commander's capabilities as a
small class the parent graph calls: hypothesis generation and bounded task planning.
Report synthesis lives in :mod:`investigator.reporting.report`.
"""

from __future__ import annotations

from typing import cast

from ...mcp_client.permissions import allowed_tools
from ...schemas.alert import Alert
from ...schemas.enums import AgentRole, TaskAgent
from ...schemas.hypothesis import Hypothesis
from ...schemas.task import InvestigationTask
from ..prompts import PROMPT_VERSION, load_prompt
from ..reasoning import DeterministicReasoning, ReasoningEngine

ROLE = "commander"
AGENT_NAME = "incident_commander"
PROMPT = load_prompt("commander")
PROMPT_ID = f"commander:{PROMPT_VERSION}"


class IncidentCommander:
    """Coordinating agent. Judgment comes from the injected :class:`ReasoningEngine`."""

    def __init__(self, reasoning: ReasoningEngine | None = None) -> None:
        self.reasoning: ReasoningEngine = reasoning or DeterministicReasoning()

    async def generate_hypotheses(self, alert: Alert, max_count: int) -> list[Hypothesis]:
        """FR-100.2 / FR-401: 3–5 distinct falsifiable hypotheses, ranked from alert data."""
        return await self.reasoning.generate_hypotheses(alert, max_count)

    async def plan_round(
        self,
        alert: Alert,
        hypotheses: list[Hypothesis],
        round_number: int,
        *,
        expandable: dict[str, list[str]] | None = None,
        max_candidate_actions: int = 2,
    ) -> list[InvestigationTask]:
        """FR-100.5/6/7: one bounded task per expandable hypothesis, within permissions.

        Tasks are planned in hypothesis rank order so retrieval-driven reprioritization
        (FR-604) changes the investigation order deterministically.

        Tool *selection* runs through the reasoning engine, so an LLM-backed Commander
        picks the checks it thinks are most discriminating; the deterministic engine
        returns the category plan unchanged. Either way the choice is filtered against the
        specialist's permissions before a task is created.

        ``expandable`` maps hypothesis_id -> the candidate actions beam search selected
        for this round (FR-702). It is how critic feedback and branch policy reach the
        next round of planning (FR-100.7): a hypothesis the beam did not select gets no
        task, and a hypothesis with no untried action produces no task at all — which is
        what lets the round loop terminate on its own rather than on a counter.

        When ``expandable`` is omitted the Commander plans the first ``max_candidate_actions``
        tools of each active hypothesis's plan, which is the round-one behaviour.
        """
        tasks: list[InvestigationTask] = []
        for priority, hyp in enumerate(sorted(hypotheses, key=lambda h: h.rank)):
            # The *role* is deterministic: it selects the permission set (FR-506).
            role_str, tools = self.reasoning.plan_for_category(hyp.category)
            permitted = allowed_tools(cast(AgentRole, role_str))  # FR-100.6: never over-grant

            if expandable is None:
                if hyp.status != "active":
                    continue
                default_plan = tools[:max_candidate_actions]
            else:
                if hyp.hypothesis_id not in expandable:
                    continue
                default_plan = expandable[hyp.hypothesis_id][:max_candidate_actions]

            # Which checks to assign *within* that permission set is the Commander's
            # judgment (FR-100.5). Already-tried tools are excluded so a later round
            # cannot re-run a check.
            already_tried = set(tools) - set(default_plan) if expandable else set()
            offered = [t for t in sorted(permitted) if t not in already_tried]
            candidates = await self.reasoning.select_task_tools(
                alert,
                hyp,
                role=role_str,
                permitted=offered,
                default_plan=default_plan,
                max_actions=max_candidate_actions,
            )
            # Enforced again here regardless of what any engine returned (FR-100.6).
            safe_tools = [t for t in candidates if t in permitted]
            if not safe_tools:
                continue
            tasks.append(
                InvestigationTask(
                    task_id=f"T{round_number}-{hyp.hypothesis_id}",
                    round_number=round_number,
                    assigned_agent=cast(TaskAgent, role_str),
                    hypothesis_ids=[hyp.hypothesis_id],
                    question=hyp.discriminating_question,
                    expected_discriminating_value=(
                        f"observation that confirms or refutes: {hyp.statement}"
                    ),
                    allowed_tool_names=safe_tools,
                    priority=priority,
                    status="pending",
                )
            )
        return tasks
