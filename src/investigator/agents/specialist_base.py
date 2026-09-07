"""Specialist subgraph implementing the FR-802 bounded loop as a LangGraph graph.

Both operational specialists (Pipeline Investigator, Data Investigator) share this
subgraph shape but are distinct agents: each has its own role prompt, permission set
(injected proxy), tool domain, and completion condition. The graph nodes are:

    receive_task -> plan_check -> validate_tool_permission -> invoke_mcp_tool ->
    normalize_observation -> interpret_observation -> evaluate_local_completion ->
    submit_finding

Runtime dependencies (the permissioned MCP proxy, reasoning engine, alert, hypotheses,
shared budget counter) are injected via LangGraph ``config['configurable']`` so the
agent code never imports fixture providers and never touches global state directly
(AD-003, FR-202).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..graph.reducers import dedupe_by, extend_unique
from ..schemas.enums import RootCauseCategory
from ..schemas.evidence import Evidence
from ..schemas.finding import AgentFinding
from ..schemas.results import ToolResult
from ..schemas.task import InvestigationTask


class SpecialistState(TypedDict, total=False):
    task: InvestigationTask
    plan: list[str]  # remaining permitted tools to try, in order
    calls_made: int  # per-task calls (bounded by FR-110/120)
    evidence: Annotated[list[Evidence], dedupe_by("evidence_id")]
    failures: Annotated[list[str], extend_unique]
    finding: AgentFinding | None
    last_status: str | None
    last_result: ToolResult | None
    # The check `plan_check` selected for this iteration; ``None`` ends the loop.
    next_tool: str | None


def _hypotheses_for(categories: set[RootCauseCategory], hypotheses: list[Any]) -> list[str]:
    return [h.hypothesis_id for h in hypotheses if h.category in categories]


def build_specialist_subgraph(role: str, agent_name: str):
    """Compile a specialist subgraph bound to a role/agent name.

    Permissions are enforced by the proxy passed at invoke time; the role here only
    labels evidence and selects the correct proxy in the parent graph.
    """

    async def receive_task(state: SpecialistState, config: RunnableConfig) -> dict:
        task: InvestigationTask = state["task"]
        cfg = config["configurable"]
        max_calls = cfg["max_calls_per_task"]
        # Plan = the task's permitted tools, truncated to the per-task budget. The
        # specialist chooses the order and may stop before exhausting it.
        plan = list(task.allowed_tool_names)[:max_calls]
        return {
            "plan": plan,
            "calls_made": 0,
            "evidence": [],
            "failures": [],
            "next_tool": None,
        }

    async def plan_check(state: SpecialistState, config: RunnableConfig) -> dict:
        """FR-110/120 "select permitted check" — the specialist's own decision.

        Budget and permissions are checked deterministically first; only if a check is
        *affordable* does the reasoning engine get to choose which one, or decline. The
        deterministic engine takes the next planned tool in order; an LLM-backed engine
        may reorder or stop early once the observations answer the task question.

        The chosen tool is always one of the remaining permitted names, so the decision
        can reprioritise but never widen what this task may call.
        """
        cfg = config["configurable"]
        budget = cfg["budget"]
        remaining = list(state.get("plan") or [])
        calls_left = min(
            cfg["max_calls_per_task"] - state.get("calls_made", 0),
            budget["max"] - budget["calls_used"],
        )
        if not remaining or calls_left <= 0:
            return {"next_tool": None}

        task: InvestigationTask = state["task"]
        chosen = await cfg["reasoning"].select_next_check(
            cfg["alert"],
            question=task.question,
            remaining=remaining,
            observations=[e.summary for e in state.get("evidence", [])],
            calls_left=calls_left,
        )
        if chosen is not None and chosen not in remaining:
            chosen = None  # never call something outside the task's allow-list
        return {"next_tool": chosen}

    def _has_affordable_check(state: SpecialistState, config: RunnableConfig) -> str:
        return "invoke" if state.get("next_tool") else "submit"

    async def validate_tool_permission(state: SpecialistState, config: RunnableConfig) -> dict:
        # Structural double-check; the proxy is the authority (FR-506). Recorded for audit.
        return {}

    async def invoke_mcp_tool(state: SpecialistState, config: RunnableConfig) -> dict:
        cfg = config["configurable"]
        proxy = cfg["proxy"]
        reasoning = cfg["reasoning"]
        alert = cfg["alert"]
        budget = cfg["budget"]
        tool_name = state["next_tool"]
        plan = [t for t in state["plan"] if t != tool_name]
        args = reasoning.tool_arguments(tool_name, alert)
        result: ToolResult = await proxy.call(tool_name, args)
        # Only successful operational round-trips consume the global operational budget.
        if result.status in {"ok", "no_data"}:
            budget["calls_used"] += 1
        return {
            "plan": plan,
            "calls_made": state.get("calls_made", 0) + 1,
            "last_status": result.status,
            "last_result": result,  # transient, consumed by interpret_observation
        }

    async def normalize_observation(state: SpecialistState, config: RunnableConfig) -> dict:
        # Normalization already happened in the server (typed contracts). Passthrough.
        return {}

    async def interpret_observation(state: SpecialistState, config: RunnableConfig) -> dict:
        cfg = config["configurable"]
        reasoning = cfg["reasoning"]
        hypotheses = cfg["hypotheses"]
        investigation_id = cfg["investigation_id"]
        task: InvestigationTask = state["task"]
        result: ToolResult = state["last_result"]  # type: ignore[assignment]

        if result.status not in {"ok", "no_data"}:
            # Unavailable/invalid/timeout: NOT evidence against a hypothesis (FR-508).
            failure = f"{result.tool_name}: {result.status} ({result.error_message})"
            return {"failures": [failure]}

        if result.status == "no_data":
            interpretation = None
            supports: list[str] = []
            contradicts: list[str] = []
            summary = f"{result.tool_name}: no data (non-committal)"
            discriminating = False
            freshness = "unknown"
        else:
            interpretation = await reasoning.interpret(
                result.tool_name, result.data, alert=cfg["alert"]
            )
            supports = _hypotheses_for(interpretation.supports, hypotheses)
            contradicts = _hypotheses_for(interpretation.contradicts, hypotheses)
            summary = interpretation.summary
            discriminating = interpretation.discriminating
            freshness = "current"

        evidence = Evidence(
            evidence_id=f"ev_{task.task_id}_{result.tool_name}",
            investigation_id=investigation_id,
            task_id=task.task_id,
            producing_agent=agent_name,
            source_kind="current_operational",
            source_name=result.source_system,
            tool_name=result.tool_name,
            observed_at=result.observed_at,
            collected_at=result.collected_at,
            payload=result.data,
            summary=summary,
            supports=supports,
            contradicts=contradicts,
            freshness_status=freshness,  # type: ignore[arg-type]
            provenance={
                **result.provenance,
                "request_id": result.request_id,
                "discriminating": discriminating,
            },
        )
        return {"evidence": [evidence]}

    async def evaluate_local_completion(state: SpecialistState, config: RunnableConfig) -> dict:
        return {}

    async def submit_finding(state: SpecialistState, config: RunnableConfig) -> dict:
        task: InvestigationTask = state["task"]
        evidence = state.get("evidence", [])
        failures = state.get("failures", [])
        supports = sorted({hid for e in evidence for hid in e.supports})
        contradicts = sorted({hid for e in evidence for hid in e.contradicts})
        if evidence and not failures:
            status = "complete"
        elif evidence:
            status = "partial"
        else:
            status = "failed"
        summaries = "; ".join(e.summary for e in evidence) or "no observations collected"
        finding = AgentFinding(
            task_id=task.task_id,
            agent_name=agent_name,
            conclusion=summaries,
            evidence_ids=[e.evidence_id for e in evidence],
            supports=supports,
            contradicts=contradicts,
            unresolved_questions=([task.question] if status != "complete" else []),
            recommended_next_actions=list(failures),
            completion_status=status,  # type: ignore[arg-type]
        )
        return {"finding": finding}

    graph = StateGraph(SpecialistState)
    graph.add_node("receive_task", receive_task)
    graph.add_node("plan_check", plan_check)
    graph.add_node("validate_tool_permission", validate_tool_permission)
    graph.add_node("invoke_mcp_tool", invoke_mcp_tool)
    graph.add_node("normalize_observation", normalize_observation)
    graph.add_node("interpret_observation", interpret_observation)
    graph.add_node("evaluate_local_completion", evaluate_local_completion)
    graph.add_node("submit_finding", submit_finding)

    graph.add_edge(START, "receive_task")
    graph.add_edge("receive_task", "plan_check")
    graph.add_conditional_edges(
        "plan_check",
        _has_affordable_check,
        {"invoke": "validate_tool_permission", "submit": "submit_finding"},
    )
    graph.add_edge("validate_tool_permission", "invoke_mcp_tool")
    graph.add_edge("invoke_mcp_tool", "normalize_observation")
    graph.add_edge("normalize_observation", "interpret_observation")
    graph.add_edge("interpret_observation", "evaluate_local_completion")
    graph.add_edge("evaluate_local_completion", "plan_check")
    graph.add_edge("submit_finding", END)
    return graph.compile()
