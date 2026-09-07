"""Parent LangGraph workflow (FR-800, FR-801, AD-001).

LangGraph is the only orchestrator. This parent graph owns coordination, shared state,
conditional routing, and deterministic control. Runtime dependencies (the live MCP
client, config) are injected via ``config['configurable']`` at invoke time so agents
never import fixtures and MCP is the sole operational boundary (AD-003).

The node set is exactly FR-800's, plus ``classify_risk`` from the v2.1 release gate.
Milestone 3 completes the loop: ``evaluate_stop`` is a conditional router that either
sends the investigation back to ``commander_plan_round`` for another round or forward to
report composition (FR-801). Specialists run in parallel (FR-203) and the graph accepts
a checkpointer for resumable runs (FR-804).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .. import evidence_view
from ..agents.commander.graph import IncidentCommander
from ..agents.data_investigator import graph as data_investigator
from ..agents.evidence_critic import graph as evidence_critic
from ..agents.pipeline_investigator import graph as pipeline_investigator
from ..agents.reasoning import build_critic_judgment, build_reasoning
from ..governance import ReviewStore
from ..observability.logging import get_logger
from ..reporting.report import (
    build_escalation_package,
    build_report,
    validate_grounding,
)
from ..retrieval import retriever, trust_gate
from ..retrieval.corpus import Corpus
from ..retrieval.influence import apply_influence, build_memory_evidence
from ..schemas.alert import Alert
from ..schemas.critic import CriticReview
from ..schemas.enums import RootCauseCategory
from ..schemas.retrieval import RetrievalInfluence
from ..schemas.state import GlobalInvestigationState
from ..search import beam, thought
from . import risk, stopping
from .journal import journaled
from .routing import route_after_parse, route_after_stop, route_after_verify
from .verification import verify_alert

_PIPELINE_SUBGRAPH = pipeline_investigator.build()
_DATA_SUBGRAPH = data_investigator.build()
_SPECIALIST_SUBGRAPHS = {
    "pipeline_investigator": _PIPELINE_SUBGRAPH,
    "data_investigator": _DATA_SUBGRAPH,
}
_CRITIC_SUBGRAPH = evidence_critic.build()


def _now() -> datetime:
    return datetime.now(UTC)


def _cfg(config: RunnableConfig) -> Any:
    return config["configurable"]["app_config"]


def _client(config: RunnableConfig):
    return config["configurable"]["client"]


def _reasoning(config: RunnableConfig):
    """The investigation's reasoning engine (AD-004).

    Injected once per run so an LLM-backed engine reuses one HTTP client and one usage
    ledger across every node. Built on demand when absent, which keeps nodes callable
    directly from tests.
    """
    injected = config["configurable"].get("reasoning")
    if injected is not None:
        return injected
    return build_reasoning(_cfg(config))


# Corpus is loaded once per (incidents, runbooks) directory pair and cached. Tests may
# inject a prebuilt corpus via config['configurable']['corpus'].
_CORPUS_CACHE: dict[str, Corpus] = {}


def _get_corpus(config: RunnableConfig, cfg: Any) -> Corpus:
    injected = config["configurable"].get("corpus")
    if injected is not None:
        return injected
    key = f"{cfg.incidents_dir}|{cfg.runbooks_dir}"
    if key not in _CORPUS_CACHE:
        _CORPUS_CACHE[key] = Corpus.from_dirs(cfg.incidents_dir, cfg.runbooks_dir)
    return _CORPUS_CACHE[key]


def _beam_policy(cfg: Any) -> beam.BeamPolicy:
    """FR-702/FR-704 policy assembled from config, never hard-coded at a call site."""
    return beam.BeamPolicy(
        beam_width=cfg.budgets.beam_width,
        max_branch_depth=cfg.budgets.max_branch_depth,
        max_candidate_actions=cfg.budgets.max_candidate_actions,
        prune_threshold=cfg.search.prune_threshold,
        min_diverse_branches=cfg.search.min_diverse_branches,
    )


def _calls_remaining(state: GlobalInvestigationState, cfg: Any) -> int:
    budgets = state.get("budgets", {})
    max_calls = int(budgets.get("max_operational_calls", cfg.budgets.max_operational_calls))
    return max_calls - int(budgets.get("calls_used", 0))


def _last_review(state: GlobalInvestigationState) -> CriticReview | None:
    """The most recent Critic review, if the Critic has run this investigation."""
    reviews = state.get("critic_reviews") or []
    return reviews[-1] if reviews else None


def _accepted_retrieval_categories(state: GlobalInvestigationState) -> set[RootCauseCategory]:
    """Categories of trust-gate-accepted memory — worth at most one rubric point (FR-703).

    Read from the trust decisions rather than the raw candidates, so memory that failed
    a gate contributes nothing to branch scoring (FR-603).
    """
    accepted = {d.doc_id for d in state.get("trust_decisions", []) if d.accepted}
    return {
        item.document.category
        for item in state.get("retrieved_items", [])
        if item.document.doc_id in accepted and item.document.category is not None
    }


def _unavailable_tools(state: GlobalInvestigationState) -> set[str]:
    """Tool names that failed this run, parsed from the recorded failure strings."""
    return {f.split(":", 1)[0].strip() for f in state.get("failures", []) if ":" in f}


# --------------------------------------------------------------------------- #
# Parent nodes
# --------------------------------------------------------------------------- #
async def initialize_investigation(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    cfg = _cfg(config)
    investigation_id = state.get("investigation_id") or f"inv_{uuid.uuid4().hex[:12]}"
    return {
        "investigation_id": investigation_id,
        "trace_id": f"trace_{uuid.uuid4().hex[:12]}",
        "round_number": 0,
        "budgets": {
            "max_operational_calls": cfg.budgets.max_operational_calls,
            "max_rounds": cfg.budgets.max_rounds,
            "calls_used": 0,
        },
        "failures": [],
        "escalation_required": False,
    }


async def parse_alert(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    try:
        alert = Alert.model_validate(state["raw_alert"])
    except Exception as exc:  # noqa: BLE001 - untrusted input (FR-300, FR-1101)
        log.log("validation", ok=False, error=str(exc))
        return {
            "alert": None,
            "parse_error": str(exc),
            "outcome": "invalid_input",
            "stop_reason": "alert failed schema validation",
            "confidence": "not_applicable",
        }
    log.log("validation", ok=True, incident_id=alert.incident_id, symptom=alert.symptom_type)
    return {"alert": alert, "scenario_id": state.get("scenario_id")}


async def verify_incident(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    cfg = _cfg(config)
    client = _client(config)
    alert: Alert = state["alert"]  # type: ignore[assignment]
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    status, detail, evidence = await verify_alert(
        alert,
        client.proxy,
        tolerance=cfg.verification.relative_tolerance,
        investigation_id=state["investigation_id"],
    )
    log.log("verification", status=status, detail=detail)
    update: dict[str, Any] = {"verification_status": status, "verification_detail": detail}
    if evidence is not None:
        update["evidence"] = [evidence]
    return update


async def commander_generate_hypotheses(
    state: GlobalInvestigationState, config: RunnableConfig
) -> dict:
    cfg = _cfg(config)
    engine = _reasoning(config)
    commander = IncidentCommander(engine)
    alert: Alert = state["alert"]  # type: ignore[assignment]
    hypotheses = await commander.generate_hypotheses(alert, cfg.hypotheses.max_count)
    # FR-100.4: the Commander also opens one investigation branch per hypothesis. The
    # root of the tree is the verified incident (FR-701).
    branches = beam.initialize_branches(hypotheses)
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    log.log(
        "hypotheses",
        count=len(hypotheses),
        categories=[h.category.value for h in hypotheses],
        branches=[b.branch_id for b in branches],
    )
    return {
        "hypotheses": hypotheses,
        "branches": branches,
        "reasoning_fallbacks": list(getattr(engine, "fallbacks", [])),
    }


async def retrieve_context(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """FR-602 retrieval routing + top-k search over the confirmed corpus."""
    cfg = _cfg(config)
    alert: Alert | None = state.get("alert")
    hypotheses = state.get("hypotheses", [])
    corpus = _get_corpus(config, cfg)
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))

    run, reason = retriever.should_retrieve(
        alert, hypotheses, corpus, enabled=cfg.retrieval.enabled
    )
    if not run or alert is None:
        log.log("retrieval", performed=False, reason=reason)
        return {
            "retrieved_items": [],
            "retrieval_influence": RetrievalInfluence(kind="not_run", detail=reason),
        }

    items = retriever.retrieve(alert, hypotheses, corpus, top_k=cfg.retrieval.top_k)
    log.log(
        "retrieval",
        performed=True,
        corpus_size=len(corpus),
        backend=corpus.backend,
        candidates=[(i.document.doc_id, round(i.score, 3)) for i in items],
    )
    return {"retrieved_items": items}


async def apply_retrieval_trust_gate(
    state: GlobalInvestigationState, config: RunnableConfig
) -> dict:
    """FR-603 trust gate + FR-604 influence. Memory never becomes current evidence."""
    cfg = _cfg(config)
    alert: Alert | None = state.get("alert")
    items = state.get("retrieved_items", [])
    hypotheses = state.get("hypotheses", [])
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))

    if alert is None or not items:
        return {}

    decisions = trust_gate.evaluate(
        items,
        alert=alert,
        hypotheses=hypotheses,
        current_evidence=state.get("evidence", []),
        cfg=cfg.retrieval,
    )
    updated_hyps, influence = apply_influence(hypotheses, items, decisions)
    memory_evidence = build_memory_evidence(
        items, decisions, investigation_id=state["investigation_id"]
    )
    log.log(
        "retrieval_trust",
        accepted=[d.doc_id for d in decisions if d.accepted],
        rejected=[d.doc_id for d in decisions if not d.accepted],
        influence=influence.kind,
        order_before=influence.order_before,
        order_after=influence.order_after,
    )
    update: dict[str, Any] = {
        "trust_decisions": decisions,
        "retrieval_influence": influence,
        "evidence": memory_evidence,
    }
    if updated_hyps:
        update["hypotheses"] = updated_hyps
    return update


async def commander_plan_round(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Plan the round's tasks for the branches the beam selected (FR-100.5/7, FR-702).

    In round one the beam *is* the Commander's own initial ranking (FR-100.4): no
    evidence exists yet, so there is nothing for the Critic to score. From round two on,
    ``beam_selected_ids`` carries the branch policy's decision into planning, which is
    how critic feedback reaches the next round (FR-100.7, FR-801).
    """
    cfg = _cfg(config)
    engine = _reasoning(config)
    commander = IncidentCommander(engine)
    alert: Alert = state["alert"]  # type: ignore[assignment]
    round_number = state.get("round_number", 0) + 1
    hypotheses = state.get("hypotheses", [])
    branches = state.get("branches", [])
    evidence = state.get("evidence", [])

    selected = list(state.get("beam_selected_ids") or [])
    if not selected:
        rank = {h.hypothesis_id: h.rank for h in hypotheses}
        live = sorted(
            (b for b in branches if b.status in beam.LIVE_STATUSES),
            key=lambda b: (rank.get(b.hypothesis_id, 10**6), b.branch_id),
        )
        selected = [b.branch_id for b in live[: cfg.budgets.beam_width]]

    by_branch = {b.branch_id: b for b in branches}
    by_hyp = {h.hypothesis_id: h for h in hypotheses}
    failed_tools = _unavailable_tools(state)
    expandable: dict[str, list[str]] = {}
    for branch_id in selected:
        branch = by_branch.get(branch_id)
        hypothesis = by_hyp.get(branch.hypothesis_id) if branch else None
        if branch is None or hypothesis is None:
            continue
        # A tool that already failed is not an untried option. Without this a hard
        # failure (bad arguments, tool absent) looks untried forever, and every
        # remaining round re-plans the same doomed call until the budget runs out.
        tried = (
            evidence_view.tools_used_for(hypothesis.hypothesis_id, evidence)
            | set(branch.action_history)
            | failed_tools
        )
        candidates = thought.candidate_tools(
            hypothesis.category, tried, cfg.budgets.max_candidate_actions
        )
        if candidates:
            expandable[hypothesis.hypothesis_id] = candidates

    tasks = await commander.plan_round(
        alert,
        hypotheses,
        round_number,
        expandable=expandable,
        max_candidate_actions=cfg.budgets.max_candidate_actions,
    )
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    log.log(
        "assignments",
        round=round_number,
        beam=selected,
        tasks=[(t.task_id, t.assigned_agent, t.allowed_tool_names) for t in tasks],
    )
    return {
        "tasks": tasks,
        "round_number": round_number,
        "beam_selected_ids": selected,
        "reasoning_fallbacks": list(getattr(engine, "fallbacks", [])),
    }


async def dispatch_specialists(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Run this round's specialist subgraphs in parallel (FR-203, FR-204).

    Admission is deterministic and happens before any concurrency: tasks are taken in
    priority order and each is given a *private* slice of the remaining global
    operational budget, so the combined expected calls of the parallel group can never
    exceed it (FR-203) and no two coroutines race on a shared counter. A task that does
    not fit is left pending for a later round rather than truncated.

    Results are merged in admission order, not completion order (FR-203), and one
    specialist raising cannot corrupt another's evidence (FR-204): failures are recorded,
    completed evidence is preserved, and the round continues.
    """
    cfg = _cfg(config)
    client = _client(config)
    reasoning = _reasoning(config)
    alert: Alert = state["alert"]  # type: ignore[assignment]
    hypotheses = state.get("hypotheses", [])
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))

    budgets: dict[str, Any] = dict(state.get("budgets", {}))
    calls_used = int(budgets.get("calls_used", 0))
    remaining = _calls_remaining(state, cfg)

    pending = [
        t
        for t in sorted(state.get("tasks", []), key=lambda t: (t.round_number, t.priority))
        if t.status == "pending" and t.round_number == state.get("round_number")
    ]

    # --- deterministic admission control (FR-203) ---
    admitted: list[tuple[Any, int]] = []
    projected = 0
    for task in pending:
        if projected >= remaining:
            break
        allowance = min(
            len(task.allowed_tool_names), cfg.budgets.max_calls_per_task, remaining - projected
        )
        if allowance <= 0:
            break
        admitted.append((task, allowance))
        projected += allowance

    deferred = [t.task_id for t in pending[len(admitted) :]]
    if deferred:
        log.log("dispatch_deferred", tasks=deferred, reason="outside remaining call budget")

    async def _run(task: Any, allowance: int) -> tuple[Any, dict, dict]:
        """Run one specialist against its own budget slice and permissioned proxy."""
        budget = {"calls_used": 0, "max": allowance}
        subgraph = _SPECIALIST_SUBGRAPHS[task.assigned_agent]
        result = await subgraph.ainvoke(
            {"task": task},
            config={
                "configurable": {
                    "proxy": client.proxy(task.assigned_agent),
                    "reasoning": reasoning,
                    "alert": alert,
                    "hypotheses": hypotheses,
                    "investigation_id": state["investigation_id"],
                    "max_calls_per_task": cfg.budgets.max_calls_per_task,
                    "budget": budget,
                }
            },
        )
        return task, budget, result

    outcomes = await asyncio.gather(
        *(_run(task, allowance) for task, allowance in admitted), return_exceptions=True
    )

    all_evidence: list[Any] = []
    all_findings: list[Any] = []
    all_failures: list[str] = []
    updated_tasks: list[Any] = []

    # Merge by admission order (stable), never by completion order (FR-203).
    for (task, _allowance), outcome in zip(admitted, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            # FR-204: isolate the failure, keep every other specialist's evidence.
            failure = f"{task.assigned_agent}: specialist subgraph raised {outcome!r}"
            all_failures.append(failure)
            updated_tasks.append(task.model_copy(update={"status": "failed"}))
            log.log("specialist_failed", task=task.task_id, error=repr(outcome))
            continue
        _task, budget, result = outcome
        all_evidence.extend(result.get("evidence", []))
        if result.get("finding") is not None:
            all_findings.append(result["finding"])
        all_failures.extend(result.get("failures", []))
        calls_used += budget["calls_used"]
        updated_tasks.append(task.model_copy(update={"status": "completed"}))

    budgets["calls_used"] = calls_used
    log.log(
        "dispatch",
        round=state.get("round_number"),
        parallel_group=[t.task_id for t, _ in admitted],
        calls_used=calls_used,
    )
    return {
        "evidence": all_evidence,
        "findings": all_findings,
        "failures": all_failures,
        "tasks": updated_tasks,
        "budgets": budgets,
        "reasoning_fallbacks": list(getattr(reasoning, "fallbacks", [])),
    }


async def merge_specialist_results(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    # Reducers already merged by stable IDs (FR-201). This node is the deterministic
    # merge boundary and audit point.
    get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-")).log(
        "merge",
        evidence=len(state.get("evidence", [])),
        findings=len(state.get("findings", [])),
    )
    return {}


async def critic_review(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Run the Evidence Critic over the round's evidence (FR-130).

    The Critic scores branches with the same FR-703 rubric the controller uses, so its
    recommendations are comparable to the decisions that follow — and visibly overruled
    when the deterministic policy disagrees (FR-903).
    """
    cfg = _cfg(config)
    engine = _reasoning(config)
    hypotheses = state.get("hypotheses", [])
    evidence = state.get("evidence", [])
    branches = beam.advance_branches(state.get("branches", []), evidence)
    policy = _beam_policy(cfg)

    preview = beam.apply_policy(
        branches,
        hypotheses,
        evidence,
        policy=policy,
        calls_remaining=_calls_remaining(state, cfg),
        retrieval_categories=_accepted_retrieval_categories(state),
        unavailable_tools=_unavailable_tools(state),
    )
    scores = {bid: float(score.total) for bid, score in preview.scores.items()}

    result = await _CRITIC_SUBGRAPH.ainvoke(
        {
            "round_number": state.get("round_number", 0),
            "hypotheses": hypotheses,
            "evidence": evidence,
            "findings": state.get("findings", []),
            "branches": branches,
            "branch_scores": scores,
            "retrieval_influence": state.get("retrieval_influence"),
            "min_supporting": cfg.stopping.min_supporting_current_observations,
            "prune_threshold": cfg.search.prune_threshold,
            "critic_llm": build_critic_judgment(cfg, engine),
        }
    )
    review = result.get("review")
    if review is None:
        return {"branches": branches}

    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    log.log(
        "critic_review",
        review_id=review.review_id,
        recommendation=review.stop_recommendation,
        unsupported_claims=len(review.unsupported_claims),
        evidence_gaps=len(review.evidence_gaps),
        prune=len(review.pruning_recommendations),
        reopen=len(review.reopening_recommendations),
    )
    return {
        "critic_reviews": [review],
        "branches": branches,
        "reasoning_fallbacks": list(getattr(engine, "fallbacks", [])),
    }


async def apply_branch_policy(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Deterministic branch policy: score, prune, protect diversity, reopen, select.

    This is where the Critic's advice becomes a decision, or does not (FR-130, FR-903).
    Hypothesis status is kept in step with branch status so the report and the escalation
    package never describe a hypothesis the search has already abandoned.
    """
    cfg = _cfg(config)
    hypotheses = state.get("hypotheses", [])
    evidence = state.get("evidence", [])
    branches = state.get("branches", [])
    policy = _beam_policy(cfg)

    decision = beam.apply_policy(
        branches,
        hypotheses,
        evidence,
        policy=policy,
        calls_remaining=_calls_remaining(state, cfg),
        retrieval_categories=_accepted_retrieval_categories(state),
        unavailable_tools=_unavailable_tools(state),
    )

    # Mirror branch dispositions onto hypothesis status (FR-301).
    status_by_hyp = {b.hypothesis_id: b.status for b in decision.branches}
    updated_hypotheses = []
    for h in hypotheses:
        branch_status = status_by_hyp.get(h.hypothesis_id)
        if branch_status == "pruned" and h.status == "active":
            falsified = evidence_view.has_critical_contradiction(h.hypothesis_id, evidence)
            updated_hypotheses.append(
                h.model_copy(update={"status": "rejected" if falsified else "weakened"})
            )
        elif branch_status == "reopened" and h.status in {"weakened", "rejected"}:
            updated_hypotheses.append(
                h.model_copy(update={"status": "active", "origin": "reopened"})
            )

    # FR-1107: deterministic policy governs hard constraints; the Critic is advisory on
    # judgment. Where the two disagree, the conflict is recorded rather than silently
    # resolved. (Routing a material disagreement to one extra discriminating check is
    # Milestone 4 — see the README.)
    review = _last_review(state)
    overruled: list[str] = []
    disagreements: list[str] = []
    if review is not None:
        round_number = state.get("round_number", 0)
        recommended_prunes = {r.split(":", 1)[0] for r in review.pruning_recommendations}
        overruled = sorted(recommended_prunes - set(decision.pruned))
        for branch_id in overruled:
            disagreements.append(
                f"round {round_number}: critic recommended pruning {branch_id}; "
                f"deterministic policy kept it"
            )
        recommended_reopens = {r.split(":", 1)[0] for r in review.reopening_recommendations}
        for branch_id in sorted(recommended_reopens - set(decision.reopened)):
            disagreements.append(
                f"round {round_number}: critic recommended reopening {branch_id}; "
                f"deterministic policy did not"
            )

    get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-")).log(
        "branch_policy",
        round=state.get("round_number"),
        selected=decision.selected_ids,
        pruned=decision.pruned,
        closed=decision.closed,
        reopened=decision.reopened,
        diversity_protected=decision.protected_ids,
        critic_prunes_overruled=overruled,
    )

    update: dict[str, Any] = {
        "branches": decision.branches,
        "beam_selected_ids": decision.selected_ids,
        "critic_disagreements": disagreements,
        # FR-703 is itemized precisely so a score can be audited rather than trusted.
        # The rubric was computed here either way; publishing it is what lets anyone
        # outside this function see *why* a branch scored what it did.
        "branch_scores": {
            branch_id: {
                "total": score.total,
                "consistency": score.consistency,
                "discriminating_value": score.discriminating_value,
                "support_independence": score.support_independence,
                "retrieved_context": score.retrieved_context,
                "cost_efficiency": score.cost_efficiency,
                "contradiction_penalty": score.contradiction_penalty,
                "reasons": score.reasons,
            }
            for branch_id, score in decision.scores.items()
        },
    }
    if updated_hypotheses:
        update["hypotheses"] = updated_hypotheses
    return update


def _has_expandable_branch(state: GlobalInvestigationState, cfg: Any) -> bool:
    """True when a selected branch still has an affordable, untried candidate action.

    This is the loop's real termination condition. The round counter is a backstop; what
    actually ends an investigation is running out of useful things to check.
    """
    hypotheses = {h.hypothesis_id: h for h in state.get("hypotheses", [])}
    branches = {b.branch_id: b for b in state.get("branches", [])}
    evidence = state.get("evidence", [])
    failed_tools = _unavailable_tools(state)
    for branch_id in state.get("beam_selected_ids") or []:
        branch = branches.get(branch_id)
        hypothesis = hypotheses.get(branch.hypothesis_id) if branch else None
        if branch is None or hypothesis is None:
            continue
        tried = (
            evidence_view.tools_used_for(hypothesis.hypothesis_id, evidence)
            | set(branch.action_history)
            | failed_tools
        )
        if thought.candidate_tools(hypothesis.category, tried, cfg.budgets.max_candidate_actions):
            return True
    return False


async def evaluate_stop(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Deterministic outcome plus the continue/stop decision (FR-900..903, FR-801).

    Only this node sets an outcome. The Critic may have recommended one; it is recorded
    for audit and carries no authority (FR-903).
    """
    cfg = _cfg(config)
    budgets = state.get("budgets", {})
    max_rounds = budgets.get("max_rounds", cfg.budgets.max_rounds)
    result = stopping.evaluate(
        verification_status=state.get("verification_status"),
        hypotheses=state.get("hypotheses", []),
        evidence=state.get("evidence", []),
        calls_used=budgets.get("calls_used", 0),
        max_calls=budgets.get("max_operational_calls", cfg.budgets.max_operational_calls),
        failures=state.get("failures", []),
        min_supporting=cfg.stopping.min_supporting_current_observations,
    )

    # FR-801 "continue -> next commander round": another round is worth running only when
    # the investigation is unresolved, in budget, and still has a check left to make.
    round_number = state.get("round_number", 0)
    calls_remaining = _calls_remaining(state, cfg)
    continue_reasons = {
        "unresolved": result.outcome == "inconclusive",
        "verified": state.get("verification_status") == "verified",
        "rounds_available": round_number < max_rounds,
        "calls_available": calls_remaining > 0,
        "expandable_branch": _has_expandable_branch(state, cfg),
    }
    should_continue = all(continue_reasons.values())

    # --- FR-1107: Critic vs deterministic control ------------------------- #
    # Deterministic policy owns hard constraints; the Critic is advisory on judgment.
    # Any conflict is recorded. A conflict is *material* only in the dangerous direction —
    # the controller about to declare `diagnosed` over a standing objection. The reverse
    # (the Critic wanting more work while the controller already abstains) is the system
    # being appropriately conservative, and escalating it would cry wolf on every
    # budget-limited run.
    review = _last_review(state)
    disagreements: list[str] = []
    checks_used = state.get("disagreement_checks_used", 0)
    policy_inputs = dict(state.get("risk_policy_inputs", {}))

    disagrees = review is not None and review.stop_recommendation != result.outcome
    material = disagrees and result.outcome == "diagnosed"

    if disagrees and review is not None:
        disagreements.append(
            f"round {round_number}: critic recommended '{review.stop_recommendation}'; "
            f"deterministic evaluator returned '{result.outcome}'"
        )
        affordable = (
            continue_reasons["expandable_branch"]
            and continue_reasons["calls_available"]
            and continue_reasons["rounds_available"]
        )
        if not should_continue and checks_used < 1 and affordable:
            # Buy one more discriminating observation to try to settle it, whichever way
            # the conflict points — an extra check can resolve it in either direction.
            should_continue = True
            checks_used += 1
            disagreements.append(
                f"round {round_number}: routing the disagreement to one additional "
                "discriminating check (budget permits)"
            )
        elif material and not should_continue:
            # It survived, or could not be bought out. Never force a diagnosis past a live
            # objection: hand it to human authority via the release gate (FR-1105).
            policy_inputs["unresolved_critic_disagreement"] = True
            disagreements.append(
                f"round {round_number}: disagreement unresolved; routing to human review "
                f"rather than releasing '{result.outcome}' over the Critic's objection"
            )

    get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-")).log(
        "stop",
        outcome=result.outcome,
        reason=result.stop_reason,
        checks=result.diagnosis_checks,
        round=round_number,
        should_continue=should_continue,
        continue_reasons=continue_reasons,
        critic_recommended=(review.stop_recommendation if review is not None else None),
        critic_disagrees=disagrees,
        material_disagreement=material,
        disagreement_checks_used=checks_used,
    )

    update: dict[str, Any] = {
        "outcome": result.outcome,
        "stop_reason": result.stop_reason,
        "confidence": result.confidence,
        "escalation_required": result.escalation_required,
        "leading_hypothesis_id": result.leading_hypothesis_id,
        "strongest_competitor_id": result.strongest_competitor_id,
        "diagnosis_checks": result.diagnosis_checks,
        "should_continue": should_continue,
        "rounds_used": round_number,
        "critic_disagreements": disagreements,
        "disagreement_checks_used": checks_used,
        "risk_policy_inputs": policy_inputs,
    }
    if result.updated_hypotheses:
        update["hypotheses"] = result.updated_hypotheses
    return update


async def commander_compose_report(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    return {"report": build_report(dict(state))}


async def validate_grounding_node(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    report = state.get("report") or {}
    evidence = state.get("evidence", [])
    grounded, issues = validate_grounding(report, evidence)
    log = get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-"))
    if grounded:
        log.log("grounding_validation", grounded=True)
        return {}
    # One bounded repair: drop dangling evidence references, then re-validate (FR-1002).
    valid_ids = {e.evidence_id for e in evidence}
    report["supporting_evidence_ids"] = [
        i for i in report.get("supporting_evidence_ids", []) if i in valid_ids
    ]
    report["contradicting_evidence_ids"] = [
        i for i in report.get("contradicting_evidence_ids", []) if i in valid_ids
    ]
    grounded, issues = validate_grounding(report, evidence)
    if grounded:
        log.log("grounding_validation", grounded=True, repaired=True)
        return {"report": report}
    # Cannot repair -> abstain/escalate (FR-1002, FR-902).
    log.log("grounding_validation", grounded=False, issues=issues)
    report["outcome"] = "inconclusive"
    return {
        "report": report,
        "outcome": "inconclusive",
        "stop_reason": "grounding validation failed; abstaining",
        "confidence": "not_applicable",
        "escalation_required": True,
    }


async def classify_risk(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Deterministic risk classification + release gate (AD-007, FR-1105, FR-1106).

    Runs immediately before report release. Model confidence is never the sole signal.
    Blocked tool attempts observed during the run are folded in as a safety metric.
    """
    alert: Alert | None = state.get("alert")
    severity = alert.severity if alert is not None else "low"
    assessment = risk.classify(
        severity=severity,
        outcome=state.get("outcome"),
        diagnosis_checks=state.get("diagnosis_checks", {}),
        failures=state.get("failures", []),
        policy_inputs=state.get("risk_policy_inputs", {}),
    )
    client = _client(config)
    blocked = getattr(client, "blocked_attempts", 0)

    report = state.get("report") or {}
    report["release_status"] = assessment.release_status
    report["risk"] = {"tier": assessment.tier, "reasons": assessment.reasons}
    report.setdefault("guardrail_and_budget_events", {})["blocked_tool_attempts"] = blocked

    get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-")).log(
        "risk_classification",
        tier=assessment.tier,
        release_status=assessment.release_status,
        reasons=assessment.reasons,
        blocked_tool_attempts=blocked,
    )

    update: dict[str, Any] = {
        "risk_tier": assessment.tier,
        "risk_reasons": assessment.reasons,
        "release_status": assessment.release_status,
        "blocked_attempts": blocked,
        "report": report,
    }
    # High/prohibited hold the diagnosis for human authority (AD-007); OR with any
    # escalation already required by the stopping/grounding gates.
    if assessment.release_status != "released":
        update["escalation_required"] = True
    return update


async def prepare_human_escalation(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    """Assemble the escalation package and open a review record (FR-1103, FR-1108).

    Opening the record at escalation time — before anyone has answered — is what makes an
    unanswered escalation visible. Without it, "nobody looked yet" is indistinguishable
    from "nobody was asked", and the FR-1305 completeness metric reads 100% on an empty
    queue.
    """
    if not state.get("escalation_required"):
        return {}
    cfg = _cfg(config)
    package = build_escalation_package(dict(state))
    investigation_id = state.get("investigation_id", "-")

    store = ReviewStore(cfg.reviews_dir)
    review = store.open_escalation(
        investigation_id, reason=state.get("stop_reason") or "escalation required"
    )
    package["review"] = store.summary(investigation_id)

    get_logger(investigation_id, state.get("trace_id", "-")).log(
        "escalation",
        prepared=True,
        reason=state.get("stop_reason"),
        review_id=review.review_id,
        disposition=review.decision,
    )
    return {"escalation_package": package}


async def persist_result(state: GlobalInvestigationState, config: RunnableConfig) -> dict:
    cfg = _cfg(config)
    outputs_dir = Path(cfg.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    release_status = state.get("release_status")
    # FR-1106 report-release gate: a prohibited investigation is not released; we persist
    # a blocked record with the policy reason (fail closed, AD-007) instead of the report.
    if release_status == "blocked":
        out: dict[str, Any] = {
            "release_status": "blocked",
            "risk": {"tier": state.get("risk_tier"), "reasons": state.get("risk_reasons")},
            "investigation_id": state.get("investigation_id"),
            "outcome_withheld": state.get("outcome"),
            "escalation_package": state.get("escalation_package"),
        }
    else:
        out = {
            "release_status": release_status,
            "report": state.get("report"),
            "escalation_package": state.get("escalation_package"),
        }
    path = outputs_dir / f"{state.get('investigation_id')}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    get_logger(state.get("investigation_id", "-"), state.get("trace_id", "-")).log(
        "persist",
        path=str(path),
        outcome=state.get("outcome"),
        release_status=release_status,
    )
    return {}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_parent_graph(
    checkpointer: Any | None = None,
    *,
    interrupt_before: list[str] | None = None,
):
    """Compile the parent graph, optionally with a checkpointer for resume (FR-804).

    ``interrupt_before`` pauses the run before the named nodes. It is the mechanism a
    resume is actually built on — and, later, the seam a human-in-the-loop approval gate
    would use.
    """
    g = StateGraph(GlobalInvestigationState)

    # Every node is wrapped once here rather than recording itself, so a node added
    # later is journaled by default instead of by remembering to (M5.1).
    for name, fn in (
        ("initialize_investigation", initialize_investigation),
        ("parse_alert", parse_alert),
        ("verify_incident", verify_incident),
        ("commander_generate_hypotheses", commander_generate_hypotheses),
        ("retrieve_context", retrieve_context),
        ("apply_retrieval_trust_gate", apply_retrieval_trust_gate),
        ("commander_plan_round", commander_plan_round),
        ("dispatch_specialists", dispatch_specialists),
        ("merge_specialist_results", merge_specialist_results),
        ("critic_review", critic_review),
        ("apply_branch_policy", apply_branch_policy),
        ("evaluate_stop", evaluate_stop),
        ("commander_compose_report", commander_compose_report),
        ("validate_grounding", validate_grounding_node),
        ("classify_risk", classify_risk),
        ("prepare_human_escalation", prepare_human_escalation),
        ("persist_result", persist_result),
    ):
        g.add_node(name, journaled(name, fn))

    g.add_edge(START, "initialize_investigation")
    g.add_edge("initialize_investigation", "parse_alert")
    g.add_conditional_edges(
        "parse_alert",
        route_after_parse,
        {"invalid": "commander_compose_report", "ok": "verify_incident"},
    )
    g.add_conditional_edges(
        "verify_incident",
        route_after_verify,
        {"verified": "commander_generate_hypotheses", "stop": "evaluate_stop"},
    )
    g.add_edge("commander_generate_hypotheses", "retrieve_context")
    g.add_edge("retrieve_context", "apply_retrieval_trust_gate")
    g.add_edge("apply_retrieval_trust_gate", "commander_plan_round")
    g.add_edge("commander_plan_round", "dispatch_specialists")
    g.add_edge("dispatch_specialists", "merge_specialist_results")
    g.add_edge("merge_specialist_results", "critic_review")
    g.add_edge("critic_review", "apply_branch_policy")
    g.add_edge("apply_branch_policy", "evaluate_stop")
    # FR-801: "continue -> next commander round". The loop is bounded by the round and
    # call budgets and, before either, by whether an expandable branch still exists.
    g.add_conditional_edges(
        "evaluate_stop",
        route_after_stop,
        {"continue": "commander_plan_round", "stop": "commander_compose_report"},
    )
    g.add_edge("commander_compose_report", "validate_grounding")
    # Risk classification + release gate run after grounding, before release (FR-1105/1106).
    g.add_edge("validate_grounding", "classify_risk")
    g.add_edge("classify_risk", "prepare_human_escalation")
    g.add_edge("prepare_human_escalation", "persist_result")
    g.add_edge("persist_result", END)

    # FR-804: with a checkpointer attached LangGraph persists after every node, covering
    # each required checkpoint point. Resume safety comes from the reducers (FR-205).
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before or [])
