"""A full investigation driven by the LLM engine, offline (AD-004, FR-903).

The transport is stubbed, so this exercises the real graph, the real beam policy, and the
real stopping evaluator against model-shaped reasoning — proving that swapping engines
changes judgment and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx

from investigator.agents.llm import LLMReasoning
from investigator.app import run_investigation
from investigator.config import load_config

from ..conftest import DIAGNOSED

# The observations the diagnosed scenario returns, and what a competent model would make
# of each. Keyed by tool so the stub can answer whatever the specialists actually call.
INTERPRETATIONS: dict[str, dict] = {
    "compare_source_and_target": {
        "supports": ["transformation_logic"],
        "contradicts": ["source_data"],
        "discriminating": True,
        "summary": "rows dropped at the transform stage while the source is intact",
    },
    "get_recent_transformation_changes": {
        "supports": ["transformation_logic"],
        "contradicts": [],
        "discriminating": True,
        "summary": "a recent transformation change landed in the window",
    },
    "get_upstream_dependencies": {
        "supports": [],
        "contradicts": ["source_data"],
        "discriminating": True,
        "summary": "upstream dependencies all succeeded on time",
    },
    "get_processing_watermark": {
        "supports": [],
        "contradicts": ["processing_state"],
        "discriminating": False,
        "summary": "watermark is current",
    },
    "get_pipeline_run_status": {
        "supports": [],
        "contradicts": ["orchestration", "infrastructure"],
        "discriminating": True,
        "summary": "the run succeeded and is terminal",
    },
    "get_task_failures": {
        "supports": [],
        "contradicts": ["orchestration"],
        "discriminating": False,
        "summary": "no task failures",
    },
}

HYPOTHESES = {
    "hypotheses": [
        {
            "category": "transformation_logic",
            "statement": "A filter change in orders_daily is dropping rows from orders_fact.",
            "discriminating_question": "Does source/target reconciliation explain the delta?",
        },
        {
            "category": "source_data",
            "statement": "Upstream data feeding orders_fact arrived incomplete.",
            "discriminating_question": "Did upstream dependencies deliver on time?",
        },
        {
            "category": "orchestration",
            "statement": "The orders_daily run failed or skipped a task.",
            "discriminating_question": "Did the run complete without task failures?",
        },
    ]
}


def _permitted_from(system: str) -> list[str]:
    """The tool names a prompt offered, in the order it offered them."""
    return [
        line.strip().removeprefix("- `").removesuffix("`")
        for line in system.splitlines()
        if line.strip().startswith("- `")
    ]


def _stub_transport(*, critic_recommendation: str = "diagnosed") -> httpx.MockTransport:
    """A stand-in model that answers all four agent decisions from the tables above."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]

        if "Propose the ranked hypotheses" in user:
            payload: dict = HYPOTHESES

        elif "Select the checks as JSON" in user:
            # Commander: pick the checks this stub knows how to interpret.
            offered = _permitted_from(system)
            known = [t for t in offered if t in INTERPRETATIONS]
            payload = {"tools": known[:2], "reason": "most discriminating available checks"}

        elif "Decide the next check as JSON" in user:
            # Specialist: work through what remains, most discriminating first.
            remaining = _permitted_from(system)
            payload = {"tool": remaining[0] if remaining else None, "reason": "next check"}

        elif "Return your review as JSON" in user:
            payload = {
                "bias_challenges": [],
                "evidence_gaps": [],
                "stop_recommendation": critic_recommendation,
                "rationale": "reconciliation and the change log agree independently.",
            }

        else:
            tool = user.split("Tool: ", 1)[1].split("\n", 1)[0].strip()
            payload = INTERPRETATIONS.get(
                tool,
                {"supports": [], "contradicts": [], "discriminating": False, "summary": tool},
            )

        return httpx.Response(
            200,
            json={
                "model": "stub/model:free",
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 50, "total_tokens": 450},
            },
        )

    return httpx.MockTransport(handler)


def _llm_cfg():
    base = load_config()
    return replace(base, reasoning=replace(base.reasoning, engine="llm"))


async def test_llm_engine_reaches_the_same_diagnosis() -> None:
    cfg = _llm_cfg()
    engine = LLMReasoning(cfg, "test-key", transport=_stub_transport())
    try:
        state = await run_investigation(DIAGNOSED, cfg, reasoning=engine)
    finally:
        await engine.aclose()

    assert state["outcome"] == "diagnosed"
    assert state["report"]["leading_hypothesis"]["category"] == "transformation_logic"
    assert len(state["report"]["supporting_evidence_ids"]) >= 2
    assert not state.get("reasoning_fallbacks"), state.get("reasoning_fallbacks")


async def test_deterministic_policy_is_unchanged_by_the_engine() -> None:
    """AD-004: the engine supplies judgment; permissions, budgets, and gates are code."""
    cfg = _llm_cfg()
    engine = LLMReasoning(cfg, "test-key", transport=_stub_transport())
    try:
        llm_state = await run_investigation(DIAGNOSED, cfg, reasoning=engine)
    finally:
        await engine.aclose()
    deterministic_state = await run_investigation(DIAGNOSED)

    # Same guardrails applied, whatever produced the judgments.
    assert llm_state["release_status"] == deterministic_state["release_status"]
    assert llm_state["blocked_attempts"] == deterministic_state["blocked_attempts"] == 0
    assert (
        llm_state["budgets"]["max_operational_calls"]
        == deterministic_state["budgets"]["max_operational_calls"]
    )
    assert llm_state["budgets"]["calls_used"] <= llm_state["budgets"]["max_operational_calls"]

    # Branch policy still scored and dispositioned every branch with a reason.
    for branch in llm_state["branches"]:
        assert branch.score is not None
        if branch.status == "pruned":
            assert branch.prune_reason


async def test_a_useless_model_still_abstains_rather_than_inventing_a_cause() -> None:
    """A model that supports nothing produces no evidence — and the system says so."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        if "Propose the ranked hypotheses" in user:
            payload: dict = HYPOTHESES
        elif "Select the checks as JSON" in user:
            payload = {"tools": _permitted_from(system)[:2]}
        elif "Decide the next check as JSON" in user:
            remaining = _permitted_from(system)
            payload = {"tool": remaining[0] if remaining else None}
        elif "Return your review as JSON" in user:
            payload = {"stop_recommendation": "inconclusive", "rationale": "nothing conclusive"}
        else:
            payload = {
                "supports": [],
                "contradicts": [],
                "discriminating": False,
                "summary": "unclear",
            }
        return httpx.Response(
            200,
            json={
                "model": "stub/model:free",
                "choices": [{"message": {"content": json.dumps(payload)}}],
            },
        )

    cfg = _llm_cfg()
    engine = LLMReasoning(cfg, "test-key", transport=httpx.MockTransport(handler))
    try:
        state = await run_investigation(DIAGNOSED, cfg, reasoning=engine)
    finally:
        await engine.aclose()

    assert state["outcome"] == "inconclusive"
    assert state["escalation_required"] is True
    assert state["confidence"] == "not_applicable"


async def test_a_dead_provider_falls_back_and_the_report_says_so() -> None:
    """The run completes on deterministic rules, and never pretends the model reasoned."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="upstream unavailable")
    )
    base = load_config()
    cfg = replace(
        base, reasoning=replace(base.reasoning, engine="llm", max_retries=0, timeout_seconds=1.0)
    )
    engine = LLMReasoning(cfg, "test-key", transport=transport)
    try:
        state = await run_investigation(DIAGNOSED, cfg, reasoning=engine)
    finally:
        await engine.aclose()

    assert state["outcome"] == "diagnosed"  # deterministic rules carried it
    fallbacks = state["report"]["guardrail_and_budget_events"]["reasoning_fallbacks"]
    assert fallbacks
    assert any("generate_hypotheses" in note for note in fallbacks)
    assert any("interpret:" in note for note in fallbacks)
