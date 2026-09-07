"""Every agent's own decisions run through the model (FR-100.5, FR-110/120, FR-130).

All four components are LLM-backed when `reasoning.engine: llm`:

* Commander      — hypotheses (FR-100.2) and which checks to assign (FR-100.5)
* Pipeline / Data Investigator — which check to run next, or stop (FR-110/120)
* Evidence Critic — bias challenges, evidence gaps, stop recommendation (FR-130)

These run offline against a stubbed transport, and assert the permission and authority
boundaries hold whatever the model says.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx

from investigator.agents.commander.graph import IncidentCommander
from investigator.agents.evidence_critic import graph as critic_module
from investigator.agents.llm import LLMCritic, LLMReasoning, OpenRouterClient
from investigator.agents.reasoning import DeterministicReasoning, build_critic_judgment
from investigator.config import load_config
from investigator.mcp_client.permissions import OBSERVABILITY_TOOLS, PIPELINE_TOOLS, allowed_tools
from investigator.schemas.enums import RootCauseCategory

from ..search.fixtures import H_TRANSFORM, evidence, hypotheses, orders_alert

CRITIC = critic_module.build()


def _cfg(**overrides):
    base = load_config()
    return replace(base, reasoning=replace(base.reasoning, engine="llm", **overrides))


def _transport(payload_for) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = payload_for(body["messages"][0]["content"], body["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "model": "stub/model:free",
                "choices": [{"message": {"content": json.dumps(payload)}}],
            },
        )

    return httpx.MockTransport(handler)


def _engine(payload_for, **cfg_overrides) -> LLMReasoning:
    return LLMReasoning(_cfg(**cfg_overrides), "test-key", transport=_transport(payload_for))


def _hyp(category: str = "transformation_logic"):
    return next(h for h in hypotheses() if h.category.value == category)


# --------------------------------------------------------------------------- #
# FR-100.5 — the Commander chooses which checks to assign
# --------------------------------------------------------------------------- #
async def test_commander_assigns_the_checks_the_model_chose() -> None:
    engine = _engine(lambda system, user: {"tools": ["get_quality_results", "get_table_metrics"]})
    commander = IncidentCommander(engine)
    tasks = await commander.plan_round(orders_alert(), hypotheses(), 1, max_candidate_actions=2)
    await engine.aclose()

    transform = next(t for t in tasks if t.hypothesis_ids == [H_TRANSFORM])
    assert transform.allowed_tool_names == ["get_quality_results", "get_table_metrics"]
    # ... and not the deterministic category plan.
    default_role, default_plan = DeterministicReasoning().plan_for_category(
        RootCauseCategory.TRANSFORMATION_LOGIC
    )
    assert transform.allowed_tool_names != default_plan[:2]


async def test_commander_cannot_assign_a_tool_outside_the_specialists_permissions() -> None:
    """FR-100.6/FR-506: the role's permission set is not the model's to widen."""
    engine = _engine(
        lambda system, user: {
            # A data-investigator task asking for pipeline tools, plus pure invention.
            "tools": ["get_pipeline_run_status", "get_execution_logs", "drop_table", "rm -rf /"]
        }
    )
    commander = IncidentCommander(engine)
    tasks = await commander.plan_round(orders_alert(), hypotheses(), 1, max_candidate_actions=2)
    await engine.aclose()

    for task in tasks:
        permitted = allowed_tools(task.assigned_agent)
        assert set(task.allowed_tool_names).issubset(permitted), task.task_id
        assert "drop_table" not in task.allowed_tool_names


async def test_a_data_task_never_receives_pipeline_tools_and_vice_versa() -> None:
    engine = _engine(lambda system, user: {"tools": sorted(PIPELINE_TOOLS | OBSERVABILITY_TOOLS)})
    commander = IncidentCommander(engine)
    tasks = await commander.plan_round(orders_alert(), hypotheses(), 1, max_candidate_actions=3)
    await engine.aclose()

    for task in tasks:
        if task.assigned_agent == "data_investigator":
            assert set(task.allowed_tool_names).issubset(OBSERVABILITY_TOOLS)
        else:
            assert set(task.allowed_tool_names).issubset(PIPELINE_TOOLS)


async def test_an_empty_or_useless_selection_falls_back_to_the_category_plan() -> None:
    engine = _engine(lambda system, user: {"tools": ["nonsense", "also_nonsense"]})
    commander = IncidentCommander(engine)
    tasks = await commander.plan_round(orders_alert(), hypotheses(), 1, max_candidate_actions=2)
    await engine.aclose()

    transform = next(t for t in tasks if t.hypothesis_ids == [H_TRANSFORM])
    _role, plan = DeterministicReasoning().plan_for_category(RootCauseCategory.TRANSFORMATION_LOGIC)
    assert transform.allowed_tool_names == plan[:2]


async def test_the_commander_is_offered_its_whole_permitted_set() -> None:
    """Judgment needs options: the model sees every tool the specialist may call."""
    seen: list[str] = []

    def payload_for(system: str, user: str) -> dict:
        seen.extend(
            line.strip().removeprefix("- `").removesuffix("`")
            for line in system.splitlines()
            if line.strip().startswith("- `")
        )
        return {"tools": []}

    engine = _engine(payload_for)
    commander = IncidentCommander(engine)
    await commander.plan_round(orders_alert(), hypotheses(), 1, max_candidate_actions=2)
    await engine.aclose()

    assert OBSERVABILITY_TOOLS.issubset(set(seen))
    assert PIPELINE_TOOLS.issubset(set(seen))


# --------------------------------------------------------------------------- #
# FR-110/120 — the specialist selects its next check, and may stop early
# --------------------------------------------------------------------------- #
async def test_specialist_selects_a_check_from_what_remains() -> None:
    engine = _engine(lambda system, user: {"tool": "get_recent_transformation_changes"})
    choice = await engine.select_next_check(
        orders_alert(),
        question="Does reconciliation explain the delta?",
        remaining=["compare_source_and_target", "get_recent_transformation_changes"],
        observations=[],
        calls_left=3,
    )
    await engine.aclose()
    assert choice == "get_recent_transformation_changes"


async def test_specialist_can_stop_early_with_budget_left() -> None:
    """Stopping when the question is answered is a real outcome, not a failure."""
    engine = _engine(lambda system, user: {"tool": None, "reason": "already answered"})
    choice = await engine.select_next_check(
        orders_alert(),
        question="Did the run complete?",
        remaining=["get_task_failures"],
        observations=["pipeline run succeeded and is terminal"],
        calls_left=3,
    )
    await engine.aclose()
    assert choice is None


async def test_a_hallucinated_or_out_of_scope_check_is_refused() -> None:
    engine = _engine(lambda system, user: {"tool": "get_pipeline_run_status"})
    choice = await engine.select_next_check(
        orders_alert(),
        question="q",
        remaining=["compare_source_and_target"],  # the pipeline tool is not on offer
        observations=[],
        calls_left=3,
    )
    await engine.aclose()
    assert choice is None  # never call something outside the task's allow-list


async def test_no_check_is_selected_without_budget() -> None:
    engine = _engine(lambda system, user: {"tool": "compare_source_and_target"})
    choice = await engine.select_next_check(
        orders_alert(),
        question="q",
        remaining=["compare_source_and_target"],
        observations=[],
        calls_left=0,
    )
    await engine.aclose()
    assert choice is None


async def test_a_dead_provider_leaves_the_specialist_working_the_plan() -> None:
    engine = LLMReasoning(
        _cfg(max_retries=0),
        "test-key",
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down")),
    )
    choice = await engine.select_next_check(
        orders_alert(),
        question="q",
        remaining=["compare_source_and_target", "get_table_metrics"],
        observations=[],
        calls_left=3,
    )
    await engine.aclose()
    assert choice == "compare_source_and_target"  # deterministic order
    assert any("select_next_check" in note for note in engine.fallbacks)


# --------------------------------------------------------------------------- #
# FR-130 — the Critic's judgment comes from the model
# --------------------------------------------------------------------------- #
async def _review(critic_llm, ev=None, hyps=None):
    ev = ev or []
    result = await CRITIC.ainvoke(
        {
            "round_number": 1,
            "hypotheses": hyps if hyps is not None else hypotheses(),
            "evidence": ev,
            "findings": [],
            "branches": [],
            "branch_scores": {},
            "retrieval_influence": None,
            "min_supporting": 2,
            "prune_threshold": 5,
            "critic_llm": critic_llm,
        }
    )
    return result["review"]


def _critic(payload_for, **cfg_overrides) -> LLMCritic:
    cfg = _cfg(**cfg_overrides)
    client = OpenRouterClient(
        cfg=cfg.reasoning, api_key="test-key", transport=_transport(payload_for)
    )
    return LLMCritic(client, on_failure=cfg.reasoning.on_failure)


async def test_critic_surfaces_model_judgment_alongside_the_rules() -> None:
    critic = _critic(
        lambda system, user: {
            "bias_challenges": ["the lead was investigated first and nothing else was tried"],
            "evidence_gaps": ["no independent check of the source path"],
            "stop_recommendation": "continue",
            "rationale": "one tool cannot corroborate itself.",
        }
    )
    review = await _review(critic, [evidence(tool="compare_source_and_target",
                                             hypothesis_id=H_TRANSFORM)])
    await critic.client.aclose()

    joined = " ".join(review.evidence_gaps)
    assert "investigated first" in joined  # model's bias challenge is recorded
    assert "no independent check" in joined
    assert "one tool cannot corroborate itself" in review.rationale_summary


async def test_deterministic_findings_survive_the_model() -> None:
    """The rules are not replaced — a model that says nothing hides nothing."""
    critic = _critic(
        lambda system, user: {
            "bias_challenges": [],
            "evidence_gaps": [],
            "stop_recommendation": "continue",
            "rationale": "",
        }
    )
    review = await _review(critic)
    await critic.client.aclose()
    assert any("never tested" in gap for gap in review.evidence_gaps)


async def test_the_model_may_add_caution_but_never_manufacture_a_diagnosis() -> None:
    """A model cannot upgrade a cautious review into a confident one (FR-903)."""
    strong = [
        evidence(tool="compare_source_and_target", hypothesis_id=H_TRANSFORM),
        evidence(tool="get_recent_transformation_changes", hypothesis_id=H_TRANSFORM),
    ]

    cautious = _critic(lambda s, u: {"stop_recommendation": "continue", "rationale": "thin"})
    downgraded = await _review(cautious, strong)
    await cautious.client.aclose()
    assert downgraded.stop_recommendation == "continue"
    assert "deferring to the more cautious view" in downgraded.rationale_summary

    # The reverse is refused: rules say continue, the model wants a diagnosis.
    eager = _critic(lambda s, u: {"stop_recommendation": "diagnosed", "rationale": "looks fine"})
    upgraded = await _review(eager, [])
    await eager.client.aclose()
    assert upgraded.stop_recommendation != "diagnosed"
    assert "may not upgrade caution to a diagnosis" in upgraded.rationale_summary


async def test_an_invalid_recommendation_is_ignored() -> None:
    critic = _critic(lambda s, u: {"stop_recommendation": "SHIP IT", "rationale": "trust me"})
    review = await _review(critic, [])
    await critic.client.aclose()
    assert review.stop_recommendation in {"continue", "diagnosed", "inconclusive"}


async def test_a_dead_provider_leaves_a_fully_rule_based_review() -> None:
    critic = _critic(lambda s, u: {}, max_retries=0)
    critic.client.transport = httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
    critic.client._client = None  # rebuild with the failing transport
    review = await _review(critic)
    await critic.client.aclose()

    assert review.review_id  # the review still happened
    assert any("never tested" in gap for gap in review.evidence_gaps)
    assert any("critic_review" in note for note in critic.fallbacks)


def test_critic_judgment_is_absent_on_the_deterministic_engine() -> None:
    assert build_critic_judgment(load_config(), DeterministicReasoning()) is None


def test_critic_judgment_reuses_the_reasoning_engines_client() -> None:
    """One HTTP client and one usage ledger per investigation, Critic included."""
    engine = LLMReasoning(_cfg(), "test-key")
    judgment = build_critic_judgment(_cfg(), engine)
    assert isinstance(judgment, LLMCritic)
    assert judgment.client is engine.client
