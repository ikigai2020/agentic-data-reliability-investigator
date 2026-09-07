"""Critic vs deterministic-control disagreement (FR-1107)."""

from __future__ import annotations

from investigator.app import run_investigation
from investigator.graph.risk import classify

from ..conftest import DIAGNOSED, INCONCLUSIVE


def _classify(**policy_inputs):
    return classify(
        severity="low",
        outcome="diagnosed",
        diagnosis_checks={"no_critical_current_contradiction": True},
        failures=[],
        policy_inputs=policy_inputs,
    )


# --------------------------------------------------------------------------- #
# The release gate honours an unresolved objection
# --------------------------------------------------------------------------- #
def test_an_unresolved_disagreement_routes_to_human_review() -> None:
    """FR-1107: never force a diagnosis past a standing Critic objection."""
    assessment = _classify(unresolved_critic_disagreement=True)
    assert assessment.tier == "high"
    assert assessment.release_status == "human_review_required"
    assert any("Critic disagreement unresolved" in r for r in assessment.reasons)


def test_a_clean_run_still_releases() -> None:
    assert _classify().release_status == "released"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
async def test_a_disagreement_is_always_recorded() -> None:
    state = await run_investigation(INCONCLUSIVE)
    assert state["critic_disagreements"]
    assert (
        state["report"]["guardrail_and_budget_events"]["critic_disagreements"]
        == state["critic_disagreements"]
    )


async def test_a_conservative_disagreement_does_not_escalate() -> None:
    """The Critic wanting more work while the system already abstains is not a conflict.

    Escalating it would cry wolf on every budget-limited run, which is most of them.
    """
    state = await run_investigation(INCONCLUSIVE)
    last = state["critic_reviews"][-1]

    assert last.stop_recommendation == "continue"
    assert state["outcome"] == "inconclusive"  # they disagree ...
    assert state["risk_policy_inputs"].get("unresolved_critic_disagreement") is not True
    assert state["risk_tier"] in {"low", "medium"}  # ... but nothing was forced


async def test_the_extra_check_budget_is_bounded() -> None:
    """FR-1107 buys *one* additional check, so a standing conflict cannot loop."""
    for scenario in (DIAGNOSED, INCONCLUSIVE):
        state = await run_investigation(scenario)
        assert state.get("disagreement_checks_used", 0) <= 1
        assert state["rounds_used"] <= state["budgets"]["max_rounds"]


async def test_an_agreed_diagnosis_releases_without_a_human() -> None:
    state = await run_investigation(DIAGNOSED)
    assert state["outcome"] == "diagnosed"
    assert state["critic_reviews"][-1].stop_recommendation == "diagnosed"
    assert state["release_status"] == "released"
    assert not state.get("escalation_package")
