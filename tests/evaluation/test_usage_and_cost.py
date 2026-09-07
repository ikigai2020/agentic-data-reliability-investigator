"""Token usage and cost (FR-1203, FR-1302).

These metrics used to be hard-coded to zero, which made them look measured when nothing
was measuring them. The path under test is engine ledger → run journal → harness →
metric set, and it is exercised with a stub engine so it can be verified without a key,
a model, or a network (NFR-001, FR-1304).
"""

from __future__ import annotations

from dataclasses import replace

from investigator.agents.llm.client import LLMCall
from investigator.agents.reasoning import DeterministicReasoning
from investigator.config import load_config
from investigator.evaluation import evaluate_scenarios
from investigator.evaluation.pricing import estimate, rate_for

from ..conftest import DIAGNOSED

RATES = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}


class _MeteredEngine(DeterministicReasoning):
    """Deterministic reasoning that reports usage, as an LLM-backed engine would.

    One call is handed over per drain, so the per-stage attribution the journal performs
    is exercised rather than bypassed.
    """

    def __init__(self, calls: int = 3) -> None:
        self.pending = [
            LLMCall(
                purpose=f"interpret:{i}",
                model="gpt-4o-mini",
                latency_ms=120,
                prompt_tokens=1000,
                completion_tokens=100,
                total_tokens=1100,
            )
            for i in range(calls)
        ]

    def drain_calls(self) -> list[LLMCall]:
        return [self.pending.pop(0)] if self.pending else []


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def test_a_dated_model_id_matches_its_configured_family() -> None:
    """Providers append dated suffixes; a point release must not silently unprice a run."""
    assert rate_for("gpt-4o-mini", RATES) == RATES["gpt-4o-mini"]
    assert rate_for("gpt-4o-mini-2024-07-18", RATES) == RATES["gpt-4o-mini"]
    assert rate_for("claude-sonnet-5", RATES) is None


def test_the_longest_configured_prefix_wins() -> None:
    rates = {"gpt-4o": {"input": 2.5, "output": 10.0}, "gpt-4o-mini": {"input": 0.15,
                                                                      "output": 0.6}}
    assert rate_for("gpt-4o-mini-2024-07-18", rates) == rates["gpt-4o-mini"]


def test_cost_is_computed_from_reported_usage() -> None:
    calls = [{"model": "gpt-4o-mini", "prompt_tokens": 1_000_000, "completion_tokens": 0}]
    cost, unpriced = estimate(calls, RATES)
    assert cost == 0.15
    assert unpriced == []


def test_an_unpriced_model_is_named_rather_than_silently_zeroed() -> None:
    """'Cost 0.00' must never be mistakable for 'cost was measured as zero'."""
    calls = [{"model": "some-new-model", "prompt_tokens": 5000, "completion_tokens": 500}]
    cost, unpriced = estimate(calls, RATES)
    assert cost == 0.0
    assert unpriced == ["some-new-model"]


def test_no_rates_ship_by_default() -> None:
    """A stale hard-coded price produces a confident wrong number."""
    assert load_config().pricing == {}


# --------------------------------------------------------------------------- #
# End to end through the harness
# --------------------------------------------------------------------------- #
async def test_usage_reaches_the_metric_set() -> None:
    cfg = replace(load_config(), pricing=RATES)
    report = await evaluate_scenarios(
        cfg, scenario_ids=[DIAGNOSED], reasoning=_MeteredEngine(calls=3)
    )
    payload = report.as_dict()

    # 3 calls × 1100 tokens, priced at 1000 prompt + 100 completion each.
    assert payload["metrics"]["mean_tokens"] == 3300.0
    assert payload["metrics"]["mean_llm_calls"] == 3.0
    assert payload["metrics"]["cost_per_investigation_usd"] > 0
    assert payload["matrix"][0]["total_tokens"] == 3300
    assert "unpriced_models" not in payload


async def test_an_unpriced_run_reports_tokens_and_names_the_model() -> None:
    report = await evaluate_scenarios(
        cfg := load_config(), scenario_ids=[DIAGNOSED], reasoning=_MeteredEngine(calls=2)
    )
    payload = report.as_dict()

    assert cfg.pricing == {}
    assert payload["metrics"]["mean_tokens"] == 2200.0
    assert payload["metrics"]["cost_per_investigation_usd"] == 0.0
    assert payload["unpriced_models"] == ["gpt-4o-mini"]


async def test_a_run_with_no_model_calls_reports_zero_without_claiming_a_measurement() -> None:
    report = await evaluate_scenarios(scenario_ids=[DIAGNOSED])
    payload = report.as_dict()

    assert payload["metrics"]["mean_tokens"] == 0.0
    assert payload["metrics"]["mean_llm_calls"] == 0.0
    assert "unpriced_models" not in payload
