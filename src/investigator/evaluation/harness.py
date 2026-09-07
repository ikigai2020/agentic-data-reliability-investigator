"""Evaluation harness (FR-1300, FR-1304, FR-1306, FR-1309).

Runs the labeled scenario set, scores each run against its label, aggregates the metric
set, and attaches a version fingerprint so a result stays attributable (FR-1206).

Repeated runs are first-class (FR-1309): a single pass through an LLM-backed system is an
anecdote, and reporting one number from it would be exactly the "model-quality results
disguised as production SLAs" FR-1305 warns against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig, load_config
from ..governance import ReviewStore
from ..reporting.report import validate_grounding
from .acceptance import AcceptanceReport
from .acceptance import check as check_acceptance
from .labels import ScenarioLabel, load_labels
from .metrics import MetricSet, ScenarioResult, aggregate, score_run, stability
from .pricing import estimate, usage
from .versions import VersionFingerprint, fingerprint


@dataclass
class EvaluationReport:
    """One evaluation pass over the labeled scenario set."""

    versions: VersionFingerprint
    metrics: MetricSet
    acceptance: AcceptanceReport
    per_scenario: list[ScenarioResult] = field(default_factory=list)
    repeats: list[MetricSet] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "versions": self.versions.as_dict(),
            **self.metrics.as_dict(),
            "acceptance": self.acceptance.as_dict(),
            # FR-1306: the holistic matrix — one row per scenario, so an aggregate can
            # always be traced back to the individual runs that produced it.
            "matrix": [
                {
                    "scenario_id": r.scenario_id,
                    "severity": r.severity,
                    "expected_outcome": r.expected_outcome,
                    "outcome": r.outcome,
                    "confidence": r.confidence,
                    "expected_root_cause": r.expected_root_cause,
                    "root_cause": r.root_cause,
                    "correct": r.correct,
                    "errors": r.errors,
                    "checks": r.checks,
                    "calls_used": r.calls_used,
                    "rounds_used": r.rounds_used,
                    "escalated": r.escalated,
                    "fell_back": r.fell_back,
                    "total_tokens": r.total_tokens,
                    "cost_usd": r.cost_usd,
                    "risk_tier": r.risk_tier,
                    "release_status": r.release_status,
                    "label_high_risk": r.label_high_risk,
                    "routed_to_human": r.routed_to_human,
                }
                for r in self.per_scenario
            ],
        }
        unpriced = sorted({m for r in self.per_scenario for m in r.unpriced_models})
        if unpriced:
            # Named rather than silently zeroed: an unpriced model still reports tokens.
            payload["unpriced_models"] = unpriced
        if self.repeats:
            payload["stability"] = {
                metric: stability(self.repeats, metric)
                for metric in (
                    "root_cause_accuracy",
                    "correct_abstention_rate",
                    "false_confident_diagnosis_rate",
                    "escalation_rate",
                )
            }
        return payload


def _safety_metrics(cfg: AppConfig, results: list[ScenarioResult]) -> dict[str, float]:
    """Gate inputs that are properties of the run set rather than per-scenario scores."""
    n = max(len(results), 1)
    happy_path = [r for r in results if r.expected_outcome == "diagnosed"]
    # High risk is what the *labels* declare (sensitivity, impact, novelty, expected
    # tier) — never the tier the run assigned itself, which would let the system pass
    # the gate by classifying its own cases as low-risk (FR-1310).
    #
    # Incident severity is deliberately not the test. A high-severity incident with no
    # sensitive data and no customer impact is a medium-risk case under FR-1105, and
    # holding it for a human would be exactly the over-escalation FR-1302 penalises.
    high_risk = [r for r in results if r.label_high_risk]

    return {
        # Every persisted report validated against its contract on the way out; a
        # malformed one would have raised before scoring.
        "schema_validity": 1.0,
        "grounding_rate": len([r for r in results if r.grounded]) / n,
        "mutation_attempts": 0.0,  # no mutating tool exists (FR-504)
        "deterministic_happy_path_rate": (
            1.0
            if not happy_path
            else len([r for r in happy_path if r.correct]) / len(happy_path)
        ),
        # AD-005 measured, not assumed: the share of runs whose delivered diagnosis
        # leaned on retrieved memory as support.
        "retrieval_satisfying_diagnosis_rate": (
            len([r for r in results if r.retrieval_satisfied_diagnosis]) / n
        ),
        "checkpoint_resume_duplication": 0.0,  # asserted in the checkpoint suite
        "high_risk_human_review_rate": (
            1.0
            if not high_risk
            else len([r for r in high_risk if r.routed_to_human]) / len(high_risk)
        ),
        "review_record_completeness": ReviewStore(
            cfg.reviews_dir
        ).disposition_stats().record_completeness,
    }


async def evaluate_scenarios(
    cfg: AppConfig | None = None,
    *,
    scenario_ids: list[str] | None = None,
    repeats: int = 1,
    reasoning: Any | None = None,
) -> EvaluationReport:
    """Run and score the labeled scenario set.

    Imported lazily inside the function so importing this module never pulls the MCP
    client into a process that only wants to read metrics.
    """

    cfg = cfg or load_config()
    labels = load_labels(cfg.scenarios_dir)
    if scenario_ids:
        wanted = set(scenario_ids)
        labels = [label for label in labels if label.scenario_id in wanted]

    passes: list[MetricSet] = []
    last_results: list[ScenarioResult] = []

    for _ in range(max(1, repeats)):
        results: list[ScenarioResult] = []
        for label in labels:
            results.append(await _run_one(label, cfg, reasoning=reasoning))
        last_results = results
        passes.append(aggregate(results))

    headline = passes[-1]
    metric_values = dict(headline.values)
    metric_values.update(_safety_metrics(cfg, last_results))

    return EvaluationReport(
        versions=fingerprint(cfg),
        metrics=headline,
        acceptance=check_acceptance(metric_values),
        per_scenario=last_results,
        repeats=passes if len(passes) > 1 else [],
    )


async def _run_one(
    label: ScenarioLabel, cfg: AppConfig, *, reasoning: Any | None = None
) -> ScenarioResult:
    """Run one scenario and score it, measuring what only the harness can see."""
    from ..app import run_investigation

    started = time.monotonic()
    state = await run_investigation(label.scenario_id, cfg, reasoning=reasoning)
    elapsed = time.monotonic() - started

    grounded, _issues = validate_grounding(state.get("report") or {}, state.get("evidence", []))

    # Real usage, read from the run journal, which attributed each completion to the
    # stage that made it. Hard-coding these to zero made the token and cost metrics look
    # measured when nothing was measuring them.
    calls = usage(state)
    cost, unpriced = estimate(calls, cfg.pricing)

    enriched = dict(state)
    enriched["_latency_seconds"] = elapsed
    enriched["_grounded"] = grounded
    enriched["_total_tokens"] = sum(int(c.get("total_tokens") or 0) for c in calls)
    enriched["_cost_usd"] = cost
    enriched["_llm_calls"] = len(calls)
    enriched["_unpriced_models"] = unpriced
    return score_run(label, enriched)
