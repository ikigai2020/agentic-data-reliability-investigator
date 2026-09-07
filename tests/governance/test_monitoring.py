"""Monitoring thresholds and response (FR-1205, NFR-009)."""

from __future__ import annotations

from investigator.config import load_config
from investigator.observability.monitoring import (
    DEFAULT_THRESHOLDS,
    Threshold,
    evaluate,
    load_thresholds,
)

ALL = list(DEFAULT_THRESHOLDS)


def test_every_threshold_names_an_owner_and_a_response() -> None:
    """FR-1205: a number nobody owns is not monitoring."""
    for threshold in DEFAULT_THRESHOLDS:
        assert threshold.owner, threshold.metric
        assert threshold.response, threshold.metric
        assert threshold.warning is not None and threshold.critical is not None


def test_a_perfect_run_raises_nothing() -> None:
    """A zero-failure metric must not warn about its own perfect score."""
    verdict = evaluate(
        {
            "grounding_failure_rate": 0.0,
            "unauthorized_tool_attempts": 0,
            "prompt_injection_successes": 0,
            "critical_contradictions_ignored": 0,
            "false_confident_diagnosis_rate": 0.0,
            "review_queue_drain_rate": 1.0,
            "tool_failure_rate": 0.0,
        },
        ALL,
    )
    assert verdict.alerts == []
    assert verdict.release_permitted is True


def test_a_must_never_happen_signal_goes_straight_to_critical() -> None:
    """One unauthorized tool call is not a rate to tolerate."""
    verdict = evaluate({"unauthorized_tool_attempts": 1}, ALL)
    assert [a.level for a in verdict.alerts] == ["critical"]


def test_a_critical_policy_breach_fails_closed() -> None:
    verdict = evaluate({"grounding_failure_rate": 0.05}, ALL)
    assert verdict.release_permitted is False
    assert verdict.halt_reasons
    assert "stop releasing diagnoses" in verdict.halt_reasons[0]


def test_a_critical_quality_breach_alerts_but_does_not_halt() -> None:
    """A latency regression is worth paging someone; it is not worth blocking diagnoses."""
    verdict = evaluate({"p95_latency_seconds": 600.0}, ALL)
    assert verdict.critical
    assert verdict.release_permitted is True


def test_a_lower_bound_metric_breaches_when_it_falls() -> None:
    assert evaluate({"review_queue_drain_rate": 1.0}, ALL).alerts == []
    assert evaluate({"review_queue_drain_rate": 0.85}, ALL).warnings
    assert evaluate({"review_queue_drain_rate": 0.5}, ALL).critical


def test_an_unmeasured_metric_is_absent_not_passing() -> None:
    """Silence must not read as success — a missing signal raises nothing either way."""
    verdict = evaluate({}, ALL)
    assert verdict.alerts == []
    assert verdict.release_permitted is True


def test_config_overrides_merge_over_the_defaults() -> None:
    thresholds = load_thresholds({"tool_failure_rate": {"warning": 0.5, "critical": 0.9}})
    by_metric = {t.metric: t for t in thresholds}

    assert by_metric["tool_failure_rate"].warning == 0.5
    assert by_metric["tool_failure_rate"].owner  # untouched fields survive the merge
    assert len(thresholds) == len(DEFAULT_THRESHOLDS)


def test_a_malformed_override_does_not_break_monitoring() -> None:
    thresholds = load_thresholds({"tool_failure_rate": {"nonsense_key": 1}})
    assert any(t.metric == "tool_failure_rate" for t in thresholds)


def test_a_new_metric_can_be_declared_in_config() -> None:
    thresholds = load_thresholds(
        {
            "custom_metric": {
                "warning": 1,
                "critical": 2,
                "owner": "me",
                "response": "look at it",
            }
        }
    )
    assert evaluate({"custom_metric": 2}, thresholds).critical


def test_alerts_carry_the_fields_a_trace_needs() -> None:
    alert = evaluate({"unauthorized_tool_attempts": 3}, ALL).alerts[0]
    fields = alert.as_log_fields()
    assert set(fields) >= {"metric", "level", "value", "threshold", "owner", "response"}


def test_monitoring_is_configured_on_by_default() -> None:
    assert load_config().monitoring.enabled is True


def test_a_threshold_can_be_relaxed_but_the_policy_kind_is_explicit() -> None:
    relaxed = Threshold(
        metric="grounding_failure_rate",
        warning=0.5,
        critical=0.9,
        owner="me",
        response="ignore",
        kind="quality",
    )
    assert evaluate({"grounding_failure_rate": 0.95}, [relaxed]).release_permitted is True
