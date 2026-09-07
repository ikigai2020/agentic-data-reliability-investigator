"""Unit tests for deterministic risk classification (v2.1: AD-007, FR-1105)."""

from __future__ import annotations

from investigator.graph import risk

_PASSING_CHECKS = {
    "verified": True,
    "two_independent_current_supports": True,
    "at_least_one_discriminating": True,
    "strongest_competitor_weakened": True,
    "no_critical_current_contradiction": True,
    "leading_has_evidence": True,
}


def test_read_only_low_impact_is_low_and_released() -> None:
    a = risk.classify(
        severity="low",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"data_sensitivity": "low", "high_impact": False},
    )
    assert a.tier == "low"
    assert a.release_status == "released"


def test_high_severity_read_only_is_medium_and_released() -> None:
    # Per M1 policy: high severity alone (read-only, non-sensitive) is medium, still released.
    a = risk.classify(
        severity="high",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"data_sensitivity": "low"},
    )
    assert a.tier == "medium"
    assert a.release_status == "released"


def test_high_impact_flag_forces_human_review() -> None:
    a = risk.classify(
        severity="high",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"high_impact": True},
    )
    assert a.tier == "high"
    assert a.release_status == "human_review_required"


def test_sensitive_data_forces_human_review() -> None:
    a = risk.classify(
        severity="low",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"data_sensitivity": "high"},
    )
    assert a.tier == "high"
    assert a.release_status == "human_review_required"


def test_tool_integrity_failure_raises_risk() -> None:
    a = risk.classify(
        severity="medium",
        outcome="inconclusive",
        diagnosis_checks=_PASSING_CHECKS,
        failures=["compare_source_and_target: provider_unavailable (backend down)"],
        policy_inputs={},
    )
    assert a.tier == "high"
    assert a.release_status == "human_review_required"


def test_unresolved_contradiction_raises_risk() -> None:
    checks = {**_PASSING_CHECKS, "no_critical_current_contradiction": False}
    a = risk.classify(
        severity="medium",
        outcome="inconclusive",
        diagnosis_checks=checks,
        failures=[],
        policy_inputs={},
    )
    assert a.tier == "high"


def test_prohibited_blocks_and_fails_closed() -> None:
    a = risk.classify(
        severity="critical",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"blocked": True},
    )
    assert a.tier == "prohibited"
    assert a.release_status == "blocked"


def test_confidence_is_never_the_sole_signal() -> None:
    # AD-007: risk policy ignores any 'model_confidence' input entirely in M1.
    high_conf = risk.classify(
        severity="low",
        outcome="diagnosed",
        diagnosis_checks=_PASSING_CHECKS,
        failures=[],
        policy_inputs={"model_confidence": 0.99, "data_sensitivity": "low"},
    )
    assert high_conf.tier == "low"  # confidence did not lower or raise the tier
