"""Evaluation harness, metrics, and acceptance gates (§18)."""

from .acceptance import GATES, AcceptanceReport, assert_passed, check
from .harness import EvaluationReport, evaluate_scenarios
from .labels import ScenarioLabel, load_label, load_labels
from .metrics import MetricSet, ScenarioResult, aggregate, score_run, stability
from .versions import VersionFingerprint, describe, diff, fingerprint

__all__ = [
    "GATES",
    "AcceptanceReport",
    "EvaluationReport",
    "MetricSet",
    "ScenarioLabel",
    "ScenarioResult",
    "VersionFingerprint",
    "aggregate",
    "assert_passed",
    "check",
    "describe",
    "diff",
    "evaluate_scenarios",
    "fingerprint",
    "load_label",
    "load_labels",
    "score_run",
    "stability",
]
