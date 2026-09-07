"""Scenario labels — the evaluation ground truth (FR-1300, FR-1310).

A label is authored **with the scenario fixture and never derived from a run**. That
separation is what makes FR-1310 evaluator independence structural rather than a promise:
the metrics module reads labels and run outputs and compares them, and has no path by
which the system under test can influence what "correct" means.

Every field FR-1300 requires is here: expected verification, root cause or abstention,
required and forbidden evidence, specialist participation, retrieval behaviour, maximum
calls, and escalation expectation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioLabel:
    """Declared expectations for one labeled scenario."""

    scenario_id: str
    title: str = ""
    expected_verification: str | None = None
    expected_outcome: str | None = None
    expected_root_cause_category: str | None = None
    required_evidence_tools: tuple[str, ...] = ()
    forbidden_evidence_tools: tuple[str, ...] = ()
    forbidden_outcomes: tuple[str, ...] = ()
    specialist_participation: tuple[str, ...] = ()
    retrieval_behavior: str = "not_required"
    max_operational_calls: int = 8
    escalation_expected: bool = False
    expected_risk_tier: str | None = None
    expected_release_status: str | None = None
    # FR-1311: severity-weighted reporting. A miss on a critical incident is not one miss.
    severity: str = "medium"
    # FR-1307: scenarios that deliberately degrade inputs (noise, outages, injection).
    robustness_class: str | None = None
    # FR-1301 #9: what the injected text *demands*. Adopting it is the definition of a
    # successful injection, so the attacker's goal is written down alongside the fixture
    # rather than inferred from the run.
    injection_target_root_cause: str | None = None
    injection_target_outcome: str | None = None
    # FR-1301 index, so coverage of the required scenario list is checkable.
    fr1301_index: int | None = None
    risk: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def expects_abstention(self) -> bool:
        return self.expected_outcome == "inconclusive"

    @property
    def expects_diagnosis(self) -> bool:
        return self.expected_outcome == "diagnosed"

    @property
    def is_high_risk(self) -> bool:
        """Whether the *label* calls this a high-risk case (FR-1305 routing gate).

        Read from the declared policy inputs and expected tier, never from what the run
        decided — a system that classifies its own cases as low-risk must not be able to
        pass the gate that way (FR-1310).

        Incident severity deliberately does not appear here. Severity drives FR-1311
        error weighting; *risk* is sensitivity, impact, and novelty, which is what
        FR-1105 routes on.
        """
        if self.expected_risk_tier in {"high", "prohibited"}:
            return True
        if str(self.risk.get("data_sensitivity", "")).lower() in {
            "high",
            "sensitive",
            "restricted",
        }:
            return True
        return any(
            bool(self.risk.get(flag))
            for flag in (
                "high_impact",
                "financial_impact",
                "privacy_impact",
                "compliance_impact",
                "novel",
                "out_of_taxonomy",
            )
        )


# Severity weights for FR-1311. Deliberately superlinear: getting a critical incident
# wrong is worse than getting four low-severity ones wrong.
SEVERITY_WEIGHTS: dict[str, float] = {
    "low": 1.0,
    "medium": 2.0,
    "high": 4.0,
    "critical": 8.0,
}


def weight_for(severity: str) -> float:
    return SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["medium"])


def _tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list | tuple):
        return tuple(str(x) for x in raw)
    return ()


def load_label(path: str | Path) -> ScenarioLabel:
    """Load one scenario's ``meta.json`` into a label."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ScenarioLabel(
        scenario_id=raw["scenario_id"],
        title=raw.get("title", ""),
        expected_verification=raw.get("expected_verification"),
        expected_outcome=raw.get("expected_outcome"),
        expected_root_cause_category=raw.get("expected_root_cause_category"),
        required_evidence_tools=_tuple(raw.get("required_evidence_tools")),
        forbidden_evidence_tools=_tuple(raw.get("forbidden_evidence_tools")),
        forbidden_outcomes=_tuple(raw.get("forbidden_outcomes")),
        specialist_participation=_tuple(raw.get("specialist_participation")),
        retrieval_behavior=raw.get("retrieval_behavior", "not_required"),
        max_operational_calls=int(raw.get("max_operational_calls", 8)),
        escalation_expected=bool(raw.get("escalation_expected", False)),
        expected_risk_tier=raw.get("expected_risk_tier"),
        expected_release_status=raw.get("expected_release_status"),
        severity=raw.get("severity", "medium"),
        robustness_class=raw.get("robustness_class"),
        injection_target_root_cause=raw.get("injection_target_root_cause"),
        injection_target_outcome=raw.get("injection_target_outcome"),
        fr1301_index=raw.get("fr1301_index"),
        risk=dict(raw.get("risk", {})),
        notes=raw.get("notes", ""),
    )


def load_labels(scenarios_dir: str | Path) -> list[ScenarioLabel]:
    """Load every labeled scenario under ``scenarios_dir``, in stable order."""
    root = Path(scenarios_dir)
    if not root.exists():
        return []
    labels: list[ScenarioLabel] = []
    for meta in sorted(root.glob("*/meta.json")):
        try:
            labels.append(load_label(meta))
        except (KeyError, json.JSONDecodeError):
            continue  # an unlabeled scenario is not an evaluation input
    return labels
