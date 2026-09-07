"""Monitoring thresholds and response (FR-1205, NFR-009).

Every threshold carries three things the spec insists on and that dashboards usually
omit: a **warning** level, a **critical** level, and — for each — an **owner** and a
**response**. A number nobody owns is not monitoring.

Critical policy failures fail closed. :func:`evaluate` returns not just alerts but a
release verdict: when a critical *policy* threshold breaches, the correct response is to
stop releasing diagnoses, not to page someone and carry on. That distinction is why
thresholds are typed ``policy`` or ``quality`` — a latency regression is worth an alert;
a grounding failure is worth halting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["ok", "warning", "critical"]
ThresholdKind = Literal["policy", "quality"]
Direction = Literal["upper", "lower"]


@dataclass(frozen=True)
class Threshold:
    """One monitored signal with its levels, owner, and response (FR-1205)."""

    metric: str
    warning: float
    critical: float
    owner: str
    response: str
    kind: ThresholdKind = "quality"
    # ``upper`` breaches when the value rises above the level (failure rates);
    # ``lower`` breaches when it falls below (success rates).
    direction: Direction = "upper"
    description: str = ""

    def level_for(self, value: float) -> Severity:
        """Critical is inclusive of its level; warning is exclusive.

        That asymmetry is deliberate. A "must never happen" signal sets both levels to the
        same number, so the first occurrence goes straight to critical rather than sitting
        at warning forever. And a zero-failure metric with ``warning=0`` reads *ok* at zero
        instead of warning about its own perfect score.
        """
        if self.direction == "upper":
            if value >= self.critical:
                return "critical"
            return "warning" if value > self.warning else "ok"
        if value <= self.critical:
            return "critical"
        return "warning" if value < self.warning else "ok"


@dataclass(frozen=True)
class Alert:
    metric: str
    level: Severity
    value: float
    breached: float
    owner: str
    response: str
    kind: ThresholdKind

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "level": self.level,
            "value": self.value,
            "threshold": self.breached,
            "owner": self.owner,
            "response": self.response,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class MonitoringVerdict:
    alerts: list[Alert] = field(default_factory=list)
    # False when a critical *policy* threshold breached: diagnoses stop being released
    # until a human clears it (FR-1205 "critical policy failures shall fail closed").
    release_permitted: bool = True
    halt_reasons: list[str] = field(default_factory=list)

    @property
    def critical(self) -> list[Alert]:
        return [a for a in self.alerts if a.level == "critical"]

    @property
    def warnings(self) -> list[Alert]:
        return [a for a in self.alerts if a.level == "warning"]


# Defaults mirror the FR-1305 acceptance thresholds: the things that must never happen
# are critical at the first occurrence, because "a few unauthorized tool calls" is not a
# rate to tolerate.
DEFAULT_THRESHOLDS: tuple[Threshold, ...] = (
    Threshold(
        metric="grounding_failure_rate",
        warning=0.0,
        critical=0.01,
        owner="data-platform-oncall",
        response="stop releasing diagnoses; investigate the grounding validator",
        kind="policy",
        description="FR-1305 requires 100% grounding on delivered reports",
    ),
    Threshold(
        metric="unauthorized_tool_attempts",
        warning=1,
        critical=1,
        owner="security-oncall",
        response="halt releases and audit the permission layer",
        kind="policy",
        description="FR-1305 requires 0 unauthorized tool calls",
    ),
    Threshold(
        metric="prompt_injection_successes",
        warning=1,
        critical=1,
        owner="security-oncall",
        response="halt releases; treat as a live compromise until proven otherwise",
        kind="policy",
        description="FR-1305 requires 0 injection successes",
    ),
    Threshold(
        metric="critical_contradictions_ignored",
        warning=1,
        critical=1,
        owner="data-platform-oncall",
        response="halt releases; review the stopping evaluator",
        kind="policy",
        description="FR-1305 requires 0 ignored critical contradictions",
    ),
    Threshold(
        metric="false_confident_diagnosis_rate",
        warning=0.0,
        critical=0.02,
        owner="data-platform-lead",
        response="disable automatic release; require human review for all diagnoses",
        kind="policy",
        description="the failure the whole system is built to avoid",
    ),
    Threshold(
        metric="review_queue_drain_rate",
        warning=0.9,
        critical=0.75,
        owner="data-platform-lead",
        response="escalations are piling up unanswered; staff the review queue",
        kind="quality",
        direction="lower",
        description="unanswered escalations are a workload signal, not a policy breach",
    ),
    Threshold(
        metric="tool_failure_rate",
        warning=0.1,
        critical=0.3,
        owner="data-platform-oncall",
        response="investigate MCP providers; expect degraded abstention rates",
    ),
    Threshold(
        metric="escalation_rate",
        warning=0.4,
        critical=0.7,
        owner="data-platform-lead",
        response="investigate whether evidence quality or thresholds regressed",
    ),
    Threshold(
        metric="abstention_rate",
        warning=0.6,
        critical=0.85,
        owner="data-platform-lead",
        response="investigate evidence sufficiency before relaxing any threshold",
    ),
    Threshold(
        metric="p95_latency_seconds",
        warning=120.0,
        critical=300.0,
        owner="data-platform-oncall",
        response="investigate provider latency and per-round budgets",
    ),
    Threshold(
        metric="cost_per_investigation_usd",
        warning=0.5,
        critical=2.0,
        owner="data-platform-lead",
        response="review model choice and call budgets",
    ),
)


def load_thresholds(raw: dict[str, Any] | None) -> list[Threshold]:
    """Merge configured overrides over the defaults, keyed by metric name."""
    merged = {t.metric: t for t in DEFAULT_THRESHOLDS}
    for metric, override in (raw or {}).items():
        base = merged.get(metric)
        fields: dict[str, Any] = {"metric": metric}
        if base is not None:
            fields.update(
                warning=base.warning,
                critical=base.critical,
                owner=base.owner,
                response=base.response,
                kind=base.kind,
                direction=base.direction,
                description=base.description,
            )
        fields.update({k: v for k, v in (override or {}).items() if k != "metric"})
        try:
            merged[metric] = Threshold(**fields)
        except TypeError:
            continue  # an unknown key in config must not break monitoring
    return sorted(merged.values(), key=lambda t: t.metric)


def evaluate(
    metrics: dict[str, float], thresholds: list[Threshold] | None = None
) -> MonitoringVerdict:
    """Compare metrics against thresholds and decide whether releases may continue."""
    thresholds = thresholds if thresholds is not None else list(DEFAULT_THRESHOLDS)
    alerts: list[Alert] = []
    halt_reasons: list[str] = []

    for threshold in thresholds:
        if threshold.metric not in metrics:
            continue  # an unmeasured signal is not a passing one; it is simply absent
        value = float(metrics[threshold.metric])
        level = threshold.level_for(value)
        if level == "ok":
            continue
        alerts.append(
            Alert(
                metric=threshold.metric,
                level=level,
                value=value,
                breached=threshold.critical if level == "critical" else threshold.warning,
                owner=threshold.owner,
                response=threshold.response,
                kind=threshold.kind,
            )
        )
        if level == "critical" and threshold.kind == "policy":
            halt_reasons.append(
                f"{threshold.metric}={value:g} breached critical "
                f"{threshold.critical:g} — {threshold.response}"
            )

    return MonitoringVerdict(
        alerts=alerts,
        release_permitted=not halt_reasons,
        halt_reasons=halt_reasons,
    )
