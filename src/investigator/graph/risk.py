"""Deterministic risk classification and release gating (v2.1: AD-007, FR-1105).

Autonomy is governed by deterministic risk policy, never by model confidence alone
(AD-007). Before a diagnosis/report is released, the investigation is classified as
``low | medium | high | prohibited`` from incident severity, data sensitivity, impact,
action reversibility (ADRI is read-only, hence reversible), evidence sufficiency,
unresolved contradictions, scenario novelty, and tool integrity/availability.

Release policy (FR-1105):
  * ``low``       -> release the read-only result once stopping+grounding pass.
  * ``medium``    -> require an additional independent validation before release. For a
                    ``diagnosed`` outcome this is already guaranteed by FR-900 (>=2
                    independent supports + refuted competitor), so it releases; an
                    abstention releases as an abstention.
  * ``high``      -> prepare a recommendation + escalation package for human review even
                    when evidence is otherwise sufficient (human authority).
  * ``prohibited``-> block the output and record the policy reason (fail closed).

This module is pure deterministic policy (AD-008); it holds no prompt logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas.enums import Outcome, ReleaseStatus, RiskTier, Severity

# Tool statuses that mean a *material* check could not be trusted/obtained. Their
# presence on a released investigation raises risk (tool integrity, FR-1105).
_INTEGRITY_FAILURE_MARKERS = (
    "malformed_response",
    "provider_unavailable",
    "server_unavailable",
    "timeout",
)

_HIGH_SEVERITIES: frozenset[str] = frozenset({"high", "critical"})


@dataclass
class RiskAssessment:
    tier: RiskTier
    release_status: ReleaseStatus
    reasons: list[str] = field(default_factory=list)


def _truthy(inputs: dict[str, Any], *keys: str) -> bool:
    return any(bool(inputs.get(k)) for k in keys)


def classify(
    *,
    severity: Severity,
    outcome: Outcome | None,
    diagnosis_checks: dict[str, bool],
    failures: list[str],
    policy_inputs: dict[str, Any] | None,
) -> RiskAssessment:
    """Classify investigation risk and decide the release status deterministically."""
    inputs = policy_inputs or {}
    reasons: list[str] = []

    # --- prohibited: explicit policy block, or any mutating/remediation request.
    # ADRI is read-only in M1, so remediation never reaches here; the flag exists so
    # the branch is exercised and fails closed.
    if _truthy(inputs, "blocked", "prohibited", "remediation_requested"):
        reasons.append("policy blocks release for this investigation (prohibited)")
        return RiskAssessment(tier="prohibited", release_status="blocked", reasons=reasons)

    # --- high: sensitivity / impact / novelty / unresolved contradiction / tool integrity.
    high = False
    if _truthy(inputs, "high_impact", "financial_impact", "privacy_impact", "compliance_impact"):
        high = True
        reasons.append("high customer/financial/privacy/compliance impact flagged")
    if str(inputs.get("data_sensitivity", "")).lower() in {"high", "sensitive", "restricted"}:
        high = True
        reasons.append("high data sensitivity")
    if _truthy(inputs, "novel", "out_of_taxonomy"):
        high = True
        reasons.append("scenario novelty / potential out-of-taxonomy root cause")
    if diagnosis_checks and diagnosis_checks.get("no_critical_current_contradiction") is False:
        high = True
        reasons.append("unresolved critical current contradiction at release")
    if _truthy(inputs, "unresolved_critic_disagreement"):
        # FR-1107: the Critic still disagrees after an extra discriminating check was
        # spent on the conflict. Human authority decides rather than the controller
        # forcing a diagnosis past a standing objection.
        high = True
        reasons.append("Evidence Critic disagreement unresolved after an additional check")
    integrity_failures = [
        f for f in failures if any(marker in f for marker in _INTEGRITY_FAILURE_MARKERS)
    ]
    if integrity_failures:
        high = True
        reasons.append(
            f"tool integrity/availability compromised ({len(integrity_failures)} failure(s))"
        )

    if high:
        return RiskAssessment(
            tier="high", release_status="human_review_required", reasons=reasons
        )

    # --- medium: elevated severity or minor (non-integrity) tool failures. Read-only,
    # non-sensitive investigations stay at most medium (per M1 policy).
    medium = False
    if severity in _HIGH_SEVERITIES:
        medium = True
        reasons.append(f"elevated incident severity ({severity})")
    if failures and not integrity_failures:
        medium = True
        reasons.append("minor tool failures recorded")

    if medium:
        # Additional independent validation is required before release. A `diagnosed`
        # outcome already satisfies FR-900's independent-support + competitor-refutation
        # requirement, so it releases; anything else releases as its (non-diagnostic) form.
        reasons.append(
            "medium risk: independent validation required (FR-900 satisfied)"
            if outcome == "diagnosed"
            else "medium risk: releasing non-diagnostic result"
        )
        return RiskAssessment(tier="medium", release_status="released", reasons=reasons)

    reasons.append("read-only, low-impact investigation")
    return RiskAssessment(tier="low", release_status="released", reasons=reasons)
