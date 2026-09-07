"""Enumerations and shared literal types (FR-300, FR-400).

Kept separate from the contracts so taxonomy and status vocabularies are reused
consistently across schemas, agents, servers, and deterministic policy.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class RootCauseCategory(str, Enum):
    """Approved root-cause taxonomy (FR-400).

    ``unknown`` is deliberately absent: it is an *outcome*, never a hypothesis.
    """

    SOURCE_DATA = "source_data"
    ORCHESTRATION = "orchestration"
    TRANSFORMATION_LOGIC = "transformation_logic"
    SCHEMA_CONTRACT = "schema_contract"
    DATA_QUALITY = "data_quality"
    PROCESSING_STATE = "processing_state"
    INFRASTRUCTURE = "infrastructure"
    LEGITIMATE_BUSINESS_CHANGE = "legitimate_business_change"


# --- Alert vocabulary (FR-300) ---
SymptomType = Literal[
    "volume_drop",
    "volume_spike",
    "freshness_delay",
    "null_spike",
    "uniqueness_failure",
    "referential_integrity_failure",
    "schema_change",
    "pipeline_failure",
    "duplicate_processing",
]

Severity = Literal["low", "medium", "high", "critical"]

# --- Status vocabularies ---
HypothesisStatus = Literal["active", "supported", "weakened", "rejected"]
HypothesisOrigin = Literal["initial_generation", "retrieval_influenced", "reopened"]
TaskAgent = Literal["pipeline_investigator", "data_investigator"]
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
EvidenceSourceKind = Literal["current_operational", "retrieved_incident", "runbook"]
FreshnessStatus = Literal["current", "historical", "unknown"]
FindingStatus = Literal["complete", "partial", "failed"]
BranchStatus = Literal["active", "pruned", "reopened", "selected", "closed"]
StopRecommendation = Literal["continue", "diagnosed", "inconclusive"]

# --- Verification and outcomes (FR-803, FR-900..902) ---
VerificationStatus = Literal["verified", "not_verified", "verification_unavailable"]
Outcome = Literal[
    "diagnosed",
    "inconclusive",
    "not_an_incident",
    "invalid_input",
]
ConfidenceBand = Literal["high", "moderate", "low", "not_applicable"]

# --- Agent roles (FR-506 permission keys) ---
AgentRole = Literal[
    "commander",
    "pipeline_investigator",
    "data_investigator",
    "critic",
]

# --- Risk-based autonomy (v2.1: AD-007, FR-1105/FR-1106) ---
RiskTier = Literal["low", "medium", "high", "prohibited"]

# Governance decision on whether a diagnosis/report may be released without a human.
ReleaseStatus = Literal["released", "human_review_required", "blocked"]

# --- Human review (FR-1108) ---
# `pending` is a real disposition: an escalation is incomplete until it is recorded,
# and "nobody looked yet" must be distinguishable from "nobody has been asked".
HumanDecisionKind = Literal["confirmed", "rejected", "modified", "pending"]


# --- Retrieval and long-term memory (Milestone 2, §11) ---
DocumentType = Literal["incident", "runbook", "schema"]

# Confirmation status governs whether a document may influence planning (FR-600, FR-603).
# Only "confirmed"/"approved" are trusted; others are never used as precedent (FR-1109).
ConfirmationStatus = Literal["confirmed", "approved", "unconfirmed", "rejected"]

ResolutionStatus = Literal["resolved", "unresolved", "superseded"]

# What retrieval did to the investigation (FR-604).
InfluenceKind = Literal["reordered", "suggested_check", "no_effect", "rejected", "not_run"]
