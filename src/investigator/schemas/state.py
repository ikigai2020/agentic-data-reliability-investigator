"""Global investigation state (FR-307).

The parent LangGraph state is a ``TypedDict`` with reducer-annotated collections so
merges append and deduplicate by stable IDs. Nodes return partial updates only.

Budgets (FR-702) are tracked here and enforced by the deterministic controller: the
global operational-call cap, the round cap, beam width, and branch depth.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from ..graph.reducers import dedupe_by, extend_unique, upsert_by
from .alert import Alert
from .branch import InvestigationBranch
from .critic import CriticReview
from .enums import ConfidenceBand, Outcome, ReleaseStatus, RiskTier, VerificationStatus
from .evidence import Evidence
from .finding import AgentFinding
from .hypothesis import Hypothesis
from .journal import StageRecord
from .retrieval import RetrievalInfluence, RetrievedItem, TrustDecision
from .task import InvestigationTask


class Budgets(TypedDict, total=False):
    """Beam-search budgets (FR-702)."""

    max_operational_calls: int
    max_rounds: int
    calls_used: int


class GlobalInvestigationState(TypedDict, total=False):
    # --- metadata ---
    investigation_id: str
    trace_id: str
    scenario_id: str
    round_number: int

    # --- alert lifecycle ---
    raw_alert: dict[str, Any]
    alert: Alert | None
    parse_error: str | None
    verification_status: VerificationStatus | None
    verification_detail: dict[str, Any] | None

    # --- reasoning artifacts ---
    hypotheses: Annotated[list[Hypothesis], upsert_by("hypothesis_id")]
    branches: Annotated[list[InvestigationBranch], upsert_by("branch_id")]
    tasks: Annotated[list[InvestigationTask], upsert_by("task_id")]
    evidence: Annotated[list[Evidence], dedupe_by("evidence_id")]
    findings: Annotated[list[AgentFinding], dedupe_by("task_id")]
    critic_reviews: Annotated[list[CriticReview], dedupe_by("review_id")]

    # --- tree search (Milestone 3, §12) ---
    # Branch IDs the beam selected for expansion in the next round (FR-702). Replaced
    # wholesale each round rather than accumulated — it is a decision, not a history.
    beam_selected_ids: list[str]
    # This round's itemized FR-703 rubric per branch. Replaced wholesale like the beam
    # selection — it is the current scoring, not a history. Each round's copy survives in
    # the run journal, which is what the beam view reads (M5.3).
    branch_scores: dict[str, Any]
    should_continue: bool
    rounds_used: int
    # FR-1107 (recording half): where advisory critic judgment and deterministic policy
    # disagreed. Routing a disagreement to an extra discriminating check is Milestone 4.
    critic_disagreements: Annotated[list[str], extend_unique]
    # Steps that degraded from LLM reasoning to deterministic rules (AD-004). A run is
    # never quietly half-modelled: the report says which steps the model did not decide.
    reasoning_fallbacks: Annotated[list[str], extend_unique]
    # FR-1107: extra rounds spent resolving a Critic/policy conflict. Bounded to one, so
    # a persistent disagreement escalates instead of looping.
    disagreement_checks_used: int

    # --- retrieval / long-term memory (Milestone 2, §11) ---
    retrieved_items: list[RetrievedItem]
    trust_decisions: list[TrustDecision]
    retrieval_influence: RetrievalInfluence | None

    # --- run journal (M5.1) ---
    # How the investigation moved, stage by stage. Deduped by stage id so replaying a
    # checkpointed node records the stage once, not twice (FR-205).
    journal: Annotated[list[StageRecord], dedupe_by("stage_id")]

    # --- control ---
    budgets: Budgets
    failures: Annotated[list[str], extend_unique]

    # --- risk-based autonomy (v2.1: AD-007, FR-1105/FR-1106) ---
    # Deterministic policy inputs (data sensitivity/impact flags) supplied by the
    # controller, not operational evidence. Kept outside prompts (AD-008).
    risk_policy_inputs: dict[str, Any]
    risk_tier: RiskTier | None
    risk_reasons: Annotated[list[str], extend_unique]
    release_status: ReleaseStatus | None
    blocked_attempts: int

    # --- outcome ---
    outcome: Outcome | None
    stop_reason: str | None
    confidence: ConfidenceBand | None
    leading_hypothesis_id: str | None
    strongest_competitor_id: str | None
    diagnosis_checks: dict[str, bool]
    escalation_required: bool
    escalation_package: dict[str, Any] | None
    report: dict[str, Any] | None
