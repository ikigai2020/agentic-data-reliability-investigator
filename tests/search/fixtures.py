"""Shared builders for Milestone 3 search and critic tests."""

from __future__ import annotations

from datetime import UTC, datetime

from investigator.agents.reasoning import DeterministicReasoning
from investigator.schemas import Alert
from investigator.schemas.branch import InvestigationBranch
from investigator.schemas.evidence import Evidence
from investigator.schemas.finding import AgentFinding
from investigator.schemas.hypothesis import Hypothesis
from investigator.search import beam

NOW = datetime(2026, 8, 27, tzinfo=UTC)

# Hypotheses generated for a volume_drop alert, in rank order:
#   H1 transformation_logic, H2 source_data, H3 orchestration, H4 data_quality
H_TRANSFORM = "H1-transformation_logic"
H_SOURCE = "H2-source_data"
H_ORCH = "H3-orchestration"
H_QUALITY = "H4-data_quality"


def orders_alert() -> Alert:
    return Alert(
        incident_id="INC-3001",
        dataset="orders_fact",
        pipeline="orders_daily",
        symptom_type="volume_drop",
        observed_value=8000,
        expected_value=20000,
        window_start=datetime(2026, 8, 15, tzinfo=UTC),
        window_end=datetime(2026, 8, 16, tzinfo=UTC),
        detected_at=datetime(2026, 8, 16, 2, tzinfo=UTC),
        severity="high",
    )


def hypotheses() -> list[Hypothesis]:
    return DeterministicReasoning().generate_hypotheses_sync(orders_alert(), max_count=5)


def branches() -> list[InvestigationBranch]:
    return beam.initialize_branches(hypotheses())


def evidence(
    *,
    tool: str,
    hypothesis_id: str,
    supports: bool = True,
    discriminating: bool = True,
    round_number: int = 1,
    evidence_id: str | None = None,
    source_kind: str = "current_operational",
    extra_supports: list[str] | None = None,
) -> Evidence:
    """One current operational observation for/against a hypothesis."""
    task_id = f"T{round_number}-{hypothesis_id}"
    supported = ([hypothesis_id] if supports else []) + (extra_supports or [])
    return Evidence(
        evidence_id=evidence_id or f"ev_{task_id}_{tool}",
        investigation_id="inv_test",
        task_id=task_id,
        producing_agent="data_investigator",
        source_kind=source_kind,  # type: ignore[arg-type]
        source_name="data_observability",
        tool_name=tool,
        observed_at=NOW,
        collected_at=NOW,
        summary=f"{tool} observation",
        supports=supported,
        contradicts=[] if supports else [hypothesis_id],
        freshness_status="current" if source_kind == "current_operational" else "historical",
        provenance={"request_id": f"req_{tool}", "discriminating": discriminating},
    )


def finding(
    *,
    hypothesis_id: str,
    evidence_ids: list[str],
    supports: list[str] | None = None,
    status: str = "complete",
    round_number: int = 1,
) -> AgentFinding:
    return AgentFinding(
        task_id=f"T{round_number}-{hypothesis_id}",
        agent_name="data_investigator",
        conclusion="observed",
        evidence_ids=evidence_ids,
        supports=supports if supports is not None else [hypothesis_id],
        contradicts=[],
        unresolved_questions=[],
        recommended_next_actions=[],
        completion_status=status,  # type: ignore[arg-type]
    )
