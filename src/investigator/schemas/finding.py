"""Agent finding contract (FR-304).

A specialist summarizes local scratch observations into this structured output
before results enter global state (FR-202).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import FindingStatus


class AgentFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_name: str
    conclusion: str
    evidence_ids: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    completion_status: FindingStatus = "complete"
