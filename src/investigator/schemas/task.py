"""Investigation task contract (FR-302).

``allowed_tool_names`` is assigned by the Commander but *enforced* deterministically
at dispatch (FR-506); the field is a contract, not the enforcement mechanism.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import TaskAgent, TaskStatus


class InvestigationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    round_number: int
    assigned_agent: TaskAgent
    hypothesis_ids: list[str] = Field(default_factory=list)
    question: str
    expected_discriminating_value: str
    allowed_tool_names: list[str] = Field(default_factory=list)
    priority: int = 0
    status: TaskStatus = "pending"
