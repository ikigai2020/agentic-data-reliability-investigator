"""Evidence contract (FR-303).

Evidence IDs are immutable and deduplicated (FR-303, FR-307 reducers).
``source_kind`` separates current operational observations from retrieved memory
(AD-005): retrieved memory shall never carry ``current_operational``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .enums import EvidenceSourceKind, FreshnessStatus


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    investigation_id: str
    task_id: str
    producing_agent: str
    source_kind: EvidenceSourceKind
    source_name: str
    tool_name: str | None = None
    observed_at: datetime | None = None
    collected_at: datetime
    payload: dict = Field(default_factory=dict)
    summary: str
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    freshness_status: FreshnessStatus = "unknown"
    provenance: dict = Field(default_factory=dict)
