"""MCP typed result envelope and error taxonomy (FR-503, FR-508).

Every MCP tool returns a ``ToolResult`` so callers get uniform provenance,
timestamps, request IDs, status, and warnings regardless of the tool. This model
is shared by both servers and the client (imported by contract, not by fixture).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# FR-508 error taxonomy. ``ok`` and ``no_data`` are non-error statuses.
ToolStatus = Literal[
    "ok",
    "no_data",
    "tool_not_found",
    "permission_denied",
    "invalid_input",
    "timeout",
    "server_unavailable",
    "provider_unavailable",
    "malformed_response",
]

# Statuses for which "the tool could not observe" — these must NOT count as
# evidence against a hypothesis (FR-508).
NON_EVIDENCE_STATUSES: frozenset[str] = frozenset(
    {
        "tool_not_found",
        "permission_denied",
        "invalid_input",
        "timeout",
        "server_unavailable",
        "provider_unavailable",
        "malformed_response",
    }
)

# Statuses eligible for a bounded transport retry (FR-507).
TRANSIENT_STATUSES: frozenset[str] = frozenset({"timeout", "server_unavailable"})


class ToolResult(BaseModel):
    """Uniform envelope returned by every MCP tool (FR-503)."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    tool_name: str
    source_system: str
    scenario_id: str
    observed_at: datetime | None = None
    collected_at: datetime
    status: ToolStatus = "ok"
    data: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None

    @property
    def is_evidence(self) -> bool:
        """True only when the result reflects an actual observation."""
        return self.status in {"ok", "no_data"}

    @property
    def is_transient_failure(self) -> bool:
        return self.status in TRANSIENT_STATUSES
