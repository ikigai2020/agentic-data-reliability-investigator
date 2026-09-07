"""MCP client-side error taxonomy helpers (FR-508).

The client is responsible for the statuses that only it can observe:
``tool_not_found``, ``permission_denied``, ``server_unavailable``, and transport
``timeout``. Server-originated statuses (``no_data``, ``invalid_input``,
``provider_unavailable``, ``malformed_response``, ``ok``) are passed through
unchanged from the ``ToolResult`` envelope.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..schemas.results import ToolResult, ToolStatus


def client_error_result(
    *,
    tool_name: str,
    server: str,
    status: ToolStatus,
    error_message: str,
    scenario_id: str = "-",
) -> ToolResult:
    """Build a ToolResult representing a client-side failure (no server round-trip)."""
    now = datetime.now(UTC)
    return ToolResult(
        request_id="client-generated",
        tool_name=tool_name,
        source_system=server,
        scenario_id=scenario_id,
        observed_at=None,
        collected_at=now,
        status=status,
        data={},
        provenance={"origin": "mcp_client", "server": server},
        warnings=[],
        error_message=error_message,
    )
