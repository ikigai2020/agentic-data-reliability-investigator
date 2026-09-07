"""Role-based MCP tool permissions (FR-506, FR-1100).

Permission enforcement lives in dispatch code, not in prompts. This module is the
single source of truth for which role may call which tool on which server.

  * Pipeline Investigator -> pipeline server only
  * Data Investigator     -> observability server only
  * Commander / Critic    -> no operational tools
"""

from __future__ import annotations

from ..schemas.enums import AgentRole

PIPELINE_SERVER = "pipeline_operations"
OBSERVABILITY_SERVER = "data_observability"

PIPELINE_TOOLS: frozenset[str] = frozenset(
    {
        "get_pipeline_run_status",
        "get_task_failures",
        "get_execution_logs",
        "get_upstream_dependencies",
        "get_processing_watermark",
    }
)

OBSERVABILITY_TOOLS: frozenset[str] = frozenset(
    {
        "get_table_metrics",
        "get_quality_results",
        "get_schema_changes",
        "compare_source_and_target",
        "get_recent_transformation_changes",
    }
)

# role -> {server: allowed tool names}
ROLE_PERMISSIONS: dict[AgentRole, dict[str, frozenset[str]]] = {
    "pipeline_investigator": {PIPELINE_SERVER: PIPELINE_TOOLS},
    "data_investigator": {OBSERVABILITY_SERVER: OBSERVABILITY_TOOLS},
    "commander": {},
    "critic": {},
}

# tool -> server (which server hosts a tool)
TOOL_SERVER: dict[str, str] = {
    **{t: PIPELINE_SERVER for t in PIPELINE_TOOLS},
    **{t: OBSERVABILITY_SERVER for t in OBSERVABILITY_TOOLS},
}


def server_for_tool(tool_name: str) -> str | None:
    return TOOL_SERVER.get(tool_name)


def is_allowed(role: AgentRole, tool_name: str) -> bool:
    """True only when ``role`` may call ``tool_name`` on the server that hosts it."""
    server = TOOL_SERVER.get(tool_name)
    if server is None:
        return False
    return tool_name in ROLE_PERMISSIONS.get(role, {}).get(server, frozenset())


def allowed_tools(role: AgentRole) -> frozenset[str]:
    result: set[str] = set()
    for tools in ROLE_PERMISSIONS.get(role, {}).values():
        result |= tools
    return frozenset(result)
