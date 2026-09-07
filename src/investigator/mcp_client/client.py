"""MCP client: discovery + permissioned dispatch (§10.1, FR-506..509).

Launches both read-only MCP servers over stdio (one subprocess each), discovers
their tools at startup, and dispatches typed calls. Scenario selection is injected
into each subprocess's environment (plan ambiguity #2) so agents never read fixtures.

The client owns the client-side error taxonomy (FR-508) and hosts the permission
layer via :meth:`proxy`. Transport is stdio for Milestone 1; the launch config is
the only thing that would change for Streamable HTTP (§10.1).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import load_config
from ..observability.logging import get_logger
from ..schemas.enums import AgentRole
from ..schemas.results import ToolResult
from .errors import client_error_result
from .permissions import OBSERVABILITY_SERVER, PIPELINE_SERVER, is_allowed, server_for_tool

_REPO_ROOT = Path(__file__).resolve().parents[3]

# server key -> module launched with `python -m <module>`
SERVER_MODULES: dict[str, str] = {
    PIPELINE_SERVER: "mcp_servers.pipeline_operations.server",
    OBSERVABILITY_SERVER: "mcp_servers.data_observability.server",
}


class InvestigatorMCPClient:
    """Async context manager holding live sessions to both MCP servers."""

    def __init__(
        self,
        scenario_id: str,
        *,
        extra_env: dict[str, str] | None = None,
        server_modules: dict[str, str] | None = None,
    ) -> None:
        self.scenario_id = scenario_id
        self._extra_env = extra_env or {}
        self._server_modules = server_modules or dict(SERVER_MODULES)
        self._cfg = load_config()
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, dict[str, Any]] = {}  # server -> {tool_name: inputSchema}
        self.unavailable_servers: dict[str, str] = {}
        # FR-1106: blocked tool attempts are measured even when no unsafe action occurs.
        self.blocked_attempts: int = 0
        self._log = get_logger()

    # --- lifecycle -------------------------------------------------------- #
    def _server_params(self, module: str) -> StdioServerParameters:
        env = {
            **os.environ,
            "INVESTIGATOR_SCENARIO": self.scenario_id,
            "PYTHONPATH": os.pathsep.join(
                [str(_REPO_ROOT / "src"), str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
            **self._extra_env,
        }
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", module],
            env=env,
            cwd=str(_REPO_ROOT),
        )

    async def __aenter__(self) -> InvestigatorMCPClient:
        for server_key, module in self._server_modules.items():
            try:
                read, write = await self._stack.enter_async_context(
                    stdio_client(self._server_params(module))
                )
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=15.0)
                self._sessions[server_key] = session
            except Exception as exc:  # noqa: BLE001 - server startup failure is recoverable
                self.unavailable_servers[server_key] = repr(exc)
                self._log.log("mcp_server_unavailable", server=server_key, error=repr(exc))
        await self.discover()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    # --- discovery (FR-509) ---------------------------------------------- #
    async def discover(self) -> dict[str, dict[str, Any]]:
        for server_key, session in self._sessions.items():
            listed = await session.list_tools()
            self._tools[server_key] = {t.name: t.inputSchema for t in listed.tools}
            self._log.log(
                "mcp_discovery",
                server=server_key,
                tools=sorted(self._tools[server_key]),
            )
        return self._tools

    @property
    def discovered_tools(self) -> dict[str, list[str]]:
        return {server: sorted(tools) for server, tools in self._tools.items()}

    def has_tool(self, server: str, tool_name: str) -> bool:
        return tool_name in self._tools.get(server, {})

    # --- dispatch (FR-503, FR-508) --------------------------------------- #
    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if server in self.unavailable_servers or server not in self._sessions:
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="server_unavailable",
                error_message=self.unavailable_servers.get(server, "server not connected"),
                scenario_id=self.scenario_id,
            )
        if not self.has_tool(server, tool_name):
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="tool_not_found",
                error_message=f"tool '{tool_name}' not exposed by '{server}'",
                scenario_id=self.scenario_id,
            )

        session = self._sessions[server]
        # Client timeout is a small margin above the server-side tool timeout (FR-507).
        client_timeout = self._cfg.mcp.tool_timeout_seconds + 5.0
        try:
            raw = await asyncio.wait_for(
                session.call_tool(tool_name, arguments), timeout=client_timeout
            )
        except TimeoutError:
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="timeout",
                error_message=f"client timeout after {client_timeout}s",
                scenario_id=self.scenario_id,
            )
        except Exception as exc:  # noqa: BLE001 - transport failure
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="server_unavailable",
                error_message=repr(exc),
                scenario_id=self.scenario_id,
            )
        return self._parse_result(server, tool_name, raw)

    def _parse_result(self, server: str, tool_name: str, raw: Any) -> ToolResult:
        payload = getattr(raw, "structuredContent", None)
        if isinstance(payload, dict) and "request_id" not in payload and "result" in payload:
            payload = payload["result"]
        if not isinstance(payload, dict) or "request_id" not in payload:
            # Fall back to text content.
            payload = None
            for item in getattr(raw, "content", []) or []:
                text = getattr(item, "text", None)
                if text:
                    try:
                        payload = json.loads(text)
                        break
                    except json.JSONDecodeError:
                        continue
        if not isinstance(payload, dict) or "request_id" not in payload:
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="malformed_response",
                error_message="no ToolResult envelope in response",
                scenario_id=self.scenario_id,
            )
        try:
            return ToolResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            return client_error_result(
                tool_name=tool_name,
                server=server,
                status="malformed_response",
                error_message=repr(exc),
                scenario_id=self.scenario_id,
            )

    def proxy(self, role: AgentRole) -> PermissionedToolProxy:
        return PermissionedToolProxy(self, role)


class PermissionedToolProxy:
    """Role-scoped façade over the client that enforces FR-506 before dispatch.

    Unauthorized calls are refused *here* and never reach a server, so the count of
    unauthorized tool calls observed by servers is structurally zero (§18).
    """

    def __init__(self, client: InvestigatorMCPClient, role: AgentRole) -> None:
        self._client = client
        self.role = role
        self._log = get_logger()

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        server = server_for_tool(tool_name)
        if server is None or not is_allowed(self.role, tool_name):
            # FR-1106: runtime enforcement — refuse and measure the blocked attempt.
            self._client.blocked_attempts += 1
            self._log.log(
                "permission_denied",
                role=self.role,
                tool=tool_name,
                server=server or "-",
                blocked_attempts=self._client.blocked_attempts,
            )
            return client_error_result(
                tool_name=tool_name,
                server=server or "-",
                status="permission_denied",
                error_message=f"role '{self.role}' may not call '{tool_name}'",
                scenario_id=self._client.scenario_id,
            )
        self._log.log("mcp_call", role=self.role, tool=tool_name, server=server)
        return await self._client.call_tool(server, tool_name, arguments)


def utc_now() -> datetime:
    return datetime.now(UTC)
