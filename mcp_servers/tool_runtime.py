"""Shared MCP tool runtime (FR-503, FR-507, FR-508).

Every tool on both servers runs through :func:`execute`, which supplies:
  * input validation against the tool's typed input contract (structured
    ``invalid_input`` on failure rather than an unhandled exception),
  * a stable ``request_id``,
  * a timeout (``asyncio.wait_for``),
  * provider-error translation into the FR-508 status taxonomy,
  * a uniform ``ToolResult`` envelope with provenance and timestamps.

Read-only is structural: tools only ever *read* through the provider protocol.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from investigator.schemas.results import ToolResult

from .provider_errors import ProviderNoData, ProviderUnavailable

TInput = TypeVar("TInput", bound=BaseModel)
TData = TypeVar("TData", bound=BaseModel)


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def _now() -> datetime:
    return datetime.now(UTC)


async def execute(
    *,
    tool_name: str,
    source_system: str,
    scenario_id: str,
    input_model: type[TInput],
    raw_inputs: dict[str, Any],
    provider_call: Callable[[TInput], Awaitable[BaseModel]],
    timeout_seconds: float,
    fixture_file: str,
) -> ToolResult:
    """Validate, invoke, and wrap a single tool call as a ``ToolResult``."""
    request_id = new_request_id()
    collected_at = _now()
    provenance: dict[str, Any] = {
        "source_system": source_system,
        "provider": "fixture",
        "fixture_file": fixture_file,
        "scenario_id": scenario_id,
        "read_only": True,
    }

    def envelope(
        status: str,
        *,
        data: dict[str, Any] | None = None,
        observed_at: datetime | None = None,
        warnings: list[str] | None = None,
        error_message: str | None = None,
    ) -> ToolResult:
        return ToolResult(
            request_id=request_id,
            tool_name=tool_name,
            source_system=source_system,
            scenario_id=scenario_id,
            observed_at=observed_at,
            collected_at=collected_at,
            status=status,  # type: ignore[arg-type]
            data=data or {},
            provenance=provenance,
            warnings=warnings or [],
            error_message=error_message,
        )

    # 1) Input validation -> structured invalid_input (FR-508).
    try:
        validated = input_model.model_validate(raw_inputs)
    except ValidationError as exc:
        return envelope("invalid_input", error_message=str(exc))

    # 2) Invoke provider with timeout + error taxonomy translation.
    try:
        result_model = await asyncio.wait_for(provider_call(validated), timeout=timeout_seconds)
    except TimeoutError:
        return envelope("timeout", error_message=f"tool exceeded {timeout_seconds}s")
    except ProviderNoData as exc:
        return envelope("no_data", error_message=str(exc))
    except ProviderUnavailable as exc:
        return envelope("provider_unavailable", error_message=str(exc))
    except ValidationError as exc:  # provider produced a shape violating the contract
        return envelope("malformed_response", error_message=str(exc))
    except Exception as exc:  # noqa: BLE001 - defensive: never leak raw tracebacks as data
        return envelope("malformed_response", error_message=repr(exc))

    # 3) Success. Fixtures represent current observations captured now (M1 deviation:
    #    observed_at == collected_at). freshness is therefore "current".
    return envelope("ok", data=result_model.model_dump(mode="json"), observed_at=collected_at)
