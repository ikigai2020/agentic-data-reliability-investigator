"""LangSmith tracing (M5.2, FR-1200, FR-1204, FR-1206).

LangGraph nodes and specialist subgraphs are instrumented through ``langchain-core``, so
the orchestration waterfall appears for free once tracing is switched on. Two things do
not come for free, and this module supplies both:

**Model calls.** The provider client speaks raw ``httpx``, so completions are invisible to
LangSmith no matter what the environment says. :func:`traced` wraps that one call so
prompts, latency, and token counts appear *inside* the graph nodes that caused them,
rather than the run showing an orchestration shape with hollow nodes.

**Attribution.** A trace nobody can attribute is a picture of a system that has since
moved on. Every run is tagged with its scenario, engine, model, and the FR-1206 version
fingerprint, which is also what makes traces sliceable per component (FR-1204).

Everything here fails safe. Tracing is off unless it is switched on *and* a key exists;
no key means no network, and the tests, the deterministic engine, and any reproducibility
run are unaffected (NFR-001, FR-1304). A tracing failure is never allowed to fail an
investigation — an investigation that dies because its telemetry backend was unreachable
is a worse outcome than an untraced investigation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from ..config import AppConfig, TracingConfig, tracing_api_key
from .logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])

# LangSmith's own span vocabulary. Kept as a literal so a typo is a type error rather
# than a span that renders as the wrong kind of thing in the trace.
RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

# Prompt text never leaves the process under these keys when include_prompts is off.
_PROMPT_FIELDS = ("system", "user")


def is_enabled(cfg: AppConfig | TracingConfig | None = None) -> bool:
    """True only when tracing is switched on *and* a key exists to send it with."""
    tracing = _tracing(cfg)
    return bool(tracing.enabled) and tracing_api_key() is not None


def _tracing(cfg: AppConfig | TracingConfig | None) -> TracingConfig:
    if cfg is None:
        from ..config import load_config

        return load_config().tracing
    return cfg if isinstance(cfg, TracingConfig) else cfg.tracing


def activate(cfg: AppConfig | None = None) -> bool:
    """Put the environment into the state ``langchain-core`` reads, and say what happened.

    Config can enable tracing, but ``langchain-core`` only consults the environment — so
    a ``tracing.enabled: true`` that never reached ``LANGSMITH_TRACING`` would silently
    do nothing. This is the one place the process mutates its own environment, and it
    only ever turns tracing *on* for a run that asked for it.
    """
    tracing = _tracing(cfg)
    log = get_logger()

    if not tracing.enabled:
        return False
    if tracing_api_key() is None:
        log.log(
            "tracing",
            enabled=False,
            reason="LANGSMITH_API_KEY is not set; tracing requested but not activated",
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = tracing.project
    if tracing.endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = tracing.endpoint
    log.log(
        "tracing",
        enabled=True,
        project=tracing.project,
        include_prompts=tracing.include_prompts,
    )
    return True


def run_metadata(
    cfg: AppConfig,
    *,
    scenario_id: str | None = None,
    investigation_id: str | None = None,
) -> dict[str, Any]:
    """FR-1206 attribution attached to the trace, which is what makes it sliceable."""
    from ..evaluation.versions import fingerprint

    versions = fingerprint(cfg).as_dict()
    return {
        "scenario_id": scenario_id,
        "investigation_id": investigation_id,
        "engine": versions.get("engine"),
        "provider": versions.get("provider"),
        "model": versions.get("model"),
        "prompts_digest": versions.get("prompts_digest"),
        "corpus_digest": versions.get("corpus_digest"),
        "tools_digest": versions.get("tools_digest"),
        "policy_digest": versions.get("policy_digest"),
    }


def run_config(cfg: AppConfig, *, scenario_id: str | None = None) -> dict[str, Any]:
    """Runnable config extras naming and tagging the run for LangSmith."""
    metadata = run_metadata(cfg, scenario_id=scenario_id)
    tags = [
        f"scenario:{scenario_id}" if scenario_id else "scenario:none",
        f"engine:{metadata['engine']}",
    ]
    if metadata.get("model"):
        tags.append(f"model:{metadata['model']}")
    return {"run_name": f"investigation:{scenario_id or 'ad-hoc'}", "tags": tags,
            "metadata": metadata}


def _redact_prompts(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep the call's shape, drop its text (config ``tracing.include_prompts: false``)."""
    return {
        key: (f"<{len(value)} chars withheld>" if key in _PROMPT_FIELDS and isinstance(value, str)
              else value)
        for key, value in inputs.items()
    }


def _process_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    # Read per call, not at import: a run may enable tracing after this module loaded.
    from ..config import load_config

    if load_config().tracing.include_prompts:
        return inputs
    return _redact_prompts(inputs)


def traced(run_type: RunType = "llm", name: str | None = None) -> Callable[[F], F]:
    """Mark a function as a LangSmith span, or leave it exactly as it was.

    ``traceable`` is itself inert when tracing is off, so the decorator is applied
    unconditionally at import; the guard here is for an install without ``langsmith``,
    not for the disabled case.
    """

    def decorate(fn: F) -> F:
        try:
            from langsmith import traceable
        except ImportError:  # pragma: no cover - langsmith ships with langchain-core
            return fn
        return traceable(  # type: ignore[return-value]
            run_type=run_type, name=name or fn.__name__, process_inputs=_process_inputs
        )(fn)

    return decorate


@dataclass
class TraceLink:
    """Where a finished run can be found, if it was traced at all."""

    project: str | None = None
    run_id: str | None = None
    url: str | None = None

    @property
    def captured(self) -> bool:
        return self.run_id is not None

    def as_dict(self) -> dict[str, Any] | None:
        if not self.captured:
            return None
        return {"provider": "langsmith", "project": self.project, "run_id": self.run_id,
                "url": self.url}


@dataclass
class _Collector:
    """Holds the link so callers can read it after the context exits."""

    link: TraceLink = field(default_factory=TraceLink)


@contextmanager
def collect(cfg: AppConfig):
    """Capture the root run of whatever is invoked inside, when tracing is on.

    Yields a holder whose ``link`` is filled in on exit. The link is what the Traces view
    deep-links to; without it a trace exists but nobody can find it from the run.
    """
    holder = _Collector()
    if not is_enabled(cfg):
        yield holder
        return

    try:
        from langchain_core.tracers.context import collect_runs
    except ImportError:  # pragma: no cover - langchain-core is a hard dependency
        yield holder
        return

    project = cfg.tracing.project
    try:
        with collect_runs() as runs:
            yield holder
            traced_runs = list(getattr(runs, "traced_runs", []) or [])
    except Exception as exc:  # noqa: BLE001 - telemetry must never fail an investigation
        get_logger().log("tracing", enabled=True, captured=False, error=repr(exc))
        return

    if not traced_runs:
        return
    root = traced_runs[0]
    holder.link = TraceLink(project=project, run_id=str(root.id), url=_run_url(root, project))
    get_logger().log(
        "tracing", enabled=True, captured=True, project=project, run_id=holder.link.run_id
    )


def _run_url(run: Any, project: str) -> str | None:
    """Best-effort deep link. The run id is the durable part; the URL is a convenience."""
    try:
        from langsmith import Client

        return str(Client().get_run_url(run=run, project_name=project))
    except Exception:  # noqa: BLE001 - resolving a URL may need the network; the id suffices
        return None
