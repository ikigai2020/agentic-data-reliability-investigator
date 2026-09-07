"""Configuration loading (NFR-001, NFR-002).

Deterministic policy — tolerances, budgets, timeouts — is loaded from
``config/default.yaml`` and env overrides, kept entirely outside model prompts
(AD-004). Import this module rather than hard-coding thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# Repo root = two levels up from this file (src/investigator/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class VerificationConfig:
    relative_tolerance: float = 0.05


@dataclass(frozen=True)
class BudgetConfig:
    """Beam-search budgets (FR-702)."""

    max_operational_calls: int = 8
    max_rounds: int = 4
    max_calls_per_task: int = 3
    beam_width: int = 3
    max_branch_depth: int = 4
    max_candidate_actions: int = 2


@dataclass(frozen=True)
class SearchConfig:
    """Tree-of-Thought / beam policy (FR-702, FR-704)."""

    enabled: bool = True
    prune_threshold: int = 5
    min_diverse_branches: int = 2


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing and resume (FR-804)."""

    enabled: bool = True
    path: str = "data/checkpoints/investigations.sqlite"


# Provider registry. Both speak the OpenAI chat-completions shape, so only the endpoint,
# the key, and the courtesy headers differ.
PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "z-ai/glm-5.2:free",
        "attribution_headers": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "attribution_headers": False,
    },
}


@dataclass(frozen=True)
class ReasoningConfig:
    """Which reasoning engine backs the agents (AD-004, §20).

    ``deterministic`` is the offline, reproducible default used by the test suite
    (NFR-001, FR-1304). ``llm`` routes the judgment-based steps through OpenRouter.
    Deterministic *policy* — permissions, budgets, stopping, risk — is never affected by
    this setting; only interpretation and hypothesis generation change.
    """

    engine: str = "deterministic"
    provider: str = "openrouter"
    # Empty values inherit the provider's defaults, so switching provider is one word.
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: float = 60.0
    max_retries: int = 2
    # What to do when the model errors or returns unusable output after retries.
    # ``fallback`` degrades to deterministic reasoning and records it; ``fail_closed``
    # raises so the investigation abstains rather than reasoning from nothing.
    on_failure: str = "fallback"
    # Truncation applied to untrusted tool payloads before they reach a prompt (FR-1101).
    max_payload_chars: int = 4000
    # OpenRouter attribution headers (optional, but good citizenship; ignored elsewhere).
    app_title: str = "Agentic Data Reliability Investigator"
    app_url: str = "https://github.com/aidatamavens/agentic-data-reliability-investigator"

    @property
    def uses_llm(self) -> bool:
        return self.engine == "llm"

    @property
    def _provider(self) -> dict[str, Any]:
        return PROVIDERS.get(self.provider, PROVIDERS["openrouter"])

    @property
    def resolved_base_url(self) -> str:
        """Explicit ``base_url`` wins; otherwise the provider's endpoint."""
        return self.base_url or str(self._provider["base_url"])

    @property
    def resolved_model(self) -> str:
        return self.model or str(self._provider["default_model"])

    @property
    def env_key_name(self) -> str:
        return str(self._provider["env_key"])

    @property
    def wants_attribution_headers(self) -> bool:
        return bool(self._provider["attribution_headers"])


@dataclass(frozen=True)
class HypothesisConfig:
    min_count: int = 3
    max_count: int = 5


@dataclass(frozen=True)
class MCPConfig:
    transport: str = "stdio"
    tool_timeout_seconds: float = 5.0
    max_transport_retries: int = 1
    max_result_items: int = 100


@dataclass(frozen=True)
class StoppingConfig:
    min_supporting_current_observations: int = 2


@dataclass(frozen=True)
class MonitoringConfig:
    """FR-1205 threshold overrides, merged over the defaults by metric name."""

    enabled: bool = True
    thresholds: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TracingConfig:
    """LangSmith tracing (M5.2, FR-1200, FR-1204).

    **Off by default, and off unless a key exists.** The test suite and any
    reproducibility run must make no network calls (NFR-001, FR-1304), so tracing is
    something you turn on for a demo or a deployment, never something that turns itself
    on because a library happened to be installed.

    Enabling it exports prompts and tool payloads — which carry untrusted alert and log
    content — to an external service. ``include_prompts: false`` keeps the orchestration
    shape, model ids, latencies, and token counts while leaving the text behind.
    """

    enabled: bool = False
    project: str = "data-reliability-investigator"
    endpoint: str = ""  # blank = LangSmith's default host; set for self-hosted
    include_prompts: bool = True

    @property
    def env_key_name(self) -> str:
        return "LANGSMITH_API_KEY"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    redact_fields: tuple[str, ...] = ("password", "token", "secret", "credential", "api_key")


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = True
    embed_dim: int = 512
    top_k: int = 5
    relevance_threshold: float = 0.12
    max_staleness_days: int = 365
    permitted_types: tuple[str, ...] = ("incident", "runbook", "schema")


@dataclass(frozen=True)
class AppConfig:
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    hypotheses: HypothesisConfig = field(default_factory=HypothesisConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Model prices per 1M tokens, keyed by model id (see evaluation/pricing.py). Empty by
    # default: a stale hard-coded price is worse than an honestly absent one.
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    data_dir: Path = REPO_ROOT / "data"
    scenarios_dir: Path = REPO_ROOT / "data" / "scenarios"
    incidents_dir: Path = REPO_ROOT / "data" / "incidents"
    runbooks_dir: Path = REPO_ROOT / "data" / "runbooks"
    indexes_dir: Path = REPO_ROOT / "data" / "indexes"
    reviews_dir: Path = REPO_ROOT / "data" / "reviews"
    backlog_dir: Path = REPO_ROOT / "data" / "backlog"
    evaluations_dir: Path = REPO_ROOT / "evaluations"
    outputs_dir: Path = REPO_ROOT / "outputs"

    @property
    def journal_dir(self) -> Path:
        """Where run journals are written (M5.1). Beside the reports, not among them."""
        return Path(self.outputs_dir) / "journal"

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path to the SQLite checkpoint database (FR-804)."""
        return REPO_ROOT / self.checkpoints.path


def _reasoning_config(raw: dict[str, Any]) -> ReasoningConfig:
    """Build the reasoning config, letting environment variables win over the file.

    Env overrides exist so a model can be swapped for a single run without editing
    committed config: ``INVESTIGATOR_REASONING_ENGINE``, ``OPENROUTER_MODEL``,
    ``OPENROUTER_BASE_URL``.
    """
    merged = dict(raw)
    for env_key, field_name in (
        ("INVESTIGATOR_REASONING_ENGINE", "engine"),
        ("INVESTIGATOR_LLM_PROVIDER", "provider"),
        ("INVESTIGATOR_LLM_MODEL", "model"),
        ("INVESTIGATOR_LLM_BASE_URL", "base_url"),
        # Kept for continuity with the OpenRouter-only configuration.
        ("OPENROUTER_MODEL", "model"),
        ("OPENROUTER_BASE_URL", "base_url"),
    ):
        value = os.environ.get(env_key)
        if value:
            merged[field_name] = value
    return ReasoningConfig(**merged)


_TRUE = frozenset({"1", "true", "yes", "on"})


def _tracing_config(raw: dict[str, Any]) -> TracingConfig:
    """Build the tracing config, letting the environment win over the file.

    Both the current ``LANGSMITH_*`` names and the older ``LANGCHAIN_*`` ones are read,
    because that is what LangChain itself honours and a half-recognised toggle is worse
    than no toggle.
    """
    merged = dict(raw)
    for env_key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        value = os.environ.get(env_key)
        if value is not None:
            merged["enabled"] = value.strip().lower() in _TRUE
            break
    # The current name wins over the legacy one; either wins over the file.
    for field_name, env_keys in (
        ("project", ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")),
        ("endpoint", ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")),
    ):
        for env_key in env_keys:
            value = os.environ.get(env_key)
            if value:
                merged[field_name] = value
                break
    return TracingConfig(**merged)


def tracing_api_key() -> str | None:
    """The LangSmith key, read at call time so a late ``.env`` still counts (NFR-006)."""
    return os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or None


def provider_api_key(cfg: ReasoningConfig) -> str | None:
    """The configured provider's key, read at call time so a late ``.env`` still counts."""
    return os.environ.get(cfg.env_key_name) or None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    raw = _load_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)
    paths = raw.get("paths", {})
    return AppConfig(
        verification=VerificationConfig(**raw.get("verification", {})),
        budgets=BudgetConfig(**raw.get("budgets", {})),
        hypotheses=HypothesisConfig(**raw.get("hypotheses", {})),
        mcp=MCPConfig(**raw.get("mcp", {})),
        stopping=StoppingConfig(**raw.get("stopping", {})),
        reasoning=_reasoning_config(raw.get("reasoning", {})),
        search=SearchConfig(**raw.get("search", {})),
        checkpoints=CheckpointConfig(**raw.get("checkpoints", {})),
        tracing=_tracing_config(raw.get("tracing", {})),
        pricing={
            str(model): {str(k): float(v) for k, v in (rate or {}).items()}
            for model, rate in (raw.get("pricing") or {}).items()
        },
        monitoring=MonitoringConfig(
            enabled=raw.get("monitoring", {}).get("enabled", True),
            thresholds=raw.get("monitoring", {}).get("thresholds", {}),
        ),
        retrieval=RetrievalConfig(
            **{
                **raw.get("retrieval", {}),
                "permitted_types": tuple(
                    raw.get("retrieval", {}).get(
                        "permitted_types", ("incident", "runbook", "schema")
                    )
                ),
            }
        ),
        logging=LoggingConfig(
            level=raw.get("logging", {}).get("level", "INFO"),
            redact_fields=tuple(
                raw.get("logging", {}).get(
                    "redact_fields",
                    ("password", "token", "secret", "credential", "api_key"),
                )
            ),
        ),
        data_dir=REPO_ROOT / paths.get("data_dir", "data"),
        scenarios_dir=REPO_ROOT / paths.get("scenarios_dir", "data/scenarios"),
        incidents_dir=REPO_ROOT / paths.get("incidents_dir", "data/incidents"),
        runbooks_dir=REPO_ROOT / paths.get("runbooks_dir", "data/runbooks"),
        indexes_dir=REPO_ROOT / paths.get("indexes_dir", "data/indexes"),
        reviews_dir=REPO_ROOT / paths.get("reviews_dir", "data/reviews"),
        backlog_dir=REPO_ROOT / paths.get("backlog_dir", "data/backlog"),
        evaluations_dir=REPO_ROOT / paths.get("evaluations_dir", "evaluations"),
        outputs_dir=REPO_ROOT / paths.get("outputs_dir", "outputs"),
    )
