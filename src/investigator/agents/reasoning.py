"""Deterministic reasoning engine (AD-004, FR-401, confirmed design decision).

Rule-based, seedless, offline reasoning behind the :class:`ReasoningEngine` protocol, so
tests are fully deterministic (FR-1304) and need no API key. This is the engine the test
suite and any reproducibility run use.

The LLM adapter in :mod:`investigator.agents.llm` implements the same protocol against
OpenRouter and is what a real deployment runs; :func:`build_reasoning` picks between them
from config. Neither engine can affect deterministic policy — permissions, budgets,
branch pruning, stopping, grounding, and the risk gate are code in every configuration.

This module owns three deterministic capabilities:
  * hypothesis generation from a verified alert (Commander),
  * mapping a hypothesis category to a specialist + permitted tools (Commander),
  * interpreting a normalized tool observation into support/contradict signals
    against the root-cause taxonomy (specialists).

All diagnostic *policy* lives here in code, never in a model prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..observability.logging import get_logger
from ..schemas.alert import Alert
from ..schemas.enums import RootCauseCategory, SymptomType
from ..schemas.hypothesis import Hypothesis

# --------------------------------------------------------------------------- #
# Symptom -> ranked candidate categories (FR-400, FR-401). Ordered by initial rank,
# derived from alert/verification data only — no fabricated probabilities.
# --------------------------------------------------------------------------- #
SYMPTOM_HYPOTHESES: dict[SymptomType, list[RootCauseCategory]] = {
    "volume_drop": [
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.SOURCE_DATA,
        RootCauseCategory.ORCHESTRATION,
        RootCauseCategory.DATA_QUALITY,
    ],
    "volume_spike": [
        RootCauseCategory.SOURCE_DATA,
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.PROCESSING_STATE,
        RootCauseCategory.LEGITIMATE_BUSINESS_CHANGE,
    ],
    "freshness_delay": [
        RootCauseCategory.ORCHESTRATION,
        RootCauseCategory.SOURCE_DATA,
        RootCauseCategory.PROCESSING_STATE,
        RootCauseCategory.INFRASTRUCTURE,
    ],
    "null_spike": [
        RootCauseCategory.SOURCE_DATA,
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.DATA_QUALITY,
        RootCauseCategory.SCHEMA_CONTRACT,
    ],
    "uniqueness_failure": [
        RootCauseCategory.PROCESSING_STATE,
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.DATA_QUALITY,
    ],
    "referential_integrity_failure": [
        RootCauseCategory.SOURCE_DATA,
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.DATA_QUALITY,
    ],
    "schema_change": [
        RootCauseCategory.SCHEMA_CONTRACT,
        RootCauseCategory.TRANSFORMATION_LOGIC,
        RootCauseCategory.SOURCE_DATA,
    ],
    "pipeline_failure": [
        RootCauseCategory.ORCHESTRATION,
        RootCauseCategory.INFRASTRUCTURE,
        RootCauseCategory.PROCESSING_STATE,
    ],
    "duplicate_processing": [
        RootCauseCategory.PROCESSING_STATE,
        RootCauseCategory.ORCHESTRATION,
        RootCauseCategory.TRANSFORMATION_LOGIC,
    ],
}

# Category -> (specialist role, ordered tool plan). One task -> one specialist.
#
# The plan is ordered most-discriminating first and is longer than one round's worth of
# actions: beam search caps candidate actions per branch per round (FR-702), so entries
# beyond that cap are the fallbacks a later round may reach for. Every tool listed must
# be within the paired role's permissions (FR-506) — `allowed_tools` filters again at
# planning time, and `search.thought.candidate_tools` applies the same filter so the
# planner and the search agree on what is actually reachable.
CATEGORY_PLAN: dict[RootCauseCategory, tuple[str, list[str]]] = {
    RootCauseCategory.TRANSFORMATION_LOGIC: (
        "data_investigator",
        ["compare_source_and_target", "get_recent_transformation_changes", "get_table_metrics"],
    ),
    RootCauseCategory.SOURCE_DATA: (
        "pipeline_investigator",
        ["get_upstream_dependencies", "get_processing_watermark", "get_pipeline_run_status"],
    ),
    RootCauseCategory.ORCHESTRATION: (
        "pipeline_investigator",
        ["get_pipeline_run_status", "get_task_failures", "get_execution_logs"],
    ),
    RootCauseCategory.INFRASTRUCTURE: (
        "pipeline_investigator",
        ["get_task_failures", "get_execution_logs", "get_pipeline_run_status"],
    ),
    RootCauseCategory.PROCESSING_STATE: (
        "pipeline_investigator",
        ["get_processing_watermark", "get_pipeline_run_status", "get_task_failures"],
    ),
    RootCauseCategory.SCHEMA_CONTRACT: (
        "data_investigator",
        ["get_schema_changes", "get_table_metrics"],
    ),
    RootCauseCategory.DATA_QUALITY: (
        "data_investigator",
        ["get_quality_results", "get_table_metrics", "compare_source_and_target"],
    ),
    RootCauseCategory.LEGITIMATE_BUSINESS_CHANGE: (
        "data_investigator",
        ["get_recent_transformation_changes", "get_table_metrics", "get_quality_results"],
    ),
}

# Human-readable statement/question templates per category.
_STATEMENTS: dict[RootCauseCategory, str] = {
    RootCauseCategory.TRANSFORMATION_LOGIC: (
        "A transformation/filter change in {pipeline} is dropping or altering rows for {dataset}."
    ),
    RootCauseCategory.SOURCE_DATA: (
        "Upstream source data feeding {dataset} arrived late, incomplete, or reduced."
    ),
    RootCauseCategory.ORCHESTRATION: (
        "The {pipeline} run failed, was skipped, or a task did not complete."
    ),
    RootCauseCategory.INFRASTRUCTURE: (
        "An infrastructure or resource failure interrupted {pipeline} execution."
    ),
    RootCauseCategory.PROCESSING_STATE: (
        "{pipeline} processed only part of the window (watermark/state incomplete)."
    ),
    RootCauseCategory.SCHEMA_CONTRACT: (
        "A schema or contract change on {dataset} broke downstream processing."
    ),
    RootCauseCategory.DATA_QUALITY: (
        "A data-quality defect (nulls/uniqueness/validity) affected {dataset}."
    ),
    RootCauseCategory.LEGITIMATE_BUSINESS_CHANGE: (
        "The change in {dataset} reflects legitimate business behavior, not a defect."
    ),
}

_QUESTIONS: dict[RootCauseCategory, str] = {
    RootCauseCategory.TRANSFORMATION_LOGIC: (
        "Does source/target reconciliation or a recent transformation change explain the delta?"
    ),
    RootCauseCategory.SOURCE_DATA: "Did upstream dependencies deliver complete, on-time data?",
    RootCauseCategory.ORCHESTRATION: "Did the pipeline run complete without task failures?",
    RootCauseCategory.INFRASTRUCTURE: "Were there infrastructure errors in the execution logs?",
    RootCauseCategory.PROCESSING_STATE: "Is the processing watermark complete for the window?",
    RootCauseCategory.SCHEMA_CONTRACT: "Did the dataset schema change during the window?",
    RootCauseCategory.DATA_QUALITY: "Do declared quality tests pass for the dataset?",
    RootCauseCategory.LEGITIMATE_BUSINESS_CHANGE: (
        "Is there evidence the change is expected business behavior?"
    ),
}


# Error signatures that point at the platform rather than the pipeline definition
# (FR-400: infrastructure vs orchestration). Matched against a failure's error_type and
# message, which the FR-302 task-failure contract already carries.
_INFRA_ERROR_MARKERS: tuple[str, ...] = (
    "oom",
    "out of memory",
    "memory",
    "disk",
    "resource",
    "quota",
    "capacity",
    "node",
    "container",
    "executor",
    "killed",
    "throttl",
)


def _looks_infrastructural(failure: dict) -> bool:
    text = f"{failure.get('error_type') or ''} {failure.get('message') or ''}".lower()
    return any(marker in text for marker in _INFRA_ERROR_MARKERS)


@dataclass(frozen=True)
class Interpretation:
    """Deterministic reading of one normalized observation."""

    supports: set[RootCauseCategory] = field(default_factory=set)
    contradicts: set[RootCauseCategory] = field(default_factory=set)
    summary: str = ""
    discriminating: bool = False


class ReasoningEngine(Protocol):
    """The seam between judgment and policy.

    The two interpretive methods are ``async`` because a real engine makes network calls;
    keeping them awaitable lets parallel specialists overlap their model calls instead of
    serialising them (FR-203). The two mechanical methods stay synchronous — they are a
    permission mapping and a field mapping, and never leave the process.
    """

    async def generate_hypotheses(self, alert: Alert, max_count: int) -> list[Hypothesis]: ...

    def plan_for_category(self, category: RootCauseCategory) -> tuple[str, list[str]]: ...

    def tool_arguments(self, tool_name: str, alert: Alert) -> dict: ...

    async def select_task_tools(
        self,
        alert: Alert,
        hypothesis: Hypothesis,
        *,
        role: str,
        permitted: list[str],
        default_plan: list[str],
        max_actions: int,
    ) -> list[str]: ...

    async def select_next_check(
        self,
        alert: Alert,
        *,
        question: str,
        remaining: list[str],
        observations: list[str],
        calls_left: int,
    ) -> str | None: ...

    async def interpret(
        self, tool_name: str, data: dict, *, alert: Alert | None = None
    ) -> Interpretation: ...


class DeterministicReasoning:
    """Rule-based :class:`ReasoningEngine` implementation for Milestone 1."""

    # --- Commander: hypothesis generation (FR-401) --------------------- #
    async def generate_hypotheses(self, alert: Alert, max_count: int) -> list[Hypothesis]:
        return self.generate_hypotheses_sync(alert, max_count)

    def generate_hypotheses_sync(self, alert: Alert, max_count: int) -> list[Hypothesis]:
        """The same rules without the protocol's async wrapper.

        The deterministic engine never awaits anything, so callers outside an event loop
        (test fixtures, offline tooling) can use this directly.
        """
        categories = SYMPTOM_HYPOTHESES.get(alert.symptom_type, [])[:max_count]
        hypotheses: list[Hypothesis] = []
        for i, category in enumerate(categories, start=1):
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"H{i}-{category.value}",
                    category=category,
                    statement=_STATEMENTS[category].format(
                        pipeline=alert.pipeline, dataset=alert.dataset
                    ),
                    discriminating_question=_QUESTIONS[category],
                    rank=i,
                    status="active",
                    origin="initial_generation",
                )
            )
        return hypotheses

    # --- Commander: task planning -------------------------------------- #
    def plan_for_category(self, category: RootCauseCategory) -> tuple[str, list[str]]:
        return CATEGORY_PLAN[category]

    async def select_task_tools(
        self,
        alert: Alert,
        hypothesis: Hypothesis,
        *,
        role: str,
        permitted: list[str],
        default_plan: list[str],
        max_actions: int,
    ) -> list[str]:
        """FR-100.5: the category plan, in order. The rules do not reconsider it."""
        return default_plan[:max_actions]

    async def select_next_check(
        self,
        alert: Alert,
        *,
        question: str,
        remaining: list[str],
        observations: list[str],
        calls_left: int,
    ) -> str | None:
        """FR-110/120 "select permitted check": take the next planned tool, in order."""
        return remaining[0] if remaining and calls_left > 0 else None

    def tool_arguments(self, tool_name: str, alert: Alert) -> dict:
        dataset = alert.dataset
        pipeline = alert.pipeline
        return {
            "get_pipeline_run_status": {"pipeline": pipeline},
            "get_task_failures": {"pipeline": pipeline},
            "get_execution_logs": {"pipeline": pipeline},
            "get_upstream_dependencies": {"pipeline": pipeline},
            "get_processing_watermark": {"pipeline": pipeline, "dataset": dataset},
            "get_table_metrics": {"dataset": dataset},
            "get_quality_results": {"dataset": dataset},
            "get_schema_changes": {"dataset": dataset},
            "compare_source_and_target": {
                "source_dataset": f"{dataset}_source",
                "target_dataset": dataset,
            },
            "get_recent_transformation_changes": {"pipeline": pipeline, "dataset": dataset},
        }[tool_name]

    # --- Specialists: observation interpretation ----------------------- #
    async def interpret(
        self, tool_name: str, data: dict, *, alert: Alert | None = None
    ) -> Interpretation:
        """``alert`` is accepted for protocol parity; the rules need no incident context."""
        return self.interpret_sync(tool_name, data)

    def interpret_sync(self, tool_name: str, data: dict) -> Interpretation:
        """The same rules without the protocol's async wrapper."""
        handler = getattr(self, f"_interpret_{tool_name}", None)
        if handler is None:
            return Interpretation(summary=f"{tool_name}: no interpretation rule")
        return handler(data)

    # -- pipeline tools --
    def _interpret_get_pipeline_run_status(self, d: dict) -> Interpretation:
        state = d.get("state")
        if state == "failed":
            return Interpretation(
                supports={RootCauseCategory.ORCHESTRATION},
                summary="pipeline run failed",
                discriminating=True,
            )
        if state in {"running", "no_run", "skipped"}:
            return Interpretation(
                supports={RootCauseCategory.PROCESSING_STATE},
                summary=f"pipeline run state={state} (incomplete)",
                discriminating=True,
            )
        if state == "success":
            return Interpretation(
                contradicts={
                    RootCauseCategory.ORCHESTRATION,
                    RootCauseCategory.INFRASTRUCTURE,
                    RootCauseCategory.PROCESSING_STATE,
                },
                summary="pipeline run succeeded and is terminal",
            )
        return Interpretation(summary=f"pipeline run state={state}")

    def _interpret_get_task_failures(self, d: dict) -> Interpretation:
        """A failed task always implicates orchestration; *why* it failed decides infra.

        Treating every failure as evidence for both orchestration and infrastructure left
        infrastructure permanently untestable — it could gain support but never be
        refuted, so it survived as an unweakened competitor and blocked FR-900 clause 4.
        The failure reason is the discriminator that was already in the contract.
        """
        failures = d.get("failures", [])
        if not failures:
            return Interpretation(
                contradicts={RootCauseCategory.ORCHESTRATION},
                summary="no task failures",
            )

        infra = [f for f in failures if _looks_infrastructural(f)]
        if infra:
            return Interpretation(
                supports={RootCauseCategory.ORCHESTRATION, RootCauseCategory.INFRASTRUCTURE},
                summary=(
                    f"{len(failures)} failed/blocked task(s); {len(infra)} with a "
                    "resource/infrastructure error"
                ),
                discriminating=True,
            )
        return Interpretation(
            supports={RootCauseCategory.ORCHESTRATION},
            contradicts={RootCauseCategory.INFRASTRUCTURE},
            summary=(
                f"{len(failures)} failed/blocked task(s), none with a resource or "
                "infrastructure error"
            ),
            discriminating=True,
        )

    def _interpret_get_execution_logs(self, d: dict) -> Interpretation:
        events = d.get("events", [])
        errors = [e for e in events if e.get("level") in {"ERROR", "CRITICAL"}]
        if errors:
            return Interpretation(
                supports={RootCauseCategory.INFRASTRUCTURE, RootCauseCategory.ORCHESTRATION},
                summary=f"{len(errors)} error log event(s)",
                discriminating=True,
            )
        return Interpretation(summary=f"{len(events)} log event(s), no errors")

    def _interpret_get_upstream_dependencies(self, d: dict) -> Interpretation:
        deps = d.get("dependencies", [])
        problematic = [x for x in deps if x.get("is_late") or x.get("state") != "success"]
        if problematic:
            return Interpretation(
                supports={RootCauseCategory.SOURCE_DATA, RootCauseCategory.ORCHESTRATION},
                summary=f"{len(problematic)} late/failed upstream dependency(ies)",
                discriminating=True,
            )
        if deps:
            return Interpretation(
                contradicts={RootCauseCategory.SOURCE_DATA},
                summary="all upstream dependencies succeeded on time",
                discriminating=True,
            )
        return Interpretation(summary="no upstream dependency data")

    def _interpret_get_processing_watermark(self, d: dict) -> Interpretation:
        lag = d.get("lag_seconds")
        if lag is None:
            return Interpretation(summary="no watermark data")
        if lag > 6 * 3600:
            return Interpretation(
                supports={RootCauseCategory.PROCESSING_STATE, RootCauseCategory.SOURCE_DATA},
                summary=f"processing lag {lag:.0f}s (large)",
                discriminating=True,
            )
        return Interpretation(
            contradicts={RootCauseCategory.PROCESSING_STATE},
            summary=f"processing lag {lag:.0f}s (within tolerance)",
        )

    # -- data tools --
    def _interpret_get_table_metrics(self, d: dict) -> Interpretation:
        # Table metrics primarily corroborate the *symptom* (used for verification),
        # so they are not treated as causal support for any single hypothesis.
        null_rate = d.get("null_rate", {})
        high_null = {c: r for c, r in null_rate.items() if r and r > 0.2}
        if high_null:
            return Interpretation(
                supports={RootCauseCategory.DATA_QUALITY, RootCauseCategory.SOURCE_DATA},
                summary=f"elevated null rate on {sorted(high_null)}",
                discriminating=True,
            )
        if (d.get("duplicate_count") or 0) > 0:
            return Interpretation(
                supports={RootCauseCategory.PROCESSING_STATE, RootCauseCategory.DATA_QUALITY},
                summary=f"{d.get('duplicate_count')} duplicate rows",
                discriminating=True,
            )
        rc, erc = d.get("row_count"), d.get("expected_row_count")
        delta = f" row_count={rc} vs expected={erc}" if rc is not None and erc is not None else ""
        return Interpretation(summary=f"table metrics observed{delta}")

    def _interpret_get_quality_results(self, d: dict) -> Interpretation:
        results = d.get("results", [])
        failures = [r for r in results if r.get("status") in {"fail", "error"}]
        if failures:
            return Interpretation(
                supports={RootCauseCategory.DATA_QUALITY},
                summary=f"{len(failures)} failing quality test(s)",
                discriminating=True,
            )
        if results:
            return Interpretation(
                contradicts={RootCauseCategory.DATA_QUALITY},
                summary="all declared quality tests pass",
            )
        return Interpretation(summary="no quality test results")

    def _interpret_get_schema_changes(self, d: dict) -> Interpretation:
        changes = d.get("changes", [])
        if changes:
            return Interpretation(
                supports={RootCauseCategory.SCHEMA_CONTRACT},
                summary=f"{len(changes)} schema change(s); breaking={d.get('breaking')}",
                discriminating=True,
            )
        return Interpretation(
            contradicts={RootCauseCategory.SCHEMA_CONTRACT},
            summary="no schema changes",
        )

    def _interpret_compare_source_and_target(self, d: dict) -> Interpretation:
        dropped = d.get("rows_dropped_by_filter") or 0
        source_count = d.get("source_count")
        target_count = d.get("target_count")
        if dropped > 0:
            return Interpretation(
                supports={RootCauseCategory.TRANSFORMATION_LOGIC},
                contradicts={RootCauseCategory.SOURCE_DATA},
                summary=f"{dropped} rows dropped at transform stage (source intact)",
                discriminating=True,
            )
        if (
            source_count is not None
            and target_count is not None
            and source_count < target_count
        ):
            return Interpretation(
                supports={RootCauseCategory.SOURCE_DATA},
                contradicts={RootCauseCategory.TRANSFORMATION_LOGIC},
                summary="source count below target — upstream shortfall",
                discriminating=True,
            )
        if source_count is not None and target_count is not None and source_count == target_count:
            return Interpretation(
                contradicts={RootCauseCategory.TRANSFORMATION_LOGIC},
                summary="source and target reconcile",
                discriminating=True,
            )
        return Interpretation(summary="reconciliation observed")

    def _interpret_get_recent_transformation_changes(self, d: dict) -> Interpretation:
        changes = d.get("changes", [])
        if changes:
            return Interpretation(
                supports={RootCauseCategory.TRANSFORMATION_LOGIC},
                summary=f"{len(changes)} recent transformation change(s)",
                discriminating=True,
            )
        return Interpretation(
            contradicts={RootCauseCategory.TRANSFORMATION_LOGIC},
            summary="no recent transformation changes",
        )


# --------------------------------------------------------------------------- #
# Engine selection
# --------------------------------------------------------------------------- #
def build_reasoning(cfg: Any) -> ReasoningEngine:
    """Return the reasoning engine named by config (``deterministic`` or ``llm``).

    Falls back to deterministic reasoning — loudly — when ``llm`` is requested without an
    ``OPENROUTER_API_KEY``, so a missing key degrades the run rather than crashing it.
    Import of the LLM adapter is deferred so the deterministic path never pays for it.
    """
    from ..config import provider_api_key

    if not getattr(cfg.reasoning, "uses_llm", False):
        return DeterministicReasoning()

    api_key = provider_api_key(cfg.reasoning)
    if not api_key:
        get_logger().log(
            "reasoning_engine",
            engine="deterministic",
            requested="llm",
            provider=cfg.reasoning.provider,
            reason=f"{cfg.reasoning.env_key_name} is not set",
        )
        return DeterministicReasoning()

    from .llm import LLMReasoning

    get_logger().log(
        "reasoning_engine",
        engine="llm",
        provider=cfg.reasoning.provider,
        model=cfg.reasoning.resolved_model,
    )
    return LLMReasoning(cfg, api_key)


def build_critic_judgment(cfg: Any, engine: ReasoningEngine | None = None) -> Any:
    """Return the Critic's LLM judgment layer, or ``None`` for a rule-only review.

    Reuses the reasoning engine's HTTP client and usage ledger when one is already open,
    so the Critic's completions appear in the same trace as everything else.
    """
    client = getattr(engine, "client", None)
    if client is None:
        return None

    from .llm import LLMCritic

    return LLMCritic(client, on_failure=cfg.reasoning.on_failure)
