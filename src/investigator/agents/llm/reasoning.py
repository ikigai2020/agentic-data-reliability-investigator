"""LLM-backed reasoning engine (AD-004, FR-401, FR-1101).

Implements the same :class:`~investigator.agents.reasoning.ReasoningEngine` protocol as
:class:`~investigator.agents.reasoning.DeterministicReasoning`, so swapping engines
changes no agent, no contract, and no deterministic policy.

**What the model does.** Every step the spec describes as an agent decision:

* ``generate_hypotheses`` — reading an alert and proposing falsifiable candidate causes
  (FR-100.2).
* ``select_task_tools`` — the Commander choosing which checks to assign, from the tools
  the target specialist is permitted to call (FR-100.5).
* ``select_next_check`` — a specialist choosing its next check, or deciding it has enough
  and stopping early (FR-110/FR-120 "select permitted check").
* ``interpret`` — reading one observation and deciding what it supports or contradicts.

**What stays deterministic, and why.** Two things a model must not be able to move:

* ``plan_for_category`` maps a category to a specialist *role*. The role selects the
  permission set, so this is a **permission boundary** (FR-506), not a judgment call.
  Which tools to use *within* that role is judgment, and is the model's to make.
* ``tool_arguments`` builds MCP call arguments from validated alert fields. Letting a
  model shape tool inputs would turn every alert into an injection surface (FR-1101) for
  no benefit — the arguments are a mechanical field mapping.

Everything downstream of this module is unchanged: the trust gate, branch scoring,
stopping evaluator, grounding validator, and risk gate are deterministic and keep final
authority (FR-903). A model that returns nonsense — or is fully hijacked by injected
text — can only ever produce categories from the approved taxonomy, because every field
is validated against a closed vocabulary before it leaves this file.
"""

from __future__ import annotations

import json
from typing import Any

from ...config import AppConfig
from ...schemas.alert import Alert
from ...schemas.enums import RootCauseCategory
from ...schemas.hypothesis import Hypothesis
from ..prompts import PROMPT_VERSION, load_prompt
from ..reasoning import DeterministicReasoning, Interpretation
from .client import LLMCall, LLMError, OpenRouterClient

_MAX_STATEMENT_CHARS = 400
_MAX_SUMMARY_CHARS = 300

HYPOTHESIS_PROMPT_ID = f"hypothesis_generation:{PROMPT_VERSION}"
INTERPRETATION_PROMPT_ID = f"observation_interpretation:{PROMPT_VERSION}"


def _taxonomy_block() -> str:
    """Render the approved taxonomy from the enum, so FR-400 has one source of truth."""
    return "\n".join(f"- `{c.value}`" for c in RootCauseCategory)


def _fill(template: str, **values: str) -> str:
    """Substitute ``{{NAME}}`` placeholders without ``str.format`` (prompts contain JSON)."""
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _categories(raw: Any) -> list[RootCauseCategory]:
    """Coerce model output into approved categories, silently dropping anything else."""
    if not isinstance(raw, list):
        return []
    out: list[RootCauseCategory] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        try:
            category = RootCauseCategory(item.strip().lower())
        except ValueError:
            continue
        if category not in out:
            out.append(category)
    return out


def _text(raw: Any, limit: int, default: str = "") -> str:
    if not isinstance(raw, str):
        return default
    return " ".join(raw.split())[:limit] or default


class LLMReasoning:
    """OpenRouter-backed :class:`ReasoningEngine`.

    ``fallbacks`` accumulates a human-readable note for every step that degraded to
    deterministic reasoning, so a run is never quietly half-modelled — the report says
    which steps the model actually decided.
    """

    def __init__(self, cfg: AppConfig, api_key: str, *, transport: Any = None) -> None:
        self.cfg = cfg
        self.reasoning_cfg = cfg.reasoning
        self.client = OpenRouterClient(
            cfg=cfg.reasoning, api_key=api_key, transport=transport
        )
        self.deterministic = DeterministicReasoning()
        self.fallbacks: list[str] = []

    # --- lifecycle ------------------------------------------------------- #
    async def aclose(self) -> None:
        await self.client.aclose()

    def drain_calls(self) -> list[LLMCall]:
        """Return and clear the recorded completions (FR-1200 trace fields)."""
        calls = list(self.client.calls)
        self.client.calls.clear()
        return calls

    def _degrade(self, step: str, error: str) -> None:
        """Record a fallback, or re-raise when policy says fail closed."""
        note = f"{step}: LLM reasoning unavailable ({error}); used deterministic rules"
        if self.reasoning_cfg.on_failure == "fail_closed":
            raise LLMError(note)
        if note not in self.fallbacks:
            self.fallbacks.append(note)

    # --- deterministic passthrough (never model-controlled) --------------- #
    def plan_for_category(self, category: RootCauseCategory) -> tuple[str, list[str]]:
        """FR-506 permission boundary — the *role* is deterministic by design."""
        return self.deterministic.plan_for_category(category)

    def tool_arguments(self, tool_name: str, alert: Alert) -> dict:
        """FR-1101 — tool inputs are built from validated alert fields, never generated."""
        return self.deterministic.tool_arguments(tool_name, alert)

    # --- FR-100.5 check selection (Commander) ----------------------------- #
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
        """Choose which checks to assign, from the tools ``role`` may actually call.

        The permitted list is the hard bound: anything the model names outside it is
        dropped, and an empty selection falls back to the deterministic plan. Choosing
        *within* a permission set is judgment; the set itself is not negotiable.
        """
        if not permitted:
            return []
        system = _fill(
            load_prompt("tool_selection"),
            ROLE=role,
            PERMITTED="\n".join(f"- `{t}`" for t in permitted),
            MAX_ACTIONS=str(max_actions),
        )
        user = (
            "<ALERT>\n"
            f"{json.dumps(alert.model_dump(mode='json'), default=str)}\n"
            "</ALERT>\n\n"
            "<HYPOTHESIS>\n"
            f"{hypothesis.statement}\n"
            f"Question to answer: {hypothesis.discriminating_question}\n"
            "</HYPOTHESIS>\n\n"
            "Select the checks as JSON."
        )

        try:
            payload, _call = await self.client.complete_json(
                purpose="select_task_tools", system=system, user=user
            )
        except LLMError as exc:
            self._degrade("select_task_tools", str(exc))
            return default_plan[:max_actions]

        allowed = set(permitted)
        chosen: list[str] = []
        for name in payload.get("tools") or []:
            if isinstance(name, str) and name in allowed and name not in chosen:
                chosen.append(name)
            if len(chosen) >= max_actions:
                break
        return chosen or default_plan[:max_actions]

    # --- FR-110/120 next-check selection (specialists) -------------------- #
    async def select_next_check(
        self,
        alert: Alert,
        *,
        question: str,
        remaining: list[str],
        observations: list[str],
        calls_left: int,
    ) -> str | None:
        """Pick the next check, or stop early when the observations already answer it.

        Returning ``None`` ends the task with budget unspent — a real outcome the
        deterministic engine could never produce, since it always walks its whole plan.
        """
        if not remaining or calls_left <= 0:
            return None
        system = _fill(
            load_prompt("next_check"),
            REMAINING="\n".join(f"- `{t}`" for t in remaining),
        )
        seen = "\n".join(f"- {o}" for o in observations) or "- (nothing observed yet)"
        user = (
            f"Task question: {question}\n"
            f"Checks left in budget: {calls_left}\n\n"
            "<OBSERVATIONS>\n"
            f"{seen}\n"
            "</OBSERVATIONS>\n\n"
            "Decide the next check as JSON."
        )

        try:
            payload, _call = await self.client.complete_json(
                purpose="select_next_check", system=system, user=user
            )
        except LLMError as exc:
            self._degrade("select_next_check", str(exc))
            return remaining[0]

        choice = payload.get("tool")
        if isinstance(choice, str) and choice in set(remaining):
            return choice
        # `null`, a hallucinated name, or an already-used tool all mean "stop".
        return None

    # --- FR-401 hypothesis generation ------------------------------------ #
    async def generate_hypotheses(self, alert: Alert, max_count: int) -> list[Hypothesis]:
        min_count = self.cfg.hypotheses.min_count
        system = _fill(
            load_prompt("hypothesis_generation"),
            TAXONOMY=_taxonomy_block(),
            MIN_COUNT=str(min_count),
            MAX_COUNT=str(max_count),
        )
        user = (
            "<ALERT>\n"
            f"{json.dumps(alert.model_dump(mode='json'), indent=2)}\n"
            "</ALERT>\n\n"
            "Propose the ranked hypotheses as JSON."
        )

        try:
            payload, _call = await self.client.complete_json(
                purpose="generate_hypotheses", system=system, user=user
            )
        except LLMError as exc:
            self._degrade("generate_hypotheses", str(exc))
            return await self.deterministic.generate_hypotheses(alert, max_count)

        hypotheses = self._build_hypotheses(payload, alert, max_count)
        if len(hypotheses) < min_count:
            hypotheses = await self._top_up(hypotheses, alert, max_count, min_count)
        return hypotheses

    def _build_hypotheses(
        self, payload: dict[str, Any], alert: Alert, max_count: int
    ) -> list[Hypothesis]:
        """Validate model output into contracts. IDs and ranks are assigned here, not there.

        Hypothesis IDs key branch IDs, task IDs, and evidence IDs across the whole graph,
        so they are always generated deterministically from the category — never taken
        from the model.
        """
        raw_items = payload.get("hypotheses")
        if not isinstance(raw_items, list):
            return []

        seen: set[RootCauseCategory] = set()
        hypotheses: list[Hypothesis] = []
        for item in raw_items:
            if not isinstance(item, dict) or len(hypotheses) >= max_count:
                continue
            categories = _categories([item.get("category")])
            if not categories or categories[0] in seen:
                continue
            category = categories[0]
            seen.add(category)
            rank = len(hypotheses) + 1
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"H{rank}-{category.value}",
                    category=category,
                    statement=_text(
                        item.get("statement"),
                        _MAX_STATEMENT_CHARS,
                        default=f"Possible {category.value} cause for {alert.dataset}.",
                    ),
                    discriminating_question=_text(
                        item.get("discriminating_question"),
                        _MAX_STATEMENT_CHARS,
                        default=f"What current evidence would refute a {category.value} cause?",
                    ),
                    rank=rank,
                    status="active",
                    origin="initial_generation",
                )
            )
        return hypotheses

    async def _top_up(
        self,
        hypotheses: list[Hypothesis],
        alert: Alert,
        max_count: int,
        min_count: int,
    ) -> list[Hypothesis]:
        """FR-401 requires at least ``min_count`` distinct hypotheses.

        A model that returns too few is completed from the deterministic symptom ranking
        rather than rejected outright — and the top-up is recorded.
        """
        self._degrade(
            "generate_hypotheses",
            f"model returned {len(hypotheses)} valid hypotheses, need {min_count}",
        )
        used = {h.category for h in hypotheses}
        merged = list(hypotheses)
        for candidate in await self.deterministic.generate_hypotheses(alert, max_count):
            if len(merged) >= max_count:
                break
            if candidate.category in used:
                continue
            used.add(candidate.category)
            merged.append(candidate)

        # Re-rank so ranks stay ordinal and IDs stay consistent with them.
        return [
            h.model_copy(
                update={"rank": i, "hypothesis_id": f"H{i}-{h.category.value}"}
            )
            for i, h in enumerate(merged, start=1)
        ]

    # --- observation interpretation --------------------------------------- #
    async def interpret(
        self, tool_name: str, data: dict, *, alert: Alert | None = None
    ) -> Interpretation:
        system = _fill(load_prompt("observation_interpretation"), TAXONOMY=_taxonomy_block())
        payload_text = json.dumps(data, default=str)[: self.reasoning_cfg.max_payload_chars]
        context = ""
        if alert is not None:
            context = (
                f"Incident: {alert.symptom_type} on dataset '{alert.dataset}' "
                f"in pipeline '{alert.pipeline}'.\n\n"
            )
        user = (
            f"{context}"
            f"Tool: {tool_name}\n\n"
            "<OBSERVATION>\n"
            f"{payload_text}\n"
            "</OBSERVATION>\n\n"
            "Interpret this observation as JSON."
        )

        try:
            parsed, _call = await self.client.complete_json(
                purpose=f"interpret:{tool_name}", system=system, user=user
            )
        except LLMError as exc:
            self._degrade(f"interpret:{tool_name}", str(exc))
            return await self.deterministic.interpret(tool_name, data, alert=alert)

        supports = _categories(parsed.get("supports"))
        contradicts = _categories(parsed.get("contradicts"))

        # A category cannot be both supported and contradicted by one observation.
        # Rather than pick a side, drop it — an incoherent signal is no signal.
        overlap = set(supports) & set(contradicts)
        if overlap:
            supports = [c for c in supports if c not in overlap]
            contradicts = [c for c in contradicts if c not in overlap]

        return Interpretation(
            supports=set(supports),
            contradicts=set(contradicts),
            summary=_text(
                parsed.get("summary"), _MAX_SUMMARY_CHARS, default=f"{tool_name} observation"
            ),
            discriminating=bool(parsed.get("discriminating")) and bool(supports or contradicts),
        )
