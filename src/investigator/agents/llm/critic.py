"""LLM judgment for the Evidence Critic (FR-130).

The Critic is a hybrid by design, and the split is deliberate rather than partial.

*Detection stays deterministic.* Finding a dangling evidence reference, a missing
``request_id``, an observation reused as support for two hypotheses, or a hypothesis that
was never tested is exact set arithmetic. Rules do it perfectly, cheaply, and identically
every run — asking a model to re-derive them would only add a way to miss one.

*Judgment goes to the model.* Whether a lead is earned or merely first, whether a
competitor was defeated or just ignored, whether two observations are genuinely
independent, and what single check would most change the picture — none of that reduces to
a rule, and FR-130.8 ("challenge Commander confirmation bias") is squarely in this half.

The model's findings are **merged with** the deterministic ones, never substituted for
them, and its stop recommendation stays advisory: deterministic code applies budgets,
pruning, and the stopping policy (FR-903).
"""

from __future__ import annotations

import json
from typing import Any

from ...schemas.enums import StopRecommendation
from ..prompts import PROMPT_VERSION, load_prompt
from .client import LLMError, OpenRouterClient

CRITIC_PROMPT_ID = f"critic_review:{PROMPT_VERSION}"

_MAX_ITEMS = 8
_MAX_ITEM_CHARS = 300
_VALID_RECOMMENDATIONS: frozenset[str] = frozenset({"continue", "diagnosed", "inconclusive"})


def _string_list(raw: Any) -> list[str]:
    """Bounded list of clean strings; anything else is dropped."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())[:_MAX_ITEM_CHARS]
        if text and text not in out:
            out.append(text)
        if len(out) >= _MAX_ITEMS:
            break
    return out


class CriticJudgment:
    """What the model contributed to a review. Empty when it was not consulted."""

    def __init__(
        self,
        *,
        bias_challenges: list[str] | None = None,
        evidence_gaps: list[str] | None = None,
        stop_recommendation: StopRecommendation | None = None,
        rationale: str = "",
    ) -> None:
        self.bias_challenges = bias_challenges or []
        self.evidence_gaps = evidence_gaps or []
        self.stop_recommendation = stop_recommendation
        self.rationale = rationale

    @property
    def consulted(self) -> bool:
        return bool(
            self.bias_challenges or self.evidence_gaps or self.stop_recommendation
        )


class LLMCritic:
    """Produces the judgment half of a :class:`CriticReview`."""

    def __init__(self, client: OpenRouterClient, on_failure: str = "fallback") -> None:
        self.client = client
        self.on_failure = on_failure
        self.fallbacks: list[str] = []

    async def judge(self, context: dict[str, Any]) -> CriticJudgment:
        """Review the round. Returns an empty judgment when the model is unavailable."""
        system = load_prompt("critic_review")
        user = (
            "<REVIEW>\n"
            f"{json.dumps(context, indent=2, default=str)}\n"
            "</REVIEW>\n\n"
            "Return your review as JSON."
        )
        try:
            payload, _call = await self.client.complete_json(
                purpose="critic_review", system=system, user=user
            )
        except LLMError as exc:
            note = (
                f"critic_review: LLM reasoning unavailable ({exc}); "
                "used deterministic rules"
            )
            if self.on_failure == "fail_closed":
                raise
            if note not in self.fallbacks:
                self.fallbacks.append(note)
            return CriticJudgment()

        raw_recommendation = payload.get("stop_recommendation")
        recommendation: StopRecommendation | None = None
        if isinstance(raw_recommendation, str):
            candidate = raw_recommendation.strip().lower()
            if candidate in _VALID_RECOMMENDATIONS:
                recommendation = candidate  # type: ignore[assignment]

        return CriticJudgment(
            bias_challenges=_string_list(payload.get("bias_challenges")),
            evidence_gaps=_string_list(payload.get("evidence_gaps")),
            stop_recommendation=recommendation,
            rationale=" ".join(str(payload.get("rationale", "")).split())[:_MAX_ITEM_CHARS],
        )
