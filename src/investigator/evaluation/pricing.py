"""Token pricing for the cost metric (FR-1203, FR-1302).

Cost is computed from the usage the provider actually reported, against rates supplied in
configuration. **No rates ship by default**, deliberately: model prices change without
warning, and a stale hard-coded price produces a confident wrong number — which is the
failure mode this whole system exists to avoid. An unpriced model contributes zero and is
named in the report, so "cost 0.00" is never mistaken for "cost was measured as zero".

Rates are per one million tokens, matching how providers publish them:

    pricing:
      gpt-4o-mini:
        input: 0.15
        output: 0.60
"""

from __future__ import annotations

from typing import Any

PER_TOKENS = 1_000_000


def rate_for(model: str, rates: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Exact match first, then the longest configured prefix.

    Providers append dated suffixes (``gpt-4o-mini-2024-07-18``), so a prefix match keeps
    a rate working across a point release without inviting a wrong match: the longest
    prefix wins, and an unrelated model matches nothing.
    """
    if model in rates:
        return rates[model]
    candidates = [key for key in rates if model.startswith(key)]
    if not candidates:
        return None
    return rates[max(candidates, key=len)]


def estimate(
    calls: list[dict[str, Any]], rates: dict[str, dict[str, float]]
) -> tuple[float, list[str]]:
    """Return ``(usd, models with no configured rate)`` for one run's completions."""
    total = 0.0
    unpriced: set[str] = set()
    for call in calls:
        model = str(call.get("model") or "")
        rate = rate_for(model, rates)
        if rate is None:
            if model:
                unpriced.add(model)
            continue
        prompt_tokens = int(call.get("prompt_tokens") or 0)
        completion_tokens = int(call.get("completion_tokens") or 0)
        total += (
            prompt_tokens * float(rate.get("input", 0.0))
            + completion_tokens * float(rate.get("output", 0.0))
        ) / PER_TOKENS
    return round(total, 6), sorted(unpriced)


def usage(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every completion a run made, read from its journal.

    The journal already attributes each call to the stage that made it, so the harness
    reads usage from there rather than draining the engine a second time — draining is
    destructive, and the stages have already consumed it.
    """
    return [call for stage in state.get("journal") or [] for call in stage.llm_calls]
