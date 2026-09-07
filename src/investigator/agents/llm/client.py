"""Chat client for OpenAI-compatible providers (§20 recommended stack).

A thin async wrapper over ``/chat/completions``. OpenRouter and OpenAI speak the same
shape, so the provider only changes the endpoint, the key, and the courtesy headers —
see :data:`investigator.config.PROVIDERS`. Any other OpenAI-compatible endpoint (a local
proxy, another vendor) works by setting ``base_url``. Kept
deliberately small: retries, JSON extraction, and usage accounting, nothing else. All
diagnostic policy lives in deterministic code (AD-004), so this module never decides
anything — it moves text.

Two properties matter for the rest of the system:

* **Every call is observable.** Each completion produces an :class:`LLMCall` record with
  model, latency, and token usage — the FR-1200 fields that a deterministic engine
  cannot populate.
* **Failure is explicit.** A bad response raises :class:`LLMError` rather than returning
  something plausible-looking; the caller decides whether to fall back or fail closed.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...config import ReasoningConfig
from ...observability.tracing import traced

# Matches a fenced code block so ```json ... ``` wrappers can be stripped.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a completion cannot be obtained or parsed into usable JSON."""


@dataclass
class LLMCall:
    """One completion, recorded for the trace (FR-1200, FR-1203)."""

    purpose: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 1
    ok: bool = True
    error: str | None = None

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "attempts": self.attempts,
            "ok": self.ok,
            "error": self.error,
        }


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Free models frequently wrap JSON in prose or code fences even when asked not to, so
    this tries the whole string, then a fenced block, then the outermost brace pair.
    Raises :class:`LLMError` rather than guessing at malformed output.
    """
    candidates: list[str] = [text.strip()]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMError(f"no JSON object in response: {text[:200]!r}")


# A 429 means two completely different things. Backpressure clears on its own and is
# worth backing off for; an exhausted balance never will, and retrying it burns wall clock
# on every call of every run before degrading to the same place anyway.
_QUOTA_EXHAUSTED_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "no credits remaining",
    "exceeded your current quota",
    "billing",
    "payment required",
)

# Statuses no retry can change: a bad key, a forbidden model, a malformed request.
_PERMANENT_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 422})


def _quota_exhausted(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _QUOTA_EXHAUSTED_MARKERS)


def error_summary(body: str) -> str:
    """One readable line from a provider error body.

    Provider errors are pretty-printed JSON. Truncating that raw leaves a fallback note
    that breaks off mid-key — and these notes are read by a person deciding whether a run
    is trustworthy, so they have to say what happened in a sentence.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return " ".join(body.split())[:200]
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = " ".join(str(error.get("message") or "").split())
        kind = str(error.get("type") or error.get("code") or "").strip()
        if message:
            return f"{message} [{kind}]" if kind else message
    if isinstance(error, str):
        return " ".join(error.split())[:200]
    return " ".join(body.split())[:200]


@dataclass
class OpenRouterClient:
    """Async OpenRouter client. One instance per investigation; call ``aclose`` when done."""

    cfg: ReasoningConfig
    api_key: str
    calls: list[LLMCall] = field(default_factory=list)
    # Injectable for tests: an httpx transport stands in for the network so the adapter
    # can be exercised offline, with no key and no live model (NFR-001, FR-1304).
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.cfg.wants_attribution_headers:
            # OpenRouter attribution — optional, and never carries investigation data.
            headers["HTTP-Referer"] = self.cfg.app_url
            headers["X-Title"] = self.cfg.app_title
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.cfg.resolved_base_url.rstrip("/"),
                timeout=httpx.Timeout(self.cfg.timeout_seconds),
                headers=self._headers(),
                transport=self.transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @traced(run_type="llm", name="chat_completion")
    async def complete_json(
        self, *, purpose: str, system: str, user: str
    ) -> tuple[dict[str, Any], LLMCall]:
        """Request a JSON object completion, retrying transient failures.

        ``response_format`` is requested but not relied upon — many free models ignore
        it, which is why :func:`extract_json_object` exists.

        The ``traced`` decorator is what puts this call *inside* the graph node that
        caused it in a LangSmith trace. Without it the orchestration shape appears with
        hollow nodes: no prompts, no models, no tokens, no latency. It is inert unless
        tracing is switched on with a key (M5.2).
        """
        payload = {
            "model": self.cfg.resolved_model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        started = time.monotonic()
        last_error: str = "unknown error"
        attempts = 0

        for attempt in range(1, self.cfg.max_retries + 2):
            attempts = attempt
            try:
                response = await self._http().post("/chat/completions", json=payload)
                if response.status_code == 429 and _quota_exhausted(response.text):
                    # Billing, not backpressure. No amount of waiting makes credit appear,
                    # so degrade now rather than sleeping through the whole retry budget on
                    # every call of every round.
                    last_error = f"HTTP 429: {error_summary(response.text)}"
                    break
                if response.status_code in _PERMANENT_STATUSES:
                    last_error = f"HTTP {response.status_code}: {error_summary(response.text)}"
                    break
                if response.status_code in (408, 429) or response.status_code >= 500:
                    # Transient: rate limits are common on free models. Back off and retry.
                    last_error = f"HTTP {response.status_code}: {error_summary(response.text)}"
                    await asyncio.sleep(min(2.0 * attempt, 8.0))
                    continue
                if response.status_code >= 400:
                    raise LLMError(f"HTTP {response.status_code}: {error_summary(response.text)}")

                body = response.json()
                if body.get("error"):
                    raise LLMError(f"provider error: {str(body['error'])[:300]}")
                choices = body.get("choices") or []
                if not choices:
                    last_error = "response contained no choices"
                    continue
                content = (choices[0].get("message") or {}).get("content") or ""
                parsed = extract_json_object(content)

                usage = body.get("usage") or {}
                call = LLMCall(
                    purpose=purpose,
                    model=body.get("model") or self.cfg.resolved_model,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    attempts=attempts,
                )
                self.calls.append(call)
                return parsed, call

            except LLMError as exc:
                last_error = str(exc)
                if attempt > self.cfg.max_retries:
                    break
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt > self.cfg.max_retries:
                    break
                await asyncio.sleep(min(2.0 * attempt, 8.0))

        failed = LLMCall(
            purpose=purpose,
            model=self.cfg.resolved_model,
            latency_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            ok=False,
            error=last_error,
        )
        self.calls.append(failed)
        raise LLMError(last_error)


async def list_models(
    cfg: ReasoningConfig, api_key: str | None = None, *, free_only: bool = False
) -> list[str]:
    """Model IDs the configured provider will serve this key.

    Queried rather than hard-coded: OpenRouter's free roster churns constantly, and an
    OpenAI account only sees the models its tier grants.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        base_url=cfg.resolved_base_url.rstrip("/"),
        timeout=httpx.Timeout(30.0),
        headers=headers,
    ) as client:
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json().get("data") or []

    models: list[str] = []
    for model in data:
        model_id = model.get("id")
        if not model_id:
            continue
        if free_only:
            pricing = model.get("pricing") or {}
            try:
                is_free = float(pricing.get("prompt", 1)) == 0.0 and (
                    float(pricing.get("completion", 1)) == 0.0
                )
            except (TypeError, ValueError):
                is_free = False
            if not is_free:
                continue
        models.append(model_id)
    return sorted(models)
