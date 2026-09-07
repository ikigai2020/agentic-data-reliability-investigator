"""Provider abstraction (§20).

OpenRouter and OpenAI speak the same chat-completions shape, so the provider only
changes the endpoint, the key, and the courtesy headers. Any other OpenAI-compatible
endpoint works by setting ``base_url``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx

from investigator.agents.llm import LLMReasoning
from investigator.agents.reasoning import DeterministicReasoning, build_reasoning
from investigator.config import PROVIDERS, load_config, provider_api_key

from ..search.fixtures import orders_alert


def _cfg(**overrides):
    base = load_config()
    return replace(base, reasoning=replace(base.reasoning, engine="llm", **overrides))


def test_each_provider_resolves_its_own_endpoint_key_and_model() -> None:
    openrouter = _cfg(provider="openrouter").reasoning
    assert openrouter.resolved_base_url == "https://openrouter.ai/api/v1"
    assert openrouter.env_key_name == "OPENROUTER_API_KEY"
    assert openrouter.wants_attribution_headers is True

    openai = _cfg(provider="openai").reasoning
    assert openai.resolved_base_url == "https://api.openai.com/v1"
    assert openai.env_key_name == "OPENAI_API_KEY"
    assert openai.wants_attribution_headers is False

    assert openrouter.resolved_model != openai.resolved_model


def test_an_explicit_model_or_endpoint_overrides_the_provider_default() -> None:
    cfg = _cfg(provider="openai", model="my-model", base_url="http://localhost:11434/v1")
    assert cfg.reasoning.resolved_model == "my-model"
    assert cfg.reasoning.resolved_base_url == "http://localhost:11434/v1"


def test_an_unknown_provider_falls_back_rather_than_crashing() -> None:
    cfg = _cfg(provider="not-a-provider")
    assert cfg.reasoning.resolved_base_url == PROVIDERS["openrouter"]["base_url"]


async def test_requests_go_to_the_configured_provider_with_the_right_headers() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": json.dumps({"hypotheses": []})}}],
            },
        )

    cfg = _cfg(provider="openai")
    engine = LLMReasoning(cfg, "sk-test", transport=httpx.MockTransport(handler))
    await engine.generate_hypotheses(orders_alert(), 5)
    await engine.aclose()

    assert seen["url"].startswith("https://api.openai.com/v1/chat/completions")
    assert seen["headers"]["authorization"] == "Bearer sk-test"
    assert seen["model"] == "gpt-4o-mini"
    # OpenRouter attribution headers must not be sent to other providers.
    assert "http-referer" not in seen["headers"]
    assert "x-title" not in seen["headers"]


async def test_openrouter_still_sends_its_attribution_headers() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"hypotheses": []})}}]},
        )

    engine = LLMReasoning(
        _cfg(provider="openrouter"), "sk-or-test", transport=httpx.MockTransport(handler)
    )
    await engine.generate_hypotheses(orders_alert(), 5)
    await engine.aclose()

    assert seen["x-title"]
    assert seen["http-referer"]


def test_the_key_is_read_from_the_providers_own_variable(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    assert provider_api_key(_cfg(provider="openai").reasoning) == "sk-openai"
    # An OpenAI key must not be mistaken for an OpenRouter one.
    assert provider_api_key(_cfg(provider="openrouter").reasoning) is None
    assert isinstance(build_reasoning(_cfg(provider="openrouter")), DeterministicReasoning)
    assert isinstance(build_reasoning(_cfg(provider="openai")), LLMReasoning)
