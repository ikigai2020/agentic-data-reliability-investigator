"""The suite cannot reach a model provider (NFR-001, FR-1304).

Reproducibility is the reason these tests are trustworthy, and it is also what keeps a
developer's real API key from being spent by a routine `pytest` run. Both properties rest
on the autouse fixture in ``tests/conftest.py``; these assertions are what would notice if
it stopped working.
"""

from __future__ import annotations

import os

from investigator.agents.reasoning import DeterministicReasoning, build_reasoning
from investigator.config import load_config, provider_api_key


def test_the_configured_engine_is_deterministic() -> None:
    assert load_config().reasoning.uses_llm is False
    assert isinstance(build_reasoning(load_config()), DeterministicReasoning)


def test_no_provider_key_is_visible_to_a_test() -> None:
    """A real key on the machine is already in os.environ by import time; it must not be
    reachable from inside a test."""
    config = load_config()
    assert provider_api_key(config.reasoning) is None
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        assert os.environ.get(name) is None, name


def test_even_an_llm_configuration_degrades_rather_than_calling_out() -> None:
    """Belt and braces: with no key, requesting the LLM engine falls back and says so."""
    from dataclasses import replace

    config = load_config()
    as_llm = replace(config, reasoning=replace(config.reasoning, engine="llm"))
    assert as_llm.reasoning.uses_llm is True
    assert isinstance(build_reasoning(as_llm), DeterministicReasoning)
