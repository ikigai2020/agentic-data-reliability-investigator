"""LangSmith tracing (M5.2, FR-1200, FR-1204, FR-1206).

The governing property is that tracing is *opt-in and fails safe*: the suite, the
deterministic engine, and any reproducibility run must make no network calls unless
someone deliberately turned tracing on and supplied a key (NFR-001, FR-1304).
"""

from __future__ import annotations

import inspect
import json

import pytest

from investigator.config import ReasoningConfig, load_config
from investigator.observability import tracing
from investigator.observability.tracing import TraceLink

TRACING_ENV = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """A process with no tracing environment at all, restored afterwards."""
    for name in TRACING_ENV:
        monkeypatch.delenv(name, raising=False)
    load_config.cache_clear()
    yield monkeypatch
    load_config.cache_clear()


# --------------------------------------------------------------------------- #
# Off by default, and off without a key
# --------------------------------------------------------------------------- #
def test_tracing_is_off_by_default(clean_env) -> None:
    cfg = load_config()
    assert cfg.tracing.enabled is False
    assert tracing.is_enabled(cfg) is False
    assert tracing.activate(cfg) is False


def test_a_key_alone_does_not_switch_tracing_on(clean_env) -> None:
    """Having a key lying around in the environment is not a request to be traced."""
    clean_env.setenv("LANGSMITH_API_KEY", "lsv2_test")
    load_config.cache_clear()
    assert tracing.is_enabled(load_config()) is False


def test_requesting_tracing_without_a_key_does_not_pretend_to_trace(clean_env) -> None:
    """Fail safe: the request is refused and recorded, not half-honoured."""
    clean_env.setenv("LANGSMITH_TRACING", "true")
    load_config.cache_clear()
    cfg = load_config()

    assert cfg.tracing.enabled is True  # the request was heard
    assert tracing.is_enabled(cfg) is False  # and refused
    assert tracing.activate(cfg) is False


def test_the_environment_switch_is_read_in_both_spellings(clean_env) -> None:
    clean_env.setenv("LANGCHAIN_TRACING_V2", "1")
    load_config.cache_clear()
    assert load_config().tracing.enabled is True


def test_activating_sets_what_langchain_core_actually_reads(clean_env) -> None:
    """Config can ask for tracing, but only the environment can deliver it."""
    clean_env.setenv("LANGSMITH_TRACING", "true")
    clean_env.setenv("LANGSMITH_API_KEY", "lsv2_test")
    clean_env.setenv("LANGSMITH_PROJECT", "test-project")
    load_config.cache_clear()
    cfg = load_config()

    assert tracing.activate(cfg) is True
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


# --------------------------------------------------------------------------- #
# FR-1206 attribution — what makes a trace sliceable
# --------------------------------------------------------------------------- #
def test_a_run_is_tagged_with_what_produced_it(clean_env) -> None:
    cfg = load_config()
    metadata = tracing.run_metadata(cfg, scenario_id="diagnosed_transformation_filter")

    assert metadata["scenario_id"] == "diagnosed_transformation_filter"
    assert metadata["engine"] == "deterministic"
    for digest in ("prompts_digest", "corpus_digest", "tools_digest", "policy_digest"):
        assert metadata[digest], digest
    # It has to survive the trip to an external service.
    assert json.loads(json.dumps(metadata))


def test_the_run_is_named_and_tagged_for_slicing(clean_env) -> None:
    config = tracing.run_config(load_config(), scenario_id="inconclusive_missing_evidence")

    assert config["run_name"] == "investigation:inconclusive_missing_evidence"
    assert "scenario:inconclusive_missing_evidence" in config["tags"]
    assert "engine:deterministic" in config["tags"]


# --------------------------------------------------------------------------- #
# The decorator
# --------------------------------------------------------------------------- #
async def test_tracing_a_function_does_not_change_it(clean_env) -> None:
    """An inert decorator is the whole basis of applying it unconditionally."""

    @tracing.traced(run_type="llm", name="test_call")
    async def double(*, value: int) -> int:
        return value * 2

    assert await double(value=21) == 42


async def test_a_traced_function_still_raises_what_it_raised(clean_env) -> None:
    @tracing.traced()
    async def boom() -> None:
        raise ValueError("provider error")

    with pytest.raises(ValueError, match="provider error"):
        await boom()


def test_prompts_are_sent_by_default_and_can_be_withheld(clean_env) -> None:
    inputs = {"purpose": "interpret", "system": "you are...", "user": "alert payload"}

    assert tracing._process_inputs(dict(inputs)) == inputs

    clean_env.setenv("LANGSMITH_TRACING", "true")
    load_config.cache_clear()
    withheld = tracing._redact_prompts(dict(inputs))
    assert withheld["purpose"] == "interpret"
    assert "you are" not in withheld["system"]
    assert "chars withheld" in withheld["system"]


def test_the_provider_api_key_never_enters_a_trace(clean_env) -> None:
    """``complete_json`` is a method on the object that holds the key.

    langsmith drops ``self`` before recording inputs, which is the only reason this is
    safe — so it is pinned here rather than assumed across an upgrade.
    """
    from langsmith.run_helpers import _get_inputs

    from investigator.agents.llm.client import OpenRouterClient

    client = OpenRouterClient(cfg=ReasoningConfig(), api_key="sk-super-secret")
    signature = inspect.signature(OpenRouterClient.complete_json)
    recorded = _get_inputs(signature, client, purpose="interpret", system="s", user="u")

    assert "self" not in recorded
    assert "sk-super-secret" not in json.dumps(recorded, default=str)


# --------------------------------------------------------------------------- #
# The link back to the run
# --------------------------------------------------------------------------- #
def test_an_uncaptured_link_is_absent_rather_than_empty() -> None:
    assert TraceLink().captured is False
    assert TraceLink().as_dict() is None
    assert TraceLink(project="p", run_id="r", url="u").as_dict() == {
        "provider": "langsmith",
        "project": "p",
        "run_id": "r",
        "url": "u",
    }


def test_collecting_with_tracing_off_touches_nothing(clean_env) -> None:
    cfg = load_config()
    with tracing.collect(cfg) as collected:
        pass
    assert collected.link.captured is False
