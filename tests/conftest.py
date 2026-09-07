"""Shared pytest fixtures and constants.

These tests are fully offline and deterministic (FR-1304): they launch the real MCP
servers as stdio subprocesses against fixture scenarios — no network, no API key.

Note: we intentionally do NOT provide a yielding async fixture for the MCP client.
``mcp.client.stdio`` uses anyio cancel scopes that must be entered and exited in the
same task; pytest-asyncio runs generator-fixture teardown in a different task, which
raises "cancel scope in a different task". Each test therefore opens the client with
``async with InvestigatorMCPClient(...)`` in its own body.
"""

from __future__ import annotations

import pytest

from investigator.config import load_config

# Environment that could point a test at a real provider. The engine override is the
# dangerous one: a single `INVESTIGATOR_REASONING_ENGINE=llm` in a developer's .env, or
# `engine: llm` in committed config, would turn every scenario, journal and UI test into
# a billed API call without changing a line of test code.
_ENGINE_OVERRIDES = (
    "INVESTIGATOR_REASONING_ENGINE",
    "INVESTIGATOR_LLM_PROVIDER",
    "INVESTIGATOR_LLM_MODEL",
    "INVESTIGATOR_LLM_BASE_URL",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
)

# Keys `build_reasoning` consults before deciding it can reach a provider at all.
_PROVIDER_KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def offline_reasoning(monkeypatch: pytest.MonkeyPatch):
    """Pin the suite to the deterministic engine, with no provider key in reach.

    NFR-001 and FR-1304 require the suite to be reproducible and offline. That has been
    true by convention — committed config says ``engine: deterministic`` — but convention
    is not a control: ``load_dotenv()`` runs at import, so a real key on the machine is
    already in ``os.environ`` by the time any test starts, and one edited config value
    would spend it.

    This makes it structural. The engine is forced, provider keys are removed for the
    duration of each test, and the config cache is cleared on both sides so neither the
    forcing nor a test's own override leaks into its neighbours. The LLM adapter tests are
    unaffected: they build engines with an explicit literal key and an ``httpx``
    ``MockTransport``, which never reaches a socket.
    """
    for name in (*_ENGINE_OVERRIDES, *_PROVIDER_KEYS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVESTIGATOR_REASONING_ENGINE", "deterministic")
    load_config.cache_clear()
    yield
    load_config.cache_clear()

DIAGNOSED = "diagnosed_transformation_filter"
INCONCLUSIVE = "inconclusive_missing_evidence"

# M4.3 scenario corpus (FR-1301). The indices are the spec's own numbering.
NOT_AN_INCIDENT = "not_an_incident_business_volume"   # #4
MISLEADING_MEMORY = "diagnosed_misleading_memory"     # #5
