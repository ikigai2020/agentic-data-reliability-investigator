"""Shared fixture-provider machinery (AD-006, FR-505, NFR-001).

Fixture providers read deterministic scenario JSON. Which scenario is active is
resolved from the environment at server launch (see plan ambiguity #2), keeping tool
inputs clean and letting the client select scenarios by launching servers with the
right env. Nothing here is imported by agent code — agents reach data only via MCP.

Environment variables:
    INVESTIGATOR_SCENARIO            active scenario id (directory name)
    INVESTIGATOR_SCENARIOS_DIR       base scenarios directory (default: data/scenarios)
    INVESTIGATOR_FORCE_PROVIDER_UNAVAILABLE   if "1", every call raises
                                              ProviderUnavailable (FR-508 test hook)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .provider_errors import ProviderNoData, ProviderUnavailable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCENARIOS_DIR = _REPO_ROOT / "data" / "scenarios"


def active_scenario_id() -> str:
    return os.environ.get("INVESTIGATOR_SCENARIO", "default")


def scenarios_dir() -> Path:
    override = os.environ.get("INVESTIGATOR_SCENARIOS_DIR")
    return Path(override) if override else _DEFAULT_SCENARIOS_DIR


def _force_unavailable() -> bool:
    return os.environ.get("INVESTIGATOR_FORCE_PROVIDER_UNAVAILABLE") == "1"


class FixtureProvider:
    """Loads and serves one server's fixture file for the active scenario."""

    def __init__(self, fixture_filename: str) -> None:
        self._fixture_filename = fixture_filename
        self._scenario_id = active_scenario_id()
        self._path = scenarios_dir() / self._scenario_id / fixture_filename
        self._data: dict[str, Any] | None = None

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    def _load(self) -> dict[str, Any]:
        if _force_unavailable():
            raise ProviderUnavailable(f"provider forced unavailable for {self._fixture_filename}")
        if self._data is None:
            if not self._path.exists():
                raise ProviderUnavailable(f"fixture not found: {self._path}")
            with self._path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)
        return self._data

    def section(self, name: str) -> dict[str, Any]:
        return self._load().get(name, {})

    def lookup(self, section: str, key: str) -> dict[str, Any]:
        """Return the raw record for ``key`` in ``section`` or raise ProviderNoData."""
        record = self.section(section).get(key)
        if record is None:
            raise ProviderNoData(f"no {section} for '{key}' in scenario '{self._scenario_id}'")
        return record
