"""Loading persisted runs for the UI (M5.3).

A run is two files: the report (``outputs/<id>.json``) says what was concluded, and the
journal (``outputs/journal/<id>.json``) says how the investigation moved. The UI needs
both, and must survive having only one — runs persisted before journals existed have no
journal, and a journal written for a blocked release has no report to go with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config
from ..schemas.journal import RunJournal, StageRecord


@dataclass(frozen=True)
class RunSummary:
    """One line in the run picker."""

    investigation_id: str
    scenario_id: str | None
    outcome: str | None
    release_status: str | None
    modified_at: float
    has_journal: bool

    @property
    def label(self) -> str:
        return f"{self.scenario_id or 'unknown'} — {self.outcome or 'no outcome'}"


@dataclass
class RunView:
    """One investigation, as much of it as was persisted."""

    investigation_id: str
    report: dict[str, Any] = field(default_factory=dict)
    escalation_package: dict[str, Any] | None = None
    release_status: str | None = None
    journal: RunJournal | None = None
    # Present only for a run executed in this session, where the full evidence objects
    # (payloads, provenance) are still in memory. Replays do not have them: payloads are
    # deliberately never written to disk (FR-1208 retention).
    live_state: dict[str, Any] | None = None

    @property
    def scenario_id(self) -> str | None:
        return (self.journal.scenario_id if self.journal else None) or (
            (self.report.get("alert") or {}).get("pipeline")
        )

    @property
    def has_journal(self) -> bool:
        return self.journal is not None and bool(self.journal.stages)

    @property
    def is_live(self) -> bool:
        return self.live_state is not None

    @property
    def rounds(self) -> list[int]:
        return self.journal.rounds() if self.journal else []

    def stages(self, node: str | None = None, round_number: int | None = None
               ) -> list[StageRecord]:
        """Stages, optionally narrowed to one node and/or one round."""
        if self.journal is None:
            return []
        stages = self.journal.stages
        if node is not None:
            stages = [s for s in stages if s.node == node]
        if round_number is not None:
            stages = [s for s in stages if s.round_number == round_number]
        return stages

    def first_delta(self, node: str, key: str, default: Any = None) -> Any:
        for stage in self.stages(node):
            if key in stage.delta:
                return stage.delta[key]
        return default

    def guardrails(self) -> dict[str, Any]:
        return self.report.get("guardrail_and_budget_events", {}) or {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_runs(cfg: AppConfig | None = None, *, limit: int = 50) -> list[RunSummary]:
    """Most recent runs first. Journalled runs are listed first — only they can replay."""
    cfg = cfg or load_config()
    summaries: list[RunSummary] = []

    for path in Path(cfg.journal_dir).glob("*.json") if cfg.journal_dir.exists() else []:
        raw = _read_json(path)
        if not raw:
            continue
        summaries.append(
            RunSummary(
                investigation_id=raw.get("investigation_id", path.stem),
                scenario_id=raw.get("scenario_id"),
                outcome=raw.get("outcome"),
                release_status=raw.get("release_status"),
                modified_at=path.stat().st_mtime,
                has_journal=True,
            )
        )

    summaries.sort(key=lambda s: s.modified_at, reverse=True)
    return summaries[:limit]


def load_run(investigation_id: str, cfg: AppConfig | None = None) -> RunView:
    """Load one run from disk, tolerating either half being absent."""
    cfg = cfg or load_config()
    report_raw = _read_json(Path(cfg.outputs_dir) / f"{investigation_id}.json")
    journal_raw = _read_json(Path(cfg.journal_dir) / f"{investigation_id}.json")

    journal: RunJournal | None = None
    if journal_raw:
        try:
            journal = RunJournal.model_validate(journal_raw)
        except Exception:  # noqa: BLE001 - a journal from an older shape must not crash the UI
            journal = None

    return RunView(
        investigation_id=investigation_id,
        report=report_raw.get("report") or {},
        escalation_package=report_raw.get("escalation_package"),
        release_status=report_raw.get("release_status") or (journal.release_status if journal
                                                            else None),
        journal=journal,
    )


def latest_run(cfg: AppConfig | None = None) -> RunView | None:
    runs = list_runs(cfg, limit=1)
    return load_run(runs[0].investigation_id, cfg) if runs else None


def view_from_state(state: dict[str, Any], cfg: AppConfig | None = None) -> RunView:
    """Build a view straight from a live run's final state, payloads included."""
    from ..graph.journal import build_journal

    return RunView(
        investigation_id=state.get("investigation_id", "live"),
        report=state.get("report") or {},
        escalation_package=state.get("escalation_package"),
        release_status=state.get("release_status"),
        journal=build_journal(state),
        live_state=state,
    )
