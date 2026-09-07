"""Version attribution and drift detection (FR-1206)."""

from __future__ import annotations

from dataclasses import replace

from investigator.config import load_config
from investigator.evaluation.versions import describe, diff, fingerprint


def test_the_same_configuration_fingerprints_identically() -> None:
    cfg = load_config()
    assert fingerprint(cfg).as_dict() == fingerprint(cfg).as_dict()


def test_switching_engine_is_a_detected_change() -> None:
    cfg = load_config()
    llm = replace(cfg, reasoning=replace(cfg.reasoning, engine="llm"))
    changes = diff(fingerprint(cfg), fingerprint(llm))

    assert "engine" in changes
    assert "model" in changes  # a model appears where there was none
    assert any("reasoning engine changed" in line for line in describe(changes))


def test_tuning_deterministic_policy_is_a_detected_change() -> None:
    """The thing most likely to be tuned and least likely to be called a 'version'."""
    cfg = load_config()
    tuned = replace(cfg, search=replace(cfg.search, prune_threshold=99))
    changes = diff(fingerprint(cfg), fingerprint(tuned))

    assert "policy_digest" in changes
    assert any("deterministic policy changed" in line for line in describe(changes))


def test_a_stopping_threshold_change_is_detected() -> None:
    cfg = load_config()
    tuned = replace(
        cfg, stopping=replace(cfg.stopping, min_supporting_current_observations=1)
    )
    assert "policy_digest" in diff(fingerprint(cfg), fingerprint(tuned))


def test_a_corpus_edit_shows_up_without_anyone_versioning_it(tmp_path) -> None:
    cfg = load_config()
    before = fingerprint(cfg)

    empty = replace(cfg, incidents_dir=tmp_path, runbooks_dir=tmp_path)
    changes = diff(before, fingerprint(empty))

    assert "corpus_digest" in changes
    assert "corpus_documents" in changes


def test_an_unchanged_configuration_reports_no_drift() -> None:
    cfg = load_config()
    assert diff(fingerprint(cfg), fingerprint(cfg)) == {}
    assert describe({}) == []
