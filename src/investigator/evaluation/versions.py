"""Component version attribution (FR-1206, NFR-009).

Every evaluation result must be attributable to the *exact* components that produced it,
because "accuracy went down" is not actionable until you know which of the model, the
prompts, the retrieval corpus, the tools, or the policy changed underneath it.

A :class:`VersionFingerprint` is content-addressed where it can be: prompts and config are
hashed rather than version-numbered, so an edit that nobody remembered to version still
shows up as a change. :func:`diff` names what moved between two runs, which is the input
FR-1206 needs to decide whether a regression evaluation is required before a change
becomes the default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..mcp_client.permissions import ROLE_PERMISSIONS


def _hash_files(paths: list[Path]) -> str:
    """Stable digest over file contents, ordered by name."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:12]


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class VersionFingerprint:
    """What produced a result (FR-1206)."""

    engine: str
    provider: str | None
    model: str | None
    prompts_digest: str
    corpus_digest: str
    corpus_documents: int
    tools_digest: str
    policy_digest: str
    beam_policy: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def differs_from(self, other: VersionFingerprint) -> bool:
        return self.as_dict() != other.as_dict()


def fingerprint(cfg: AppConfig) -> VersionFingerprint:
    """Capture the versions of everything that can change an outcome."""
    prompts_dir = Path(cfg.data_dir).parent / "config" / "prompts"
    prompt_files = list(prompts_dir.glob("*.md")) if prompts_dir.exists() else []

    corpus_files = [
        *(Path(cfg.incidents_dir).glob("*.json") if Path(cfg.incidents_dir).exists() else []),
        *(Path(cfg.runbooks_dir).glob("*.json") if Path(cfg.runbooks_dir).exists() else []),
    ]

    # Deterministic policy that decides outcomes — the thing most likely to be tuned and
    # least likely to be remembered as a "version".
    policy: dict[str, Any] = {
        "budgets": {
            "max_operational_calls": cfg.budgets.max_operational_calls,
            "max_rounds": cfg.budgets.max_rounds,
            "max_calls_per_task": cfg.budgets.max_calls_per_task,
            "beam_width": cfg.budgets.beam_width,
            "max_branch_depth": cfg.budgets.max_branch_depth,
            "max_candidate_actions": cfg.budgets.max_candidate_actions,
        },
        "search": {
            "prune_threshold": cfg.search.prune_threshold,
            "min_diverse_branches": cfg.search.min_diverse_branches,
        },
        "stopping": {
            "min_supporting": cfg.stopping.min_supporting_current_observations,
        },
        "retrieval": {
            "top_k": cfg.retrieval.top_k,
            "relevance_threshold": cfg.retrieval.relevance_threshold,
            "max_staleness_days": cfg.retrieval.max_staleness_days,
            "permitted_types": list(cfg.retrieval.permitted_types),
        },
        "verification": {"relative_tolerance": cfg.verification.relative_tolerance},
    }

    uses_llm = cfg.reasoning.uses_llm
    return VersionFingerprint(
        engine=cfg.reasoning.engine,
        provider=cfg.reasoning.provider if uses_llm else None,
        model=cfg.reasoning.resolved_model if uses_llm else None,
        prompts_digest=_hash_files(prompt_files),
        corpus_digest=_hash_files(corpus_files),
        corpus_documents=len(corpus_files),
        tools_digest=_hash_obj(
            {role: sorted(t for tools in servers.values() for t in tools)
             for role, servers in ROLE_PERMISSIONS.items()}
        ),
        policy_digest=_hash_obj(policy),
        beam_policy={**policy["search"], **policy["budgets"]},
    )


def diff(before: VersionFingerprint, after: VersionFingerprint) -> dict[str, tuple[Any, Any]]:
    """Which components moved between two runs.

    FR-1206: a material change requires regression evaluation before it becomes the
    default configuration. This is what tells you a change was material.
    """
    left, right = before.as_dict(), after.as_dict()
    return {
        key: (left[key], right[key])
        for key in left
        if key != "beam_policy" and left[key] != right[key]
    }


def describe(changes: dict[str, tuple[Any, Any]]) -> list[str]:
    """Human-readable change list for a drift report."""
    labels = {
        "engine": "reasoning engine",
        "provider": "LLM provider",
        "model": "model",
        "prompts_digest": "prompt text",
        "corpus_digest": "retrieval corpus",
        "corpus_documents": "corpus size",
        "tools_digest": "tool permissions",
        "policy_digest": "deterministic policy",
    }
    return [
        f"{labels.get(key, key)} changed: {before!r} -> {after!r}"
        for key, (before, after) in sorted(changes.items())
    ]
