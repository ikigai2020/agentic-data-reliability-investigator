"""Prompt loading (NFR-002: prompts kept separate from routing and policy)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"

PROMPT_VERSION = "v1"


@cache
def load_prompt(name: str) -> str:
    """Load a versioned role prompt by base name (e.g. ``commander``)."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        return f"[missing prompt: {name}]"
    return path.read_text(encoding="utf-8")
