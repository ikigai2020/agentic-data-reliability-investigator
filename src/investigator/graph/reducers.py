"""State reducers for the parent LangGraph (FR-307, FR-205).

Reducers append and deduplicate by stable IDs so that replaying a checkpoint never
duplicates accepted evidence or silently repeats completed work. Nodes return
partial updates; reducers merge them — nodes never mutate state in place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def _get_id(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def dedupe_by(key: str) -> Callable[[Sequence[T] | None, Sequence[T] | None], list[T]]:
    """Build a reducer that appends new items, keeping the first seen per ``key``.

    First-write-wins preserves evidence immutability (FR-303): an item whose ID was
    already accepted is dropped rather than overwritten.
    """

    def _reduce(existing: Sequence[T] | None, update: Sequence[T] | None) -> list[T]:
        merged: list[T] = list(existing or [])
        seen: set[Any] = {_get_id(item, key) for item in merged}
        for item in update or []:
            item_id = _get_id(item, key)
            if item_id is not None and item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item)
        return merged

    return _reduce


def upsert_by(key: str) -> Callable[[Sequence[T] | None, Sequence[T] | None], list[T]]:
    """Build a reducer that appends new items and replaces existing ones by ``key``.

    Used for mutable-status collections (hypotheses, tasks) whose *status* legitimately
    advances (e.g. ``active`` -> ``weakened``) while their identity is stable.
    """

    def _reduce(existing: Sequence[T] | None, update: Sequence[T] | None) -> list[T]:
        merged: list[T] = list(existing or [])
        index: dict[Any, int] = {
            _get_id(item, key): i for i, item in enumerate(merged) if _get_id(item, key) is not None
        }
        for item in update or []:
            item_id = _get_id(item, key)
            if item_id is not None and item_id in index:
                merged[index[item_id]] = item
            else:
                if item_id is not None:
                    index[item_id] = len(merged)
                merged.append(item)
        return merged

    return _reduce


def extend_unique(
    existing: Sequence[T] | None, update: Sequence[T] | None
) -> list[T]:
    """Append scalar items (e.g. failure strings), skipping exact duplicates."""
    merged: list[T] = list(existing or [])
    for item in update or []:
        if item not in merged:
            merged.append(item)
    return merged
