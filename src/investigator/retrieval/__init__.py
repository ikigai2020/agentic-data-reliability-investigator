"""Retrieval and long-term memory (Milestone 2, §11).

Trust-gated retrieval over a confirmed corpus: memory may change investigation order or
suggest checks, but never establishes a diagnosis (AD-005).
"""

from __future__ import annotations

from .corpus import Corpus
from .embedder import DeterministicEmbedder, Embedder
from .index import VectorIndex

__all__ = ["Corpus", "DeterministicEmbedder", "Embedder", "VectorIndex"]
