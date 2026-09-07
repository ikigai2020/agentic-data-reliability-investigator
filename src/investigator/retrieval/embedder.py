"""Deterministic offline text embedder (NFR-001, FR-1304).

Milestone 2 uses a hashing character-n-gram embedder behind the :class:`Embedder`
protocol so retrieval is fully reproducible offline with no model download or API key —
the same "deterministic default, swappable adapter" pattern as the reasoning engine. A
real embedding model (e.g. ``langchain-openai`` / sentence-transformers) can implement
the same protocol later without changing the retriever, index, or trust gate.

Vectors are L2-normalized so inner product equals cosine similarity in FAISS.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> np.ndarray: ...


class DeterministicEmbedder:
    """Hashing char-3-gram + word bag-of-features embedder.

    Deterministic across runs and machines: features are hashed into a fixed-width
    vector, then L2-normalized. Good enough to rank a small curated corpus by lexical
    overlap; the trust gate (FR-603) does the discriminating metadata filtering.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.sha1(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)
        # Word unigrams (stronger signal) ...
        for tok in tokens:
            vec[self._bucket(f"w:{tok}")] += 2.0
        # ... plus character 3-grams for fuzzy overlap.
        joined = " ".join(tokens)
        for i in range(len(joined) - 2):
            vec[self._bucket(f"c:{joined[i : i + 3]}")] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self.embed(t) for t in texts]).astype("float32")
