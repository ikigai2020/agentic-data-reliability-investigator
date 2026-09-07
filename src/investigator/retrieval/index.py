"""Vector index over the retrieval corpus (FR-600).

FAISS stores embeddings and vector IDs (§11); text and metadata live in the corpus
(JSON). ``IndexFlatIP`` over L2-normalized vectors gives exact cosine similarity, which
is appropriate for the small curated Milestone-2 corpus and keeps results deterministic.

A pure-NumPy fallback implements the same interface so the system still runs if FAISS is
unavailable; FAISS is used by default when importable.
"""

from __future__ import annotations

import numpy as np

try:  # FAISS is the primary backend (recommended stack, §20).
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover - fallback path
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False


class VectorIndex:
    """Cosine-similarity vector index mapping row position -> stable id."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype="float32")
        self._faiss_index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape[0] == 0:
            return
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, vectors]) if self._matrix.size else vectors
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, cosine_score)`` pairs, best first."""
        if not self._ids:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1).astype("float32"))
        k = min(top_k, len(self._ids))
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, k)
            pairs = [
                (self._ids[i], float(s))
                for i, s in zip(idx[0], scores[0], strict=False)
                if i >= 0
            ]
        else:  # pragma: no cover - fallback path
            sims = (self._matrix @ q[0]).astype(float)
            order = np.argsort(-sims)[:k]
            pairs = [(self._ids[i], float(sims[i])) for i in order]
        return pairs

    def __len__(self) -> int:
        return len(self._ids)
