"""Retrieval corpus loading and vector search (FR-600, FR-601).

The corpus holds human-confirmed incidents and approved runbooks (and optional schema
docs). Current metrics/logs/run state are never represented here (FR-600) — those only
ever arrive as ``current_operational`` evidence via MCP.

Text + metadata live in JSON; embeddings + vector ids live in the FAISS index.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..schemas.retrieval import CorpusDocument, RetrievedItem
from .embedder import DeterministicEmbedder, Embedder
from .index import VectorIndex


def _embedding_text(doc: CorpusDocument) -> str:
    """Compose the text used to embed a document (title + body + metadata keywords)."""
    parts = [
        doc.title,
        doc.text,
        doc.type,
        doc.pipeline or "",
        doc.dataset or "",
        doc.symptom_type or "",
        doc.category.value if doc.category else "",
        doc.environment or "",
        doc.version or "",
    ]
    return " ".join(p for p in parts if p)


class Corpus:
    """An embedded, searchable collection of memory documents."""

    def __init__(
        self, documents: Iterable[CorpusDocument], embedder: Embedder | None = None
    ) -> None:
        self._embedder = embedder or DeterministicEmbedder()
        self.documents: dict[str, CorpusDocument] = {d.doc_id: d for d in documents}
        self._index = VectorIndex(dim=self._embedder.dim)
        ids = list(self.documents)
        if ids:
            vectors = self._embedder.embed_many(
                [_embedding_text(self.documents[i]) for i in ids]
            )
            self._index.add(ids, vectors)

    @property
    def backend(self) -> str:
        return self._index.backend

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query_text: str, top_k: int) -> list[RetrievedItem]:
        query_vec = self._embedder.embed(query_text)
        hits = self._index.search(query_vec, top_k)
        items: list[RetrievedItem] = []
        for rank, (doc_id, score) in enumerate(hits, start=1):
            items.append(
                RetrievedItem(document=self.documents[doc_id], score=max(0.0, score), rank=rank)
            )
        return items

    # --- loading -------------------------------------------------------- #
    @classmethod
    def from_dirs(
        cls, *dirs: Path, embedder: Embedder | None = None
    ) -> Corpus:
        """Load every ``*.json`` document under the given directories."""
        docs: list[CorpusDocument] = []
        for directory in dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                records = raw if isinstance(raw, list) else [raw]
                for record in records:
                    docs.append(CorpusDocument.model_validate(record))
        return cls(docs, embedder=embedder)
