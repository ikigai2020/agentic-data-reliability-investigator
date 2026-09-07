"""Unit tests for the deterministic embedder, FAISS index, and corpus search."""

from __future__ import annotations

import numpy as np

from investigator.retrieval.corpus import Corpus
from investigator.retrieval.embedder import DeterministicEmbedder
from investigator.retrieval.index import VectorIndex

from .fixtures import eval_items


def test_embedder_is_deterministic_and_normalized() -> None:
    emb = DeterministicEmbedder(dim=256)
    v1 = emb.embed("orders_daily volume drop orders_fact")
    v2 = emb.embed("orders_daily volume drop orders_fact")
    assert np.array_equal(v1, v2)  # reproducible
    assert v1.shape == (256,)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5  # L2-normalized


def test_similar_text_scores_higher_than_unrelated() -> None:
    emb = DeterministicEmbedder()
    q = emb.embed("orders_daily volume drop orders_fact transformation filter")
    near = emb.embed("orders_daily volume drop orders_fact filter regression")
    far = emb.embed("inventory_hourly null spike stock levels")
    assert float(q @ near) > float(q @ far)


def test_vector_index_returns_self_first() -> None:
    emb = DeterministicEmbedder()
    idx = VectorIndex(dim=emb.dim)
    texts = {"a": "orders volume drop", "b": "payments orchestration failure"}
    idx.add(list(texts), emb.embed_many(list(texts.values())))
    hits = idx.search(emb.embed(texts["a"]), top_k=2)
    assert hits[0][0] == "a"
    assert idx.backend in {"faiss", "numpy"}


def test_corpus_search_ranks_relevant_first() -> None:
    corpus = Corpus([i.document for i in eval_items()])
    results = corpus.search("volume_drop orders_fact orders_daily transformation_logic", top_k=3)
    top_ids = [r.document.doc_id for r in results]
    assert top_ids[0] in {"REL-1", "REL-2", "OUTDATED-1"}  # an orders_daily doc leads
    assert all(0.0 <= r.score <= 1.0 for r in results)
