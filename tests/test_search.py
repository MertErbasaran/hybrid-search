"""Tests run offline: a tiny corpus and a fake embedder instead of a real model.

    pytest
"""

import zlib

import numpy as np
import pytest

from search.data import Document
from search.fusion import reciprocal_rank_fusion
from search.index import SearchIndex, _fts_query
from search.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims. Deterministic and good enough for tests."""

    synonyms = {"car": "vehicle", "automobile": "vehicle"}  # lets "vector" find what BM25 can't

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                word = self.synonyms.get(word.strip(".,"), word.strip(".,"))
                out[i, zlib.crc32(word.encode()) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


DOCS = [
    Document("d1", "Electric cars", "Battery electric car range and charging infrastructure."),
    Document("d2", "Vitamin D", "Vitamin D supplementation and bone fracture risk in older adults."),
    Document("d3", "Automobile safety", "Crash tests show modern automobile safety systems save lives."),
    Document("d4", "Sleep", "Sleep deprivation impairs memory consolidation."),
]


@pytest.fixture
def index(tmp_path):
    return SearchIndex.build(DOCS, FakeEmbedder(), tmp_path / "index")


# ---------- fusion ----------

def test_rrf_rewards_items_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    order = [item for item, _ in fused]
    assert order[0] == "b"  # in both lists beats rank 1 in only one list
    assert set(order) == {"a", "b", "c"}


def test_rrf_scores():
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["y"]], k=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] > fused["x"]


# ---------- metrics ----------

def test_metrics():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b": 1, "d": 1}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert mrr_at_k(ranked, relevant, 10) == 0.5
    assert ndcg_at_k(["b", "d"], relevant, 10) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(ranked, relevant, 10) < 1
    assert mrr_at_k(["x"], relevant, 10) == 0.0


# ---------- index ----------

def test_fts_query_is_safe():
    assert _fts_query('covid-19 (ACE2)? "x"') == '"covid" OR "19" OR "ace2" OR "x"'
    assert _fts_query("?!") is None


def test_bm25_finds_keyword_match(index):
    assert index.doc_ids(index.bm25_ranking("vitamin fractures", 3))[0] == "d2"


def test_bm25_handles_special_characters(index):
    index.search('vitamin-D (bone)? "', mode="bm25")  # must not raise


def test_vector_finds_synonym_that_bm25_misses(index):
    assert "d3" not in index.doc_ids(index.bm25_ranking("vehicle", 4))
    assert index.doc_ids(index.ranking("vehicle", "vector", 2))[0] in {"d1", "d3"}


def test_hybrid_contains_results_from_both(index):
    ids = index.doc_ids(index.ranking("vitamin vehicle", "hybrid", 4))
    assert "d2" in ids  # keyword hit
    assert {"d1", "d3"} & set(ids)  # semantic hit


def test_search_returns_hits_and_reload_works(index, tmp_path):
    hits = index.search("sleep memory", mode="hybrid", k=2)
    assert hits[0].doc_id == "d4" and hits[0].title == "Sleep"
    reloaded = SearchIndex.load(tmp_path / "index", FakeEmbedder())
    assert reloaded.search("sleep memory", k=1)[0].doc_id == "d4"


def test_unknown_mode_raises(index):
    with pytest.raises(ValueError):
        index.ranking("x", "magic", 3)
