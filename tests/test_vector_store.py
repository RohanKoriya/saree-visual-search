import os
import sys
import shutil

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.vector_store import VectorStore, ImageRecord

DIM = 8


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype("float32")
    return v / np.linalg.norm(v)


def _build_store(n: int = 5) -> VectorStore:
    store = VectorStore(dim=DIM)
    embeddings = np.stack([_unit_vector(i) for i in range(n)])
    records = [ImageRecord(id=f"id{i}", image_path=f"/tmp/{i}.jpg", image_url="") for i in range(n)]
    store.add(embeddings, records)
    return store


class TestVectorSearch:
    def test_query_returns_results(self):
        store = _build_store()
        query = _unit_vector(0)
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_results_contain_valid_metadata(self):
        store = _build_store()
        results = store.search(_unit_vector(1), top_k=2)
        for record, score in results:
            assert isinstance(record, ImageRecord)
            assert record.id.startswith("id")
            assert isinstance(score, float)

    def test_top_k_respected(self):
        store = _build_store(n=10)
        for k in [1, 3, 7]:
            results = store.search(_unit_vector(2), top_k=k)
            assert len(results) == k

    def test_top_k_larger_than_corpus_is_clamped(self):
        store = _build_store(n=3)
        results = store.search(_unit_vector(0), top_k=100)
        assert len(results) == 3

    def test_empty_store_returns_empty(self):
        store = VectorStore(dim=DIM)
        results = store.search(_unit_vector(0), top_k=5)
        assert results == []

    def test_self_query_is_top_result(self):
        store = _build_store()
        query = _unit_vector(3)
        results = store.search(query, top_k=1)
        assert results[0][0].id == "id3"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)


class TestRankingOrder:
    def test_scores_sorted_descending(self):
        store = _build_store(n=8)
        results = store.search(_unit_vector(0), top_k=8)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_embeddings_both_returned(self):
        store = VectorStore(dim=DIM)
        v = _unit_vector(42)
        embeddings = np.stack([v, v, _unit_vector(1)])
        records = [
            ImageRecord(id="dup_a", image_path="a", image_url=""),
            ImageRecord(id="dup_b", image_path="b", image_url=""),
            ImageRecord(id="other", image_path="c", image_url=""),
        ]
        store.add(embeddings, records)
        results = store.search(v, top_k=3)
        ids = {r.id for r, _ in results}
        assert "dup_a" in ids and "dup_b" in ids
        # both duplicates should score (near-)identically and rank above 'other'
        top_two_scores = [s for _, s in results[:2]]
        assert top_two_scores[0] == pytest.approx(top_two_scores[1], abs=1e-4)


class TestPersistence:
    def test_save_and_reload_roundtrip(self, tmp_path):
        store = _build_store(n=4)
        index_dir = str(tmp_path / "idx")
        store.save(index_dir)

        reloaded = VectorStore.load(index_dir)
        assert len(reloaded) == len(store)
        assert reloaded.dim == store.dim

        results_before = store.search(_unit_vector(0), top_k=2)
        results_after = reloaded.search(_unit_vector(0), top_k=2)
        assert [r.id for r, _ in results_before] == [r.id for r, _ in results_after]
