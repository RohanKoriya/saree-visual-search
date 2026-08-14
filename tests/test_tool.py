import os
import sys
import io

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.reranker import ScoredResult
from src.search.vector_store import ImageRecord
from src.tools import visual_search as vs


class StubEngine:
    """A fake SearchEngine so tool tests don't need to load the real CLIP model."""

    def __init__(self, results):
        self._results = results

    def search(self, image, top_k=5):
        return self._results[:top_k]


def _stub_result(rank, sku, score):
    record = ImageRecord(id=sku, image_path=f"/tmp/{sku}.jpg", image_url=f"https://example.com/{sku}.jpg", name=f"Saree {sku}", sku=sku)
    return ScoredResult(
        record=record, final_score=score, rank=rank,
        embedding_score=score, pallu_score=score, border_score=score, color_score=score,
    )


def _write_temp_image(tmp_path) -> str:
    img = Image.new("RGB", (16, 16), color=(10, 200, 10))
    path = str(tmp_path / "query.jpg")
    img.save(path, format="JPEG")
    return path


def _invoke(image_source, top_k=5):
    tool_call = {
        "name": "search_similar_sarees",
        "args": {"image_source": image_source, "top_k": top_k},
        "id": "1",
        "type": "tool_call",
    }
    msg = vs.search_similar_sarees.invoke(tool_call)
    return msg.artifact


class TestValidInput:
    def test_local_path_returns_ok_status(self, tmp_path):
        vs.set_search_engine(StubEngine([_stub_result(1, "SKU1", 0.9), _stub_result(2, "SKU2", 0.8)]))
        image_path = _write_temp_image(tmp_path)
        artifact = _invoke(image_path, top_k=2)
        assert artifact["status"] == "ok"
        assert len(artifact["results"]) == 2

    def test_top_k_is_respected(self, tmp_path):
        results = [_stub_result(i, f"SKU{i}", 1.0 - i * 0.1) for i in range(1, 6)]
        vs.set_search_engine(StubEngine(results))
        image_path = _write_temp_image(tmp_path)
        artifact = _invoke(image_path, top_k=3)
        assert len(artifact["results"]) == 3

    def test_top_k_clamped_to_max_20(self, tmp_path):
        results = [_stub_result(i, f"SKU{i}", 1.0) for i in range(1, 4)]
        vs.set_search_engine(StubEngine(results))
        image_path = _write_temp_image(tmp_path)
        # requesting an absurd top_k should not error
        artifact = _invoke(image_path, top_k=999)
        assert artifact["status"] == "ok"


class TestInvalidInput:
    def test_nonexistent_path_returns_error_status(self):
        vs.set_search_engine(StubEngine([]))
        artifact = _invoke("/no/such/file/exists.jpg")
        assert artifact["status"] == "error"
        assert artifact["results"] == []
        assert "message" in artifact and artifact["message"]

    def test_garbage_string_returns_error_not_exception(self):
        vs.set_search_engine(StubEngine([]))
        artifact = _invoke("definitely not a path or url")
        assert artifact["status"] == "error"

    def test_no_engine_loaded_returns_error(self, tmp_path):
        vs.set_search_engine(None)
        image_path = _write_temp_image(tmp_path)
        artifact = _invoke(image_path)
        assert artifact["status"] == "error"
        assert "index" in artifact["message"].lower()


class TestStructuredOutput:
    def test_result_fields_present(self, tmp_path):
        vs.set_search_engine(StubEngine([_stub_result(1, "SKU1", 0.876543)]))
        image_path = _write_temp_image(tmp_path)
        artifact = _invoke(image_path, top_k=1)
        match = artifact["results"][0]
        assert set(["image", "name", "sku", "score", "rank"]).issubset(match.keys())
        assert match["rank"] == 1
        assert match["sku"] == "SKU1"
        assert 0.0 <= match["score"] <= 1.0
