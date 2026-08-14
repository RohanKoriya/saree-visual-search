"""
FAISS-backed vector store for saree image embeddings.

Design notes (Phase 3):
- Embeddings are L2-normalized (see src/embeddings/model.py), so we use
  IndexFlatIP (inner product). For normalized vectors, inner product is
  mathematically equivalent to cosine similarity, giving stable,
  interpretable scores in [-1, 1] (in practice ~[0, 1] for CLIP image pairs).
- IndexFlatIP does exact (brute-force) search. For a catalog of ~1000
  sarees this is fast (<10ms) and simpler/more reliable than an
  approximate index (e.g. IVF/HNSW) which would need tuning and only
  pays off at much larger scale. We keep this configurable in case the
  catalog grows.
- The index and a parallel metadata list are persisted to disk together
  (index/index.faiss + index/metadata.json) so the app never has to
  regenerate embeddings on startup -- it just loads both files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import faiss
import numpy as np


@dataclass
class ImageRecord:
    """Metadata for a single indexed image, aligned by position with the FAISS index.

    `pallu_embedding`, `border_embedding`, and `color_histogram` are optional
    signals used by the reranker (src/search/reranker.py) for fine-grained
    similarity beyond the whole-image CLIP embedding. They are computed once
    at index-build time (scripts/build_index.py) and stored here so query-time
    reranking never has to re-embed the full candidate set.
    """

    id: str
    image_path: str
    image_url: str
    name: str | None = None
    sku: str | None = None
    width: int | None = None
    height: int | None = None
    dominant_colors: list[str] | None = None
    pallu_embedding: list[float] | None = None
    border_embedding: list[float] | None = None
    color_histogram: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(**d)


class VectorStore:
    """Wraps a FAISS IndexFlatIP plus aligned metadata, with save/load."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.records: list[ImageRecord] = []

    def add(self, embeddings: np.ndarray, records: list[ImageRecord]) -> None:
        if embeddings.shape[0] != len(records):
            raise ValueError("Number of embeddings must match number of records")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {embeddings.shape[1]}")
        # Defensive re-normalization in case caller passed non-normalized vectors.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (embeddings / norms).astype("float32")
        self.index.add(normalized)
        self.records.extend(records)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[ImageRecord, float]]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        metadata = {
            "dim": self.dim,
            "records": [r.to_dict() for r in self.records],
        }
        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "VectorStore":
        with open(os.path.join(index_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        store = cls(dim=metadata["dim"])
        store.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        store.records = [ImageRecord.from_dict(r) for r in metadata["records"]]
        return store

    def __len__(self) -> int:
        return len(self.records)
