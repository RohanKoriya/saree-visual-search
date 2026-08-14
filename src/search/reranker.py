"""
Reranker: candidate retrieval + fine-grained rescoring.

Pipeline (Phase 6):

    query image
        -> full-image embedding
        -> FAISS retrieves top ~20-30 candidates (recall stage, vector_store.py)
        -> for each candidate, combine multiple similarity signals:
             - full-image CLIP embedding similarity   (weight: w_embedding)
             - pallu-region CLIP embedding similarity  (weight: w_pallu)
             - border-region CLIP embedding similarity (weight: w_border)
             - saturation-weighted color histogram sim  (weight: w_color)
        -> weighted sum -> final_score
        -> re-sort, return top_k

This is deliberately a small, fixed, explainable formula rather than a
learned reranker -- the assignment asks for something explainable and
deterministic, not maximal sophistication.
Weight tuning note:
--------------------
These weights were chosen based on a real evaluation (scripts/evaluate.py)
against 5444 query-item pairs across 198 design families in the full
catalog, measuring mean rank of same-design color variants (lower=better):

    baseline (embedding-only):    119.10
    color_only:                   211.86  (hurts badly -- dropped)
    pallu_only:                   113.37
    border_only:                  112.31
    region_only (pallu+border):   110.18  <- best, used below

Color histogram similarity measurably hurt ranking at scale, likely
because histogram intersection doesn't account for hue being circular
(near-red hues at opposite ends of the 0-255 bin range score as
dissimilar despite being visually close). It's kept in the code
(weight=0, score still computed and returned in ScoredResult for
transparency) rather than deleted, in case it's worth fixing and
re-testing later -- but should not be re-enabled without re-evaluating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.search.vector_store import ImageRecord
from src.utils.image_utils import histogram_similarity


@dataclass(frozen=True)
class RerankWeights:
    embedding: float = 0.60
    pallu: float = 0.20
    border: float = 0.20
    color: float = 0.0

    def normalized(self) -> "RerankWeights":
        total = self.embedding + self.pallu + self.border + self.color
        if total <= 0:
            raise ValueError("Reranker weights must sum to a positive number")
        return RerankWeights(
            embedding=self.embedding / total,
            pallu=self.pallu / total,
            border=self.border / total,
            color=self.color / total,
        )


@dataclass
class ScoredResult:
    record: ImageRecord
    final_score: float
    rank: int
    # Breakdown kept for transparency / debugging / the evaluation notebook.
    embedding_score: float
    pallu_score: float | None
    border_score: float | None
    color_score: float | None


def _cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


class Reranker:
    def __init__(self, weights: RerankWeights | None = None):
        self.weights = (weights or RerankWeights()).normalized()

    def rerank(
        self,
        query_pallu_embedding: list[float] | None,
        query_border_embedding: list[float] | None,
        query_color_histogram: np.ndarray | None,
        candidates: list[tuple[ImageRecord, float]],
        top_k: int,
    ) -> list[ScoredResult]:
        """
        candidates: list of (ImageRecord, embedding_similarity) from the FAISS
        recall stage (already full-image cosine similarity in [~0,1]).
        Returns the top_k candidates re-sorted by weighted final_score.
        """
        w = self.weights
        scored: list[ScoredResult] = []

        for record, embedding_score in candidates:
            pallu_score = _cosine(query_pallu_embedding, record.pallu_embedding)
            border_score = _cosine(query_border_embedding, record.border_embedding)

            color_score = None
            if query_color_histogram is not None and record.color_histogram is not None:
                color_score = histogram_similarity(
                    query_color_histogram, np.asarray(record.color_histogram)
                )

            # If a signal is unavailable for this record (e.g. metadata was
            # built before that signal existed), fall back to redistributing
            # its weight onto the embedding score rather than silently
            # treating the missing signal as a zero (which would unfairly
            # punish otherwise-good candidates).
            parts = [(w.embedding, embedding_score)]
            if pallu_score is not None:
                parts.append((w.pallu, pallu_score))
            if border_score is not None:
                parts.append((w.border, border_score))
            if color_score is not None:
                parts.append((w.color, color_score))

            weight_sum = sum(p[0] for p in parts)
            final_score = sum(p[0] * p[1] for p in parts) / weight_sum if weight_sum > 0 else embedding_score

            scored.append(
                ScoredResult(
                    record=record,
                    final_score=final_score,
                    rank=0,  # assigned after sort
                    embedding_score=embedding_score,
                    pallu_score=pallu_score,
                    border_score=border_score,
                    color_score=color_score,
                )
            )

        scored.sort(key=lambda s: s.final_score, reverse=True)
        scored = scored[:top_k]
        for i, s in enumerate(scored, 1):
            s.rank = i
        return scored
