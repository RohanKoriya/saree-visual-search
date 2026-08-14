"""
SearchEngine: orchestrates the full retrieve-then-rerank pipeline
(Phase 6/8). This is the single place that ties together the embedding
model, the FAISS vector store, and the reranker. Both the Streamlit UI
and the agent tool call into this -- neither talks to FAISS or the model
directly.

    query image
        -> embed (full, pallu, border) + color histogram
        -> FAISS top RECALL_K candidates (full-image cosine similarity)
        -> reranker combines signals -> top_k
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.embeddings.model import ClipEmbedder, ClipModelConfig
from src.search.reranker import Reranker, RerankWeights, ScoredResult
from src.search.vector_store import VectorStore
from src.utils.image_utils import compute_color_histogram, get_region_crops

DEFAULT_RECALL_K = 25  # candidates pulled from FAISS before reranking


@dataclass
class SearchEngineConfig:
    index_dir: str = "index"
    clip_local_checkpoint: str | None = None
    recall_k: int = DEFAULT_RECALL_K
    rerank_weights: RerankWeights | None = None


class SearchEngine:
    def __init__(self, config: SearchEngineConfig | None = None):
        self.config = config or SearchEngineConfig()

        clip_config = (
            ClipModelConfig(local_checkpoint=self.config.clip_local_checkpoint)
            if self.config.clip_local_checkpoint
            else ClipModelConfig()
        )
        self.embedder = ClipEmbedder(clip_config)
        self.store = VectorStore.load(self.config.index_dir)
        self.reranker = Reranker(self.config.rerank_weights)

    def search(self, image: Image.Image, top_k: int = 5) -> list[ScoredResult]:
        if len(self.store) == 0:
            return []

        img = image.convert("RGB")
        crops = get_region_crops(img)

        full_emb = self.embedder.embed_image(crops["full"])
        pallu_emb = self.embedder.embed_image(crops["pallu"]).tolist()
        border_emb = self.embedder.embed_image(crops["border"]).tolist()
        color_hist = compute_color_histogram(img)

        recall_k = min(self.config.recall_k, len(self.store))
        candidates = self.store.search(full_emb, top_k=recall_k)

        return self.reranker.rerank(
            query_pallu_embedding=pallu_emb,
            query_border_embedding=border_emb,
            query_color_histogram=color_hist,
            candidates=candidates,
            top_k=top_k,
        )

    @property
    def catalog_size(self) -> int:
        return len(self.store)
