"""
Agent-callable visual search tool (Phase 8 / assignment section 5).

This is intentionally a thin, typed wrapper around SearchEngine. The LLM
never performs the vector search itself -- it only decides *when* to call
this tool and with what arguments; all actual retrieval/reranking is
deterministic code in src/search/.

Design choice on `image_source`: the assignment's example schema takes a
single `image_path`. We generalize it to `image_source`, which accepts
EITHER a local file path (used when the Streamlit app has already saved
an uploaded file to disk) OR an http(s) URL (used when the user pastes an
image link). This keeps the tool's surface small -- one argument -- while
covering both input modes from section 7 of the assignment. The tool
itself does not fetch arbitrary user-typed URLs blindly: it reuses the
same validation path as the rest of the app (src/utils/query_image.py),
which checks URL scheme, content-type, and size before ever handing bytes
to the model.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.search.engine import SearchEngine
from src.utils.query_image import (
    InvalidImageError,
    load_image_from_bytes,
    load_image_from_url,
)

_engine: SearchEngine | None = None


def set_search_engine(engine: SearchEngine) -> None:
    """Injected once at app startup (see app.py) so the tool reuses the
    already-loaded model/index instead of constructing its own."""
    global _engine
    _engine = engine


class SareeMatch(BaseModel):
    image: str = Field(description="Local path or URL of the matched saree image.")
    name: str | None = Field(default=None, description="Product name, if known.")
    sku: str | None = Field(default=None, description="Product SKU, if known.")
    score: float = Field(description="Similarity score in [0, 1], higher is more similar.")
    rank: int = Field(description="1-indexed rank among the returned results.")


class SearchSimilarSareesResult(BaseModel):
    status: Literal["ok", "error"]
    message: str | None = Field(
        default=None, description="Human-readable error message when status == 'error'."
    )
    results: list[SareeMatch] = Field(default_factory=list)


@tool(response_format="content_and_artifact")
def search_similar_sarees(image_source: str, top_k: int = 5) -> tuple[str, dict]:
    """Search the saree catalog and return the most visually similar sarees.

    Call this tool when the user provides or references an image (an
    uploaded image or an image URL) and asks to find similar, matching,
    or comparable sarees. Do NOT call this tool for general conversation,
    questions about what the app can do, or requests that don't involve
    an actual image.

    Args:
        image_source: Either a local file path to the query image (e.g. an
            uploaded file already saved to disk) or an http(s) URL pointing
            directly to an image.
        top_k: Number of similar sarees to return (default 5, max 20).

    Returns:
        A structured result containing, for each match: image reference,
        product name/SKU if known, a similarity score in [0, 1], and its
        rank. On failure, returns a status of "error" with a human-readable
        message instead of raising.
    """
    top_k = max(1, min(top_k, 20))

    if _engine is None:
        result = SearchSimilarSareesResult(
            status="error", message="Search index is not loaded yet. Please try again shortly."
        )
        return result.model_dump_json(), result.model_dump()

    try:
        if image_source.startswith("http://") or image_source.startswith("https://"):
            image = load_image_from_url(image_source)
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                image = load_image_from_bytes(f.read())
        else:
            result = SearchSimilarSareesResult(
                status="error",
                message=f"Could not find an image at '{image_source}'. "
                "Please upload an image or provide a direct image URL.",
            )
            return result.model_dump_json(), result.model_dump()

        scored = _engine.search(image, top_k=top_k)

    except InvalidImageError as e:
        result = SearchSimilarSareesResult(status="error", message=str(e))
        return result.model_dump_json(), result.model_dump()
    except Exception as e:  # noqa: BLE001 -- never let a tool call crash the agent loop
        result = SearchSimilarSareesResult(
            status="error", message=f"Search failed unexpectedly: {e}"
        )
        return result.model_dump_json(), result.model_dump()

    matches = [
        SareeMatch(
            image=s.record.image_url or s.record.image_path,
            name=s.record.name,
            sku=s.record.sku,
            score=round(s.final_score, 4),
            rank=s.rank,
        )
        for s in scored
    ]
    result = SearchSimilarSareesResult(status="ok", results=matches)
    return result.model_dump_json(), result.model_dump()
