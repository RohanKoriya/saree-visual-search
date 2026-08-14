"""
Streamlit chat UI for the saree visual similarity search agent
(Phase 10 / assignment sections 1 "Frontend" and 8 "Search Results UI").

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from src.agent.agent import build_agent, run_turn
from src.search.engine import SearchEngine, SearchEngineConfig
from src.tools.visual_search import set_search_engine
from src.utils.query_image import InvalidImageError, load_image_from_url

st.set_page_config(page_title="Saree Visual Search", page_icon="🧵", layout="wide")

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
CLIP_LOCAL_CHECKPOINT = os.environ.get("CLIP_LOCAL_CHECKPOINT")  # optional, sandbox-only escape hatch

CARDS_PER_ROW = 5


# ---------------------------------------------------------------------------
# Minimal CSS -- only what native Streamlit components can't do on their own
# (text truncation on long names/SKUs, the match-score badge, and a couple
# of spacing tweaks). No gradients, no glassmorphism, no theme overrides.
# ---------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        .app-subtitle {
            color: rgba(250, 250, 250, 0.6);
            font-size: 0.95rem;
            margin-top: -0.6rem;
            margin-bottom: 1.25rem;
        }

        .card-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 0.5rem;
        }
        .card-rank {
            font-weight: 600;
            font-size: 0.85rem;
            color: rgba(250, 250, 250, 0.85);
        }
        .card-match {
            font-size: 0.78rem;
            font-weight: 600;
            color: #4ade80;
            background: rgba(74, 222, 128, 0.12);
            padding: 2px 9px;
            border-radius: 999px;
            white-space: nowrap;
        }
        .card-name {
            font-weight: 600;
            font-size: 0.92rem;
            line-height: 1.3rem;
            margin: 0.5rem 0 0.15rem 0;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 2.6rem;
        }
        .card-sku {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.76rem;
            color: rgba(250, 250, 250, 0.45);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin: 0;
        }

        .empty-state {
            border: 1px dashed rgba(250, 250, 250, 0.18);
            border-radius: 10px;
            padding: 2rem 1.75rem;
            margin-top: 0.5rem;
        }
        .empty-state h4 {
            margin-top: 0;
            margin-bottom: 0.75rem;
        }
        .empty-state ol {
            margin: 0;
            padding-left: 1.2rem;
            color: rgba(250, 250, 250, 0.75);
        }
        .empty-state li { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached resource loading -- the model and FAISS index are loaded ONCE per
# server process, never per request (assignment section 9, "Caching and
# Performance"). Unchanged from the original -- presentation-only edits
# don't touch this.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and search index...")
def get_search_engine() -> SearchEngine | None:
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return None
    engine = SearchEngine(
        SearchEngineConfig(index_dir=INDEX_DIR, clip_local_checkpoint=CLIP_LOCAL_CHECKPOINT)
    )
    set_search_engine(engine)
    return engine


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown("### 🧵 Saree Visual Search")
    st.markdown(
        '<div class="app-subtitle">Upload a saree photo or paste an image URL to find '
        "visually similar sarees from the catalog.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(engine: SearchEngine) -> tuple[str | None, object | None, int]:
    """Renders the sidebar and returns (query_image_path, query_image_preview, top_k).

    Image acquisition logic (upload / URL -> temp file) is unchanged from the
    original -- only the surrounding layout/labels are reorganized.
    """
    st.sidebar.metric("Sarees indexed", engine.catalog_size)
    top_k = st.sidebar.slider("Results to show", min_value=1, max_value=10, value=5)

    st.sidebar.divider()
    st.sidebar.subheader("Query image")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a saree image", type=["jpg", "jpeg", "png", "webp"]
    )
    image_url = st.sidebar.text_input("...or paste an image URL")

    query_image_path = None
    query_image_preview = None

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getvalue())
        tmp.close()
        query_image_path = tmp.name
        query_image_preview = uploaded_file.getvalue()
    elif image_url:
        try:
            img = load_image_from_url(image_url)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(tmp.name, format="JPEG")
            tmp.close()
            query_image_path = tmp.name
            query_image_preview = img
        except InvalidImageError as e:
            st.sidebar.error(str(e))

    if query_image_preview is not None:
        with st.sidebar.container(border=True):
            st.image(query_image_preview, caption="Query image", use_container_width=True)

    return query_image_path, query_image_preview, top_k


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
        <h4>Get started</h4>
        <ol>
            <li>Upload a saree image (or paste an image URL) in the sidebar</li>
            <li>Choose how many results you'd like to see</li>
            <li>Ask to find visually similar sarees</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_card(result: dict) -> None:
    with st.container(border=True):
        image_ref = result.get("image")
        try:
            st.image(image_ref, use_container_width=True)
        except Exception:
            st.warning("Image preview unavailable")

        name = result.get("name") or "Unnamed saree"
        sku = result.get("sku")

        st.markdown(
            f"""
            <div class="card-meta">
                <span class="card-rank">#{result['rank']}</span>
                <span class="card-match">{result['score'] * 100:.1f}% match</span>
            </div>
            <div class="card-name" title="{name}">{name}</div>
            {f'<p class="card-sku" title="{sku}">SKU: {sku}</p>' if sku else ""}
            """,
            unsafe_allow_html=True,
        )


def render_results_grid(results: list[dict]) -> None:
    count = len(results)
    st.caption(f"Found {count} visually similar saree{'s' if count != 1 else ''}.")

    for start in range(0, count, CARDS_PER_ROW):
        row = results[start : start + CARDS_PER_ROW]
        cols = st.columns(len(row), gap="medium")
        for col, r in zip(cols, row):
            with col:
                render_product_card(r)


def render_chat_message(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])
        if entry.get("results"):
            render_results_grid(entry["results"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_styles()
    render_header()

    engine = get_search_engine()
    if engine is None:
        st.error(
            f"No search index found at `{INDEX_DIR}/`. Run "
            "`python scripts/build_index.py --csv <your_catalog.csv>` first, "
            "then restart the app."
        )
        st.stop()

    if not os.environ.get("LLM_API_KEY"):
        st.error(
            "LLM_API_KEY is not set. Copy `.env.example` to `.env`, add your Gemini "
            "API key, and restart the app."
        )
        st.stop()

    try:
        agent = get_agent()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of LangChain BaseMessage -- agent memory, untouched
    if "display_log" not in st.session_state:
        st.session_state.display_log = []  # list of dicts for rendering only

    query_image_path, query_image_preview, top_k = render_sidebar(engine)

    if not st.session_state.display_log and not query_image_path:
        render_empty_state()

    # -- Replay chat history --------------------------------------------------
    for entry in st.session_state.display_log:
        render_chat_message(entry)

    # -- Chat input --------------------------------------------------------
    user_text = st.chat_input("Ask me to find similar sarees, or just say hello...")
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
            if query_image_path:
                st.image(query_image_preview, width=200)
        st.session_state.display_log.append(
            {"role": "user", "text": user_text, "results": None}
        )

        with st.chat_message("assistant"):
            prev_len = len(st.session_state.chat_history)  # <-- mark where THIS turn starts
            with st.spinner("Thinking..."):
                try:
                    reply, new_messages = run_turn(
                        agent, st.session_state.chat_history, user_text, query_image_path
                    )
                except Exception as e:  # noqa: BLE001 -- surface as a friendly chat error, don't crash the app
                    error_message = str(e)
                
                    if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                        reply = "The AI chat is temporarily unavailable because the Gemini API quota has been reached. Please try again later."
                    else:
                        reply = "Sorry, something went wrong. Please try again."
                
                    new_messages = st.session_state.chat_history

            st.session_state.chat_history = new_messages

            # Only look for a tool call made in THIS turn -- new_messages
            # includes the full accumulated history, so scanning all of it
            # would re-surface a previous turn's search results even when
            # no new search happened (e.g. the user just said "hello").
            turn_messages = new_messages[prev_len:]
            results = None
            for m in turn_messages:
                if getattr(m, "name", None) == "search_similar_sarees" and hasattr(m, "artifact"):
                    artifact = m.artifact
                    if artifact and artifact.get("status") == "ok":
                        results = artifact["results"]

            # Presentation choice only: when a search produced results, show
            # a short summary instead of the agent's full itemized text reply
            # (the cards below already show every result in detail -- no
            # need to repeat it as a text list too). Non-search replies
            # (greetings, errors, capability questions) are shown as-is.
            if results:
                count = len(results)
                display_text = f"Found {count} visually similar saree{'s' if count != 1 else ''}."
            else:
                display_text = reply

            st.markdown(display_text)
            if results:
                render_results_grid(results)

        st.session_state.display_log.append(
            {"role": "assistant", "text": display_text, "results": results}
        )


if __name__ == "__main__":
    main()