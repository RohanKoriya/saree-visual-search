# Saree Visual Similarity Search

A chat agent that finds visually similar sarees from a product catalog. Upload
a photo (or paste an image URL), describe what you want in plain language, and
the agent finds the closest matches using image embeddings, FAISS vector
search, and a multi-signal reranker — with the actual retrieval done by
deterministic code, not the LLM.

---

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Setup](#3-setup)
4. [Technology Choices](#4-technology-choices)
5. [Search Quality Improvements](#5-search-quality-improvements)
6. [Search Quality Evaluation](#6-search-quality-evaluation)
7. [Assumptions](#7-assumptions)
8. [Trade-offs](#8-trade-offs)
9. [Bugs Found & Fixed During Development](#9-bugs-found--fixed-during-development)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Project Structure](#12-project-structure)

---

## 1. Overview

The catalog is a real saree retailer's product feed (Byrappa Silks,
Bangalore): a CSV of `Name, SKU, Stock, Retail Price, Discounted Price,
image_url, Website Link` — 1074 rows across 655 unique SKUs (many SKUs have
multiple product photos). Every product is a saree, so naive whole-image
embedding search tends to return loose, generic matches — this is measured
directly in [Section 6](#6-search-quality-evaluation), not just assumed.

The user interacts through a Streamlit chat interface. An LLM agent (Gemini,
via LangChain/LangGraph) decides when the user is asking for a visual search
versus just chatting, and calls a single typed tool — `search_similar_sarees`
— to perform the actual search. The LLM never computes similarity itself.

## 2. Architecture

```
User
 ↓
Streamlit (app.py) -- chat UI, image upload / URL input, product-card grid
 ↓
LLM Agent (src/agent/agent.py) -- Gemini + LangGraph ReAct loop
 ↓  (decides WHEN to call the tool; never computes similarity itself)
Visual Search Tool (src/tools/visual_search.py) -- typed input/output, error handling
 ↓
SearchEngine (src/search/engine.py) -- orchestrates the pipeline below
 ↓
Image Embedding Model (src/embeddings/model.py) -- OpenCLIP ViT-B-32
 ↓
Vector Index (src/search/vector_store.py) -- FAISS IndexFlatIP (cosine via normalized vectors)
 ↓  (top ~25 candidates)
Reranker (src/search/reranker.py) -- embedding + pallu-region + border-region similarity
 ↓
Results -- ranked list with scores, shown as product cards
```

Offline indexing (`scripts/build_index.py`) is a completely separate path
from runtime search: it downloads/validates images, computes all embeddings
once, and persists everything to `index/`. The Streamlit app and the agent
tool only ever **load** that index — they never regenerate embeddings for
the catalog at request time.

## 3. Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt   # see CPU-only torch note below

cp .env.example .env
# edit .env: set LLM_API_KEY to a free-tier Gemini key from https://ai.google.dev/

# Build the index (downloads images referenced in the CSV, embeds, indexes)
python scripts/build_index.py --csv data/byrappa_tejas_31july.csv \
    --images-dir data/images --index-dir index

streamlit run app.py
```

Re-running `build_index.py` is safe and incremental: already-downloaded
images aren't re-downloaded. Pass `--limit N` while developing to only
process the first N rows.

**CPU-only torch:** the default `pip install torch` can pull a large CUDA
build. For free-tier / CPU-only deployment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Running the evaluation and tests

```bash
python scripts/evaluate.py --index-dir index   # search quality, see Section 6
python -m pytest tests/ -v                      # correctness tests, see Section 10
```

## 4. Technology Choices

| Component       | Choice                                                               | Why                                                                                                                                                                                                                                                                                                                                                      |
| --------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedding model | OpenCLIP `ViT-B-32`, `laion2b_e16` weights                           | Free, pretrained (not trained from scratch), strong general visual-semantic embeddings, small enough (~600MB, 512-dim) to run on CPU within free-tier constraints. Not fine-tuned on sarees specifically — see [Trade-offs](#8-trade-offs).                                                                                                              |
| Vector index    | FAISS `IndexFlatIP`                                                  | Exact (brute-force) inner-product search. At ~1000 vectors this is fast (single-digit ms) and simpler/more reliable than an approximate index (IVF/HNSW), which would need tuning that isn't justified at this scale. Cosine similarity is achieved by L2-normalizing every embedding before insertion/query, making inner product == cosine similarity. |
| Agent framework | LangChain + LangGraph (`create_react_agent`)                         | Standard, well-supported tool-calling loop; keeps the LLM's role limited to _deciding when_ to call the search tool, per the explicit separation between "decide when to search" (LLM) and "perform the search" (deterministic code).                                                                                                                    |
| LLM             | Gemini (`gemini-3.6-flash` by default), via `langchain-google-genai` | Usable free tier; provider is fully configurable via `LLM_API_KEY` / `LLM_MODEL` env vars, no keys hardcoded anywhere.                                                                                                                                                                                                                                   |
| Frontend        | Streamlit                                                            | Chat interface with a product-card results grid; `st.cache_resource` used for the model and index so they load once per server process, not per request.                                                                                                                                                                                                 |

## 5. Search Quality Improvements

Beyond `image → embedding → top 5`, the pipeline adds:

1. **Normalized embeddings + cosine similarity (via inner product).** Every
   embedding (full image, pallu crop, border crop) is L2-normalized at
   creation time, so FAISS's `IndexFlatIP` and the reranker's cosine
   calculations are both stable and directly comparable.

2. **Candidate retrieval + reranking.** FAISS returns the top ~25 candidates
   by full-image embedding similarity; the reranker then rescoring using
   multiple signals before returning the final top-k. This two-stage design
   means the more expensive scoring only runs on a short candidate list, not
   the whole catalog.

3. **Region crops (pallu / border).** The catalog's product photography is
   consistent: a saree draped on a mannequin, plain grey background, pallu
   hanging down the right side, border visible as a diagonal sash. Based on
   that consistent composition, `src/utils/image_utils.py` defines two fixed
   fractional crops approximating the pallu and border regions, and separate
   CLIP embeddings are computed for each at index time. **This measurably
   improved ranking quality at full scale** — see Section 6.

4. **Saturation-weighted color histogram — tried, and dropped.** A color
   histogram over the whole image, weighted by each pixel's HSV saturation
   so the plain grey/white studio background contributes ~nothing, was also
   implemented and tested. It **measurably hurt** ranking quality at full
   scale (Section 6) and is kept in the code at weight `0.0` rather than
   deleted, since the underlying idea may be salvageable with a fix — see
   the note in Section 6.

5. **Weighted, explainable final score:**
   ```
   final_score = w_embedding * embedding_similarity
               + w_pallu     * pallu_region_similarity
               + w_border    * border_region_similarity
               + w_color     * color_histogram_similarity
   ```
   Current weights (`src/search/reranker.py::RerankWeights`), **chosen from
   a real evaluation run, not guessed:**
   ```
   embedding = 0.60
   pallu     = 0.20
   border    = 0.20
   color     = 0.00
   ```

## 6. Search Quality Evaluation

`scripts/evaluate.py` uses real ground truth already present in the catalog:
products whose SKU shares a numeric prefix are color variants of the same
underlying design (e.g. `AA313400`/`AA313402`/`AA313403` are all "Munga Crape
Sarees" in different colors, confirmed via the `Name` field — this is
observed structure in the data, not an invented metric). For each query
image, it checks where that image's same-design variants land in the ranked
results and reports the **mean rank** (lower = better; 1.0 is ideal).

### Full-catalog result

Run against the fully-built index: **956 images across 198 design families**
with ≥2 color variants, **5444 total query-item pairs** — large enough for
the result to be meaningful, not anecdotal.

| Configuration                |  Mean rank | vs. baseline                         |
| ---------------------------- | ---------: | ------------------------------------ |
| Baseline (embedding only)    |     119.10 | —                                    |
| + color histogram only       |     211.86 | **worse** (dropped)                  |
| + pallu region only          |     113.37 | better                               |
| + border region only         |     112.31 | better                               |
| **+ pallu + border (final)** | **110.18** | **best, ~7.5% better than baseline** |

**What this means:** the region-crop hypothesis (pallu/border) held up under
real, large-scale testing and was kept. The color histogram signal did not —
it made rankings measurably worse, likely because histogram intersection on
a linear 0–255 hue axis doesn't account for hue being circular (near-red
hues sitting at opposite ends of the bin range score as dissimilar despite
being visually close), and/or because its score distribution has higher
variance than the embedding score, disproportionately swinging rankings
despite a modest nominal weight. It was **dropped** (weight set to `0.0`,
code kept for a future fix + re-test) rather than kept for the sake of
"more signals," per the principle that added complexity should only stay if
it demonstrably helps.

### Manual spot-checks

Beyond the automated metric, results were manually inspected across several
diverse query images (uploaded photos, not catalog images) during
development. Observed pattern: results are consistently on-theme (correct
general saree style/weight/drape) and mostly color-consistent, with
occasional color misses (e.g. one blue result appearing among mostly-white
matches) — an expected, honestly-acknowledged consequence of the color
signal being disabled per the measured result above, not a bug.

### What would improve this further

- Fix the color histogram's hue-circularity issue (e.g. wrap-around binning
  or a perceptual color space) and re-run the ablation — it may become a net
  positive once fixed, given color is intuitively a strong signal for this
  domain.
- Try a larger CLIP backbone (`ViT-L-14`) if compute budget allows, and
  re-run this same evaluation to see whether it changes which signals help.
- Layer in a genuine garment segmentation model instead of fixed-fraction
  crops, if further gains are needed and the added dependency/latency is
  acceptable.

## 7. Assumptions

- The CSV's `SKU` is a stable identifier for a _design_; a single SKU may
  have multiple photo rows (1074 rows / 655 unique SKUs), which the indexer
  now treats as distinct image records with disambiguated filenames (see
  [Section 9](#9-bugs-found--fixed-during-development)).
- Product photography follows one consistent layout (mannequin, plain
  background, pallu draped right, diagonal border sash) closely enough that
  fixed fractional crops are a reasonable proxy for "pallu region" /
  "border region" without object detection or segmentation. This was tested,
  not just assumed — see Section 6.
- `Name` and `SKU` are the only trustworthy metadata fields; no color,
  fabric, or pattern labels are invented beyond what's literally present in
  those fields (e.g. `sku_family` in `scripts/evaluate.py` is derived
  directly from the SKU numbering pattern, not guessed).

## 8. Trade-offs

- **Embedding model size vs. accuracy.** `ViT-B-32` was chosen over larger
  OpenCLIP variants (e.g. `ViT-L-14`) specifically for CPU inference speed
  under free-tier hosting constraints. A larger model would likely improve
  fine-grained discrimination at the cost of slower indexing and per-query
  latency — worth A/B testing if more compute is available.
- **No saree-specific fine-tuning.** CLIP was pretrained on general web
  image-text pairs, not fashion/textile-specific data. The region-crop
  signals exist specifically to compensate for that gap without training a
  custom model, which was out of scope.
- **Exact vs. approximate FAISS index.** `IndexFlatIP` is exact but O(n) per
  query. Fine at ~1000 vectors; would need IVF/HNSW well before six figures
  of catalog size.
- **Fixed-fraction region crops vs. real segmentation.** Cheap and requires
  no additional model, but is a blunt approximation that assumes consistent
  photography. It was validated to help (Section 6), which justifies keeping
  it despite being a heuristic rather than a learned approach.
- **Color histogram dropped, not fixed.** The likely bug (hue circularity)
  is understood but fixing and re-validating it was deprioritized in favor
  of shipping a working, evaluated system — documented here as known future
  work rather than silently left broken.

## 9. Bugs Found & Fixed During Development

Documented here because honest process matters more than a polished-looking
result. These were all found through direct testing, not by inspection alone.

- **Duplicate-SKU image collision (the most serious bug).** `build_index.py`
  originally named downloaded image files by SKU only. Since ~419 of the
  1074 CSV rows share a SKU with another row (multiple photos per design),
  rows sharing a SKU were downloading to the _same_ local filename — later
  rows silently reused an earlier row's cached file instead of downloading
  their own photo. Symptom in production: multiple visibly different
  catalog entries returned **identical similarity scores** for a search,
  because they'd been embedded from the same actual image despite having
  different metadata. **Fix:** filenames are now keyed by an MD5 hash of the
  image URL, not just the SKU, so different photos never collide
  (`scripts/build_index.py::ensure_local_image`).
- **Evaluation script silently grouping incorrectly, then silently
  evaluating nothing.** After the fix above, `record.id` was made unique by
  appending a row index (e.g. `AA313400_5`). The evaluation script's
  same-design grouping logic was, in two separate places, accidentally
  computed from this row-suffixed `id` instead of the clean `sku` field —
  first silently breaking family grouping, then (after a partial fix)
  causing every query to be skipped and a `ZeroDivisionError` on the final
  summary. **Fix:** both grouping computations in `scripts/evaluate.py` now
  use `record.sku`, not `record.id`.
- **Stale search results reappearing on unrelated chat turns.** The
  Streamlit app looked for a tool-call result across the _entire_
  accumulated conversation history on every turn, so asking "hello" after a
  real search would re-display that search's results even though no new
  search happened. **Fix:** `app.py` now only scans messages produced in the
  _current_ turn (`new_messages[prev_len:]`) when deciding whether to render
  a results grid.
- **Reranker weights initially untunable, then found to be net negative.**
  Early weights (`color=0.20`) were a guess, honestly documented as such.
  Once real evaluation was possible at full catalog scale, that guess turned
  out to be actively harmful (see Section 6) — corrected based on measured
  ablation results rather than intuition.

## 10. Testing

```bash
python -m pytest tests/ -v
```

25 tests covering image validation (valid/invalid images, valid/invalid
URLs), vector search (top-k correctness, ranking order, duplicate handling,
save/reload persistence), and the agent tool (valid input, invalid input,
structured output shape) — using synthetic embeddings or a stub search
engine, so they run fast without needing the real CLIP model or a live
catalog. This is a separate concern from the search-_quality_ evaluation in
Section 6.

## 11. Deployment

To deploy (e.g. Streamlit Community Cloud):

1. Build the full index locally (`python scripts/build_index.py ...`, per
   Setup above).
2. Either commit `index/index.faiss` + `index/metadata.json` to the repo
   (a few MB, not prohibitively large), or add a build step that runs
   `build_index.py` on first deploy.
3. Set `LLM_API_KEY` and `LLM_MODEL` as secrets in the hosting platform.
4. Deploy the repo; entry point is `app.py`.

Alternatively, build and run the provided `Dockerfile` anywhere that
supports it — see the file for details on mounting a pre-built index.

_(Fill in the live URL here once deployed.)_

## 12. Project Structure

```
project/
├── app.py                       # Streamlit chat UI (product-card results grid)
├── src/
│   ├── agent/agent.py           # Gemini + LangGraph agent, system prompt
│   ├── embeddings/model.py      # OpenCLIP wrapper
│   ├── search/
│   │   ├── vector_store.py      # FAISS index + metadata persistence
│   │   ├── engine.py            # orchestrates embed -> retrieve -> rerank
│   │   └── reranker.py          # multi-signal weighted reranking
│   ├── tools/visual_search.py   # the agent-callable tool (typed I/O)
│   └── utils/
│       ├── image_utils.py       # region crops, color histogram
│       └── query_image.py       # validation for user-supplied images
├── scripts/
│   ├── build_index.py           # offline: download, embed, index, persist
│   └── evaluate.py              # baseline vs reranked evaluation (ablation)
├── data/                        # CSV catalog + downloaded images (gitignored)
├── index/                       # persisted FAISS index + metadata (gitignored)
├── tests/                       # pytest suite (25 tests, all passing)
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```
