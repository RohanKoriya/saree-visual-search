# """
# Search quality evaluation (Phase 14 / assignment section 14).

# This script compares BASELINE (raw full-image CLIP embedding similarity)
# against IMPROVED (reranked: embedding + pallu + border + color) search,
# using whatever ground-truth "same family" signal is actually present in
# the catalog metadata -- specifically, products whose SKU shares the same
# prefix are color variants of the same underlying design (verified by
# inspecting the `Name` field, e.g. "Munga Crape Sarees - Mustard/Blue/Black
# Colour" all share SKU prefix AA3134). This is real, observable structure
# in the data, not an invented metric.

# Metric: for each query, we check whether same-family items rank above
# different-family items, and report the mean rank of family members
# (lower = better) under baseline vs reranked ordering.

# IMPORTANT: this script's default sample (6 images) is far too small to
# draw statistically meaningful conclusions about the reranker's weights --
# it exists to (a) prove the evaluation harness itself works correctly and
# (b) give a qualitative, honestly-labeled first look. Re-run it against
# the full indexed catalog (after scripts/build_index.py has processed all
# ~1074 rows) for a real evaluation, and update README's "Search Quality
# Evaluation" section with those numbers instead of these.
# """

# from __future__ import annotations

# import argparse
# import os
# import sys
# from collections import defaultdict

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from PIL import Image

# from src.search.engine import SearchEngine, SearchEngineConfig
# from src.search.vector_store import VectorStore
# from src.utils.image_utils import compute_color_histogram, get_region_crops


# def sku_family(sku: str) -> str:
#     """Heuristic: strip the trailing numeric suffix to get a 'design family' key.
#     e.g. AA313400, AA313402, AA313403 -> 'AA3134'. This groups color variants
#     of the same design based on this catalog's observed SKU numbering pattern.
#     """
#     i = len(sku)
#     while i > 0 and sku[i - 1].isdigit():
#         i -= 1
#     # keep all but the last 2 digits of the numeric suffix (observed pattern:
#     # last 2 digits vary across color variants, e.g. ...00/...02/...03)
#     digits = sku[i:]
#     return sku[:i] + digits[:-2] if len(digits) > 2 else sku[:i]


# def evaluate(index_dir: str, local_checkpoint: str | None):
#     engine = SearchEngine(SearchEngineConfig(index_dir=index_dir, clip_local_checkpoint=local_checkpoint))
#     store = engine.store

#     families = defaultdict(list)
#     for r in store.records:
#         families[sku_family(r.sku)].append(r.id)

#     eval_skus = [sku for fam, skus in families.items() if len(skus) >= 2 for sku in skus]
#     if not eval_skus:
#         print("No same-family groups found in this index (need >= 2 items sharing a design). "
#               "Nothing to evaluate against this ground-truth signal.")
#         return

#     print(f"Found {len(eval_skus)} images across {sum(1 for f in families.values() if len(f) >= 2)} "
#           f"design families with >=2 color variants.\n")

#     baseline_ranks, reranked_ranks = [], []

#     for query_id in eval_skus:
#         query_record = next(r for r in store.records if r.id == query_id)
#         family_key = sku_family(query_record.sku)
#         same_family = set(families[family_key]) - {query_id}
#         if not same_family:
#             continue

#         img = Image.open(query_record.image_path).convert("RGB")
#         crops = get_region_crops(img)
#         full_emb = engine.embedder.embed_image(crops["full"])
#         pallu_emb = engine.embedder.embed_image(crops["pallu"]).tolist()
#         border_emb = engine.embedder.embed_image(crops["border"]).tolist()
#         color_hist = compute_color_histogram(img)

#         baseline = store.search(full_emb, top_k=len(store))
#         baseline_order = [r.id for r, _ in baseline if r.id != query_id]

#         reranked = engine.reranker.rerank(pallu_emb, border_emb, color_hist, baseline, top_k=len(store))
#         reranked_order = [s.record.id for s in reranked if s.record.id != query_id]

#         b_ranks = [baseline_order.index(sku) + 1 for sku in same_family]
#         r_ranks = [reranked_order.index(sku) + 1 for sku in same_family]

#         baseline_ranks.extend(b_ranks)
#         reranked_ranks.extend(r_ranks)

#         print(f"Query {query_id} (family {family_key}, {len(same_family)} same-family item(s)):")
#         print(f"  baseline mean rank of same-family items:  {sum(b_ranks)/len(b_ranks):.2f}")
#         print(f"  reranked mean rank of same-family items:  {sum(r_ranks)/len(r_ranks):.2f}")

#     print("\n--- Overall (lower mean rank = better; ideal = 1.0) ---")
#     print(f"Baseline (embedding-only):        {sum(baseline_ranks)/len(baseline_ranks):.2f}")
#     print(f"Reranked (embedding+color+region): {sum(reranked_ranks)/len(reranked_ranks):.2f}")
#     print(f"\n(n={len(baseline_ranks)} query-item pairs -- see this script's module docstring "
#           f"for why this sample size is NOT sufficient to conclude the reranker generally helps.)")


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate baseline vs reranked search quality.")
#     parser.add_argument("--index-dir", default="index")
#     parser.add_argument("--local-checkpoint", default=None)
#     args = parser.parse_args()
#     evaluate(args.index_dir, args.local_checkpoint)


# if __name__ == "__main__":
#     main()



"""
Search quality evaluation (Phase 14 / assignment section 14).

This script compares BASELINE (raw full-image CLIP embedding similarity)
against IMPROVED (reranked: embedding + pallu + border + color) search,
using whatever ground-truth "same family" signal is actually present in
the catalog metadata -- specifically, products whose SKU shares the same
prefix are color variants of the same underlying design (verified by
inspecting the `Name` field, e.g. "Munga Crape Sarees - Mustard/Blue/Black
Colour" all share SKU prefix AA3134). This is real, observable structure
in the data, not an invented metric.

Metric: for each query, we check whether same-family items rank above
different-family items, and report the mean rank of family members
(lower = better) under baseline vs reranked ordering.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.search.engine import SearchEngine, SearchEngineConfig
from src.search.reranker import Reranker, RerankWeights
from src.search.vector_store import VectorStore
from src.utils.image_utils import compute_color_histogram, get_region_crops

# Ablation configs: isolate each signal so we know which one is actually
# helping vs hurting, rather than judging the combined reranker as one
# black box. "baseline" (embedding-only) is computed separately below and
# is the reference point everything else is compared against.
ABLATION_CONFIGS = {
    "color_only":         RerankWeights(embedding=0.70, pallu=0.0,  border=0.0,  color=0.30),
    "pallu_only":         RerankWeights(embedding=0.70, pallu=0.30, border=0.0,  color=0.0),
    "border_only":        RerankWeights(embedding=0.70, pallu=0.0,  border=0.30, color=0.0),
    "combined_current":   RerankWeights(embedding=0.60, pallu=0.20, border=0.20, color=0.0),
}


def sku_family(sku: str) -> str:
    """Heuristic: strip the trailing numeric suffix to get a 'design family' key.
    e.g. AA313400, AA313402, AA313403 -> 'AA3134'. This groups color variants
    of the same design based on this catalog's observed SKU numbering pattern.
    """
    i = len(sku)
    while i > 0 and sku[i - 1].isdigit():
        i -= 1
    digits = sku[i:]
    return sku[:i] + digits[:-2] if len(digits) > 2 else sku[:i]


def evaluate(index_dir: str, local_checkpoint: str | None):
    engine = SearchEngine(SearchEngineConfig(index_dir=index_dir, clip_local_checkpoint=local_checkpoint))
    store = engine.store

    families = defaultdict(list)
    for r in store.records:
        families[sku_family(r.sku)].append(r.id)

    eval_skus = [sku for fam, skus in families.items() if len(skus) >= 2 for sku in skus]
    if not eval_skus:
        print("No same-family groups found in this index (need >= 2 items sharing a design). "
              "Nothing to evaluate against this ground-truth signal.")
        return

    print(f"Found {len(eval_skus)} images across {sum(1 for f in families.values() if len(f) >= 2)} "
          f"design families with >=2 color variants.\n")

    rerankers = {name: Reranker(weights) for name, weights in ABLATION_CONFIGS.items()}
    ranks = {"baseline": []}
    for name in ABLATION_CONFIGS:
        ranks[name] = []

    for query_id in eval_skus:
        query_record = next(r for r in store.records if r.id == query_id)
        family_key = sku_family(query_record.sku)
        same_family = set(families[family_key]) - {query_id}
        if not same_family:
            continue

        img = Image.open(query_record.image_path).convert("RGB")
        crops = get_region_crops(img)
        full_emb = engine.embedder.embed_image(crops["full"])
        pallu_emb = engine.embedder.embed_image(crops["pallu"]).tolist()
        border_emb = engine.embedder.embed_image(crops["border"]).tolist()
        color_hist = compute_color_histogram(img)

        candidates = store.search(full_emb, top_k=len(store))

        baseline_order = [r.id for r, _ in candidates if r.id != query_id]
        ranks["baseline"].extend(baseline_order.index(sku) + 1 for sku in same_family)

        for name, reranker in rerankers.items():
            reranked = reranker.rerank(pallu_emb, border_emb, color_hist, candidates, top_k=len(store))
            reranked_order = [s.record.id for s in reranked if s.record.id != query_id]
            ranks[name].extend(reranked_order.index(sku) + 1 for sku in same_family)

    print("--- Overall (lower mean rank = better; ideal = 1.0) ---")
    n = len(ranks["baseline"])
    print(f"baseline (embedding-only):          {sum(ranks['baseline'])/n:.2f}")
    for name in ABLATION_CONFIGS:
        print(f"{name:35s}  {sum(ranks[name])/n:.2f}")
    print(f"\n(n={n} query-item pairs)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs reranked search quality.")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--local-checkpoint", default=None)
    args = parser.parse_args()
    evaluate(args.index_dir, args.local_checkpoint)


if __name__ == "__main__":
    main()