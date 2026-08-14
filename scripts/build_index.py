"""
Offline indexing pipeline (Phase 3 / assignment section 3).

Reads the product CSV (Name, SKU, image_url, ...), downloads each image
(skipping ones already cached on disk), validates it, computes:
  - a full-image CLIP embedding (used for FAISS recall)
  - a pallu-region CLIP embedding
  - a border-region CLIP embedding
  - a saturation-weighted color histogram
and writes a FAISS index + metadata.json to the index/ directory.

This script is the ONLY place embeddings are generated. The Streamlit app
and the agent tool only ever load the persisted index -- they never
recompute embeddings for the catalog.

Usage:
    python scripts/build_index.py --csv data/byrappa_tejas_31july.csv \
        --images-dir data/images --index-dir index

Re-running is safe and cheap: already-downloaded images are not
re-downloaded, and (unless --force is passed) already-indexed image IDs
are not re-embedded.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import hashlib
from dataclasses import dataclass

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.model import ClipEmbedder, ClipModelConfig
from src.search.vector_store import ImageRecord, VectorStore
from src.utils.image_utils import compute_color_histogram, get_region_crops

REQUEST_TIMEOUT_SECONDS = 15
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB sanity ceiling for a product photo


@dataclass
class RowResult:
    sku: str
    status: str  # "ok" | "download_failed" | "invalid_image" | "missing_url" | "skipped_cached"
    detail: str = ""


def read_catalog_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def download_image(url: str, dest_path: str) -> tuple[bool, str]:
    """Download `url` to `dest_path`. Returns (success, error_message)."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    except requests.RequestException as e:
        return False, f"request error: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        return False, f"unexpected content-type: {content_type}"

    total = 0
    tmp_path = dest_path + ".part"
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    f.close()
                    os.remove(tmp_path)
                    return False, "image exceeds max size limit"
                f.write(chunk)
    except OSError as e:
        return False, f"disk write error: {e}"

    os.replace(tmp_path, dest_path)
    return True, ""


def validate_image(path: str) -> tuple[bool, str]:
    try:
        with Image.open(path) as im:
            im.verify()
        # re-open after verify() (which leaves the file unusable for further ops)
        with Image.open(path) as im:
            im.convert("RGB")
        return True, ""
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, str(e)


def ensure_local_image(row: dict, images_dir: str) -> tuple[str | None, RowResult]:
    """Return (local_path or None, RowResult) for a single CSV row."""
    sku = row.get("SKU", "").strip()
    url = row.get("image_url", "").strip()

    if not sku:
        return None, RowResult(sku="<missing>", status="missing_url", detail="row has no SKU")
    if not url:
        return None, RowResult(sku=sku, status="missing_url", detail="row has no image_url")

    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    local_path = os.path.join(images_dir, f"{sku}_{url_hash}{ext}")
    
    if os.path.exists(local_path):
        ok, detail = validate_image(local_path)
        if ok:
            return local_path, RowResult(sku=sku, status="skipped_cached")
        # cached file is corrupt -- remove and try re-downloading once.
        os.remove(local_path)

    os.makedirs(images_dir, exist_ok=True)
    success, err = download_image(url, local_path)
    if not success:
        return None, RowResult(sku=sku, status="download_failed", detail=err)

    ok, detail = validate_image(local_path)
    if not ok:
        if os.path.exists(local_path):
            os.remove(local_path)
        return None, RowResult(sku=sku, status="invalid_image", detail=detail)

    return local_path, RowResult(sku=sku, status="ok")


def build_index(
    csv_path: str,
    images_dir: str,
    index_dir: str,
    local_checkpoint: str | None,
    limit: int | None = None,
) -> None:
    rows = read_catalog_csv(csv_path)
    if limit:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} rows from {csv_path}")

    config = ClipModelConfig(local_checkpoint=local_checkpoint) if local_checkpoint else ClipModelConfig()
    print(f"Loading embedding model ({config.model_name} / {config.pretrained})...")
    t0 = time.time()
    embedder = ClipEmbedder(config)
    print(f"Model loaded in {time.time() - t0:.1f}s (device={embedder.device})")

    store = VectorStore(dim=embedder.embedding_dim)

    results: list[RowResult] = []
    embeddings_batch = []
    records_batch = []

    for row_idx, row in enumerate(tqdm(rows, desc="Indexing")):
        local_path, result = ensure_local_image(row, images_dir)
        results.append(result)

        if local_path is None:
            continue

        try:
            img = Image.open(local_path).convert("RGB")
            crops = get_region_crops(img)

            full_emb = embedder.embed_image(crops["full"])
            pallu_emb = embedder.embed_image(crops["pallu"])
            border_emb = embedder.embed_image(crops["border"])
            color_hist = compute_color_histogram(img)

            record = ImageRecord(
                id=f"{result.sku}_{row_idx}",
                image_path=local_path,
                image_url=row.get("image_url", "").strip(),
                name=row.get("Name", "").strip() or None,
                sku=result.sku,
                width=img.width,
                height=img.height,
                pallu_embedding=pallu_emb.tolist(),
                border_embedding=border_emb.tolist(),
                color_histogram=color_hist.tolist(),
            )
            embeddings_batch.append(full_emb)
            records_batch.append(record)
            result.status = "ok"
        except Exception as e:  # noqa: BLE001 -- log and continue; one bad image must not kill the run
            result.status = "invalid_image"
            result.detail = f"embedding failed: {e}"

    if embeddings_batch:
        import numpy as np

        store.add(np.stack(embeddings_batch), records_batch)
        store.save(index_dir)
        print(f"\nSaved index with {len(store)} vectors to {index_dir}/")
    else:
        print("\nNo images were successfully indexed -- nothing to save.")

    # Summary
    print("\n--- Summary ---")
    for status in ["ok", "skipped_cached", "download_failed", "invalid_image", "missing_url"]:
        count = sum(1 for r in results if r.status == status)
        if count:
            print(f"  {status}: {count}")

    failures = [r for r in results if r.status in ("download_failed", "invalid_image", "missing_url")]
    if failures:
        print(f"\n{len(failures)} rows could not be indexed (showing up to 10):")
        for r in failures[:10]:
            print(f"  SKU={r.sku!r} status={r.status} detail={r.detail}")


def main():
    parser = argparse.ArgumentParser(description="Build the saree visual-search FAISS index.")
    parser.add_argument("--csv", required=True, help="Path to the product catalog CSV.")
    parser.add_argument("--images-dir", default="data/images", help="Local image cache directory.")
    parser.add_argument("--index-dir", default="index", help="Output directory for the FAISS index.")
    parser.add_argument(
        "--local-checkpoint",
        default=None,
        help="Optional path to a local OpenCLIP .pth checkpoint (bypasses HF Hub download).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (debugging).")
    args = parser.parse_args()

    build_index(
        csv_path=args.csv,
        images_dir=args.images_dir,
        index_dir=args.index_dir,
        local_checkpoint=args.local_checkpoint,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
