"""
Validation and loading for user-supplied query images (Phase 7).

Two input paths are supported:
  - a local file path / uploaded file's bytes (Streamlit file_uploader)
  - a remote image URL

Both are funneled through `load_query_image`, which returns a validated
PIL.Image or raises `InvalidImageError` with a user-friendly message.
"""

from __future__ import annotations

import io
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
REQUEST_TIMEOUT_SECONDS = 10
ALLOWED_SCHEMES = {"http", "https"}


class InvalidImageError(Exception):
    """Raised for any user-facing image validation failure. Message is safe to display."""


def _validate_url_format(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise InvalidImageError(
            "That doesn't look like a valid image URL (must start with http:// or https://)."
        )
    if not parsed.netloc:
        raise InvalidImageError("That doesn't look like a valid image URL.")


def load_image_from_url(url: str) -> Image.Image:
    _validate_url_format(url)

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    except requests.RequestException:
        raise InvalidImageError(
            "Couldn't reach that URL. Please check the link and try again."
        )

    if resp.status_code != 200:
        raise InvalidImageError(f"That URL returned an error (HTTP {resp.status_code}).")

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError("That URL doesn't point to an image.")

    data = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise InvalidImageError("That image is too large (max 15 MB).")
        data.write(chunk)

    return _open_and_validate(data.getvalue())


def load_image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError("That image is too large (max 15 MB).")
    return _open_and_validate(data)


def _open_and_validate(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise InvalidImageError(
            "That file doesn't look like a readable image. Please try a JPG, PNG, or WEBP."
        )
    return img.convert("RGB")
