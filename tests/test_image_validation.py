import io
import sys
import os

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.query_image import (
    InvalidImageError,
    load_image_from_bytes,
    _validate_url_format,
)


def _make_valid_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestValidImage:
    def test_valid_jpeg_loads(self):
        img = load_image_from_bytes(_make_valid_jpeg_bytes())
        assert img.size == (32, 32)
        assert img.mode == "RGB"

    def test_valid_webp_loads(self):
        pil_img = Image.new("RGB", (16, 16), color=(0, 128, 0))
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP")
        img = load_image_from_bytes(buf.getvalue())
        assert img.size == (16, 16)


class TestInvalidImage:
    def test_garbage_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            load_image_from_bytes(b"this is definitely not an image")

    def test_empty_bytes_raises(self):
        with pytest.raises(InvalidImageError):
            load_image_from_bytes(b"")

    def test_oversized_image_raises(self):
        big = b"0" * (16 * 1024 * 1024)  # exceeds 15MB limit
        with pytest.raises(InvalidImageError):
            load_image_from_bytes(big)


class TestInvalidUrl:
    def test_missing_scheme_raises(self):
        with pytest.raises(InvalidImageError):
            _validate_url_format("byrappasilk.in/image.webp")

    def test_ftp_scheme_raises(self):
        with pytest.raises(InvalidImageError):
            _validate_url_format("ftp://example.com/image.jpg")

    def test_no_host_raises(self):
        with pytest.raises(InvalidImageError):
            _validate_url_format("https://")

    def test_valid_https_url_passes(self):
        _validate_url_format("https://example.com/image.jpg")  # should not raise
