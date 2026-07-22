from __future__ import annotations

import os

import numpy as np
from PIL import Image, UnidentifiedImageError

# Guard against decompression-bomb DoS: a small file can decode to a huge
# in-memory bitmap. Cap the total pixel count (configurable via env). PIL
# raises Image.DecompressionBombError when a decoded image exceeds this.
try:
    _MAX_PIXELS = int(os.environ.get("CV_MAX_IMAGE_PIXELS", str(64_000_000)))  # ~64 MP
except ValueError:
    _MAX_PIXELS = 64_000_000
Image.MAX_IMAGE_PIXELS = _MAX_PIXELS


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".ppm", ".pgm"}


def is_supported_image(path: str) -> bool:
    ext = path.lower().rsplit(".", 1)[-1]
    return f".{ext}" in SUPPORTED_EXTENSIONS


def read_image(path: str) -> np.ndarray | None:
    try:
        pil_image: Image.Image = Image.open(path)
        pil_image.load()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)
    except (UnidentifiedImageError, FileNotFoundError, OSError, Image.DecompressionBombError):
        return None


def read_image_pil(path: str) -> Image.Image | None:
    try:
        pil_image: Image.Image = Image.open(path)
        pil_image.load()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return pil_image
    except (UnidentifiedImageError, FileNotFoundError, OSError, Image.DecompressionBombError):
        return None


def get_image_size(path: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except (UnidentifiedImageError, FileNotFoundError, OSError, Image.DecompressionBombError):
        return None
