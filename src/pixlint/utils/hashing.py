from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
from scipy.fftpack import dct as _scipy_dct

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def md5_hash(filepath: str) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    pil_img = Image.fromarray(image)
    img = pil_img.convert("L").resize((hash_size, hash_size), LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    dct = _dct_2d(pixels)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    diff = dct_low > med
    hash_bits = "".join("1" if b else "0" for row in diff for b in row)
    return hex(int(hash_bits, 2))


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    pil_img = Image.fromarray(image)
    img = pil_img.convert("L").resize((hash_size + 1, hash_size), LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_bits = "".join("1" if b else "0" for row in diff for b in row)
    return hex(int(hash_bits, 2))


def compute_ahash(image: np.ndarray, hash_size: int = 8) -> str:
    pil_img = Image.fromarray(image)
    img = pil_img.convert("L").resize((hash_size, hash_size), LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    avg = pixels.mean()
    diff = pixels > avg
    hash_bits = "".join("1" if b else "0" for row in diff for b in row)
    return hex(int(hash_bits, 2))


def hamming_distance(hash1: str, hash2: str) -> int:
    h1 = int(hash1, 16)
    h2 = int(hash2, 16)
    return bin(h1 ^ h2).count("1")


def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    intersection = np.minimum(hist1, hist2).sum()
    return intersection / max(hist1.sum(), hist2.sum())


def _dct_2d(x: np.ndarray) -> np.ndarray:
    return _scipy_dct(_scipy_dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")
