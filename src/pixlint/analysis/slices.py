"""Weak-slice / bias discovery.

Buckets the dataset along class, brightness, image-size, and aspect-ratio
dimensions and surfaces slices that are under-represented or low-quality — so an
agent can target exactly what to collect or augment next. This is "slice
discovery" (a research-grade dataset-debugging idea) exposed as one MCP tool.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import get_image_size
from pixlint.utils.schemas import SliceInfo, SliceReport


def _brightness_bucket(path: str) -> str | None:
    img = cv2.imread(path)
    if img is None:
        return None
    mean = float(np.mean(img))
    if mean < 60:
        return "dark"
    if mean > 180:
        return "bright"
    return "normal"


def _size_bucket(w: int | None, h: int | None) -> str:
    if not (w and h):
        return "unknown"
    area = w * h
    if area < 256 * 256:
        return "small"
    if area > 1024 * 1024:
        return "large"
    return "medium"


def _aspect_bucket(w: int | None, h: int | None) -> str:
    if not (w and h) or h == 0:
        return "unknown"
    r = w / h
    if r < 0.9:
        return "portrait"
    if r > 1.1:
        return "landscape"
    return "square"


def discover_slices(
    dataset: CVDataset,
    max_brightness_samples: int = 300,
    underrep_pct: float = 3.0,
    low_quality_threshold: float = 50.0,
) -> SliceReport:
    """Find weak (under-represented or low-quality) slices of the dataset."""
    n = len(dataset.images)
    if n == 0:
        return SliceReport(dataset_id=dataset.dataset_id, num_images=0)

    # Quality per image (best-effort; skip if it fails).
    quality_by_id: dict[str, float] = {}
    try:
        from pixlint.analysis.quality import analyze_quality
        report = analyze_quality(dataset)
        quality_by_id = {q.image_id: q.overall_score for q in report.per_image
                         if q.overall_score is not None}
    except Exception:
        pass

    # Accumulate counts and quality per slice across dimensions.
    slice_counts: dict[str, Counter] = {
        "class": Counter(), "brightness": Counter(), "size": Counter(), "aspect": Counter(),
    }
    slice_quality: dict[str, dict[str, list[float]]] = {
        dim: defaultdict(list) for dim in slice_counts
    }

    # Brightness is expensive (reads pixels) — sample it.
    step = max(1, n // max_brightness_samples)

    for idx, img in enumerate(dataset.images):
        w, h = img.width, img.height
        if not (w and h):
            size = get_image_size(img.path)
            if size:
                w, h = size
        q = quality_by_id.get(img.image_id)

        classes = {ann.label for ann in img.annotations} or {"__unlabeled__"}
        for c in classes:
            slice_counts["class"][c] += 1
            if q is not None:
                slice_quality["class"][c].append(q)

        for dim, key in (("size", _size_bucket(w, h)), ("aspect", _aspect_bucket(w, h))):
            slice_counts[dim][key] += 1
            if q is not None:
                slice_quality[dim][key].append(q)

        if idx % step == 0:
            b = _brightness_bucket(img.path)
            if b:
                slice_counts["brightness"][b] += 1
                if q is not None:
                    slice_quality["brightness"][b].append(q)

    all_slices: list[SliceInfo] = []
    weak: list[SliceInfo] = []
    for dim, counter in slice_counts.items():
        total_dim = sum(counter.values()) or 1
        for key, count in counter.items():
            pct = round(100.0 * count / total_dim, 2)
            qs = slice_quality[dim].get(key, [])
            avg_q = round(float(np.mean(qs)), 2) if qs else None
            problems: list[str] = []
            # Under-representation only meaningful for class/brightness (not size/aspect majorities).
            if dim in ("class", "brightness") and pct < underrep_pct and count < n:
                problems.append(f"under-represented ({pct}% of {dim})")
            if avg_q is not None and avg_q < low_quality_threshold:
                problems.append(f"low avg quality ({avg_q})")
            info = SliceInfo(
                slice_key=f"{dim}={key}", dimension=dim, count=count,
                percentage=pct, avg_quality=avg_q, problems=problems,
            )
            all_slices.append(info)
            if problems:
                weak.append(info)

    weak.sort(key=lambda s: (s.count, s.avg_quality if s.avg_quality is not None else 999))

    recommendations: list[str] = []
    rare_classes = [s for s in weak if s.dimension == "class" and "under-represented" in " ".join(s.problems)]
    if rare_classes:
        names = ", ".join(s.slice_key.split("=", 1)[1] for s in rare_classes[:5])
        recommendations.append(
            f"Collect or augment more examples of rare classes: {names}. "
            f"Use augment_dataset_tool or filter_dataset_tool to rebalance.")
    lowq = [s for s in weak if s.avg_quality is not None and s.avg_quality < low_quality_threshold]
    if lowq:
        recommendations.append(
            f"{len(lowq)} slice(s) have low average quality; inspect with "
            f"query_dataset_tool (max_quality={low_quality_threshold:.0f}) and clean or replace them.")
    if not recommendations:
        recommendations.append("No severely weak slices detected — distribution looks balanced.")

    return SliceReport(
        dataset_id=dataset.dataset_id,
        num_images=n,
        weak_slices=weak[:50],
        all_slices=sorted(all_slices, key=lambda s: (s.dimension, -s.count)),
        recommendations=recommendations,
    )
