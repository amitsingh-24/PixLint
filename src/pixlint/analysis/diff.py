from __future__ import annotations

from typing import Any

from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.quality import analyze_quality
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DatasetDiff


def dataset_diff(
    dataset_a: CVDataset,
    dataset_b: CVDataset,
) -> DatasetDiff:
    ids_a = {img.image_id for img in dataset_a.images}
    ids_b = {img.image_id for img in dataset_b.images}

    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)
    common = sorted(ids_a & ids_b)

    dist_shift = _compute_distribution_shift(dataset_a, dataset_b)
    quality_changes = _compute_quality_changes(dataset_a, dataset_b)

    return DatasetDiff(
        dataset_id_a=dataset_a.dataset_id,
        dataset_id_b=dataset_b.dataset_id,
        added_images=added,
        removed_images=removed,
        common_images=len(common),
        distribution_shift=dist_shift,
        quality_changes=quality_changes,
    )


def _compute_distribution_shift(
    dataset_a: CVDataset, dataset_b: CVDataset,
) -> dict[str, Any] | None:
    try:
        dist_a = analyze_distribution(dataset_a, analyses=["class_counts"])
        dist_b = analyze_distribution(dataset_b, analyses=["class_counts"])

        class_a = {d.class_name: d.count for d in dist_a.class_distribution}
        class_b = {d.class_name: d.count for d in dist_b.class_distribution}

        all_classes = sorted(set(list(class_a.keys()) + list(class_b.keys())))
        shifts = {}
        for cls in all_classes:
            ca = class_a.get(cls, 0)
            cb = class_b.get(cls, 0)
            if ca != cb:
                shifts[cls] = {"before": ca, "after": cb, "delta": cb - ca}

        return {
            "class_shifts": shifts,
            "total_before": dist_a.total_images,
            "total_after": dist_b.total_images,
            "imbalance_before": dist_a.imbalance_ratio,
            "imbalance_after": dist_b.imbalance_ratio,
        }
    except Exception:
        return None


def _compute_quality_changes(
    dataset_a: CVDataset, dataset_b: CVDataset,
) -> dict[str, Any] | None:
    try:
        qual_a = analyze_quality(dataset_a)
        qual_b = analyze_quality(dataset_b)
        return {
            "avg_quality_before": qual_a.average_overall,
            "avg_quality_after": qual_b.average_overall,
            "delta": round(qual_b.average_overall - qual_a.average_overall, 2),
            "flagged_before": qual_a.flagged_images,
            "flagged_after": qual_b.flagged_images,
        }
    except Exception:
        return None
