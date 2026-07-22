from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import (
    ComplexityMetric,
    DistributionMetric,
    DistributionReport,
    LabelCoOccurrence,
    SpatialHeatmap,
)


def analyze_distribution(
    dataset: CVDataset,
    analyses: list[str] | None = None,
    group_by: str | None = None,
) -> DistributionReport:
    if analyses is None:
        analyses = ["class_counts", "spatial", "co_occurrence", "complexity"]

    class_counter: Counter[str] = Counter()
    annotation_counter: Counter[str] = Counter()
    total_anns = 0

    for img in dataset.images:
        ann_labels = [ann.label for ann in img.annotations]
        if ann_labels:
            for label in set(ann_labels):
                class_counter[label] += 1
            for label in ann_labels:
                annotation_counter[label] += 1
                total_anns += 1
        else:
            class_counter["__background__"] += 1

    total_images = len(dataset.images)
    class_distribution = []
    for cls_name in sorted(class_counter.keys()):
        count = class_counter[cls_name]
        ann_count = annotation_counter.get(cls_name, 0)
        pct = round(count / total_images * 100, 2) if total_images > 0 else 0.0
        class_distribution.append(
            DistributionMetric(
                class_name=cls_name,
                count=count,
                percentage=pct,
                annotation_count=ann_count,
            )
        )

    imbalance_ratio = None
    gini = None
    if len(class_distribution) > 1:
        counts = [d.count for d in class_distribution]
        max_count = max(counts)
        min_count = max(min([c for c in counts if c > 0] or [1]), 1)
        imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else None
        total = sum(counts)
        if total > 0:
            sorted_counts = np.sort(counts)
            cumsum = np.cumsum(sorted_counts, dtype=float)
            cumsum_total = cumsum[-1]
            if cumsum_total > 0:
                cumsum /= cumsum_total
            n = len(sorted_counts)
            gini = round((n + 1 - 2 * np.sum(cumsum)) / n, 4) if n > 0 else None

    spatial_heatmap = None
    if "spatial" in analyses:
        spatial_heatmap = _compute_spatial_heatmap(dataset)

    label_co_occurrence = None
    if "co_occurrence" in analyses:
        label_co_occurrence = _compute_co_occurrence(dataset)

    complexity_scores = None
    if "complexity" in analyses:
        complexity_scores = _compute_complexity(dataset)

    return DistributionReport(
        dataset_id=dataset.dataset_id,
        class_distribution=class_distribution,
        total_images=total_images,
        total_annotations=total_anns,
        imbalance_ratio=imbalance_ratio,
        gini_coefficient=gini,
        spatial_heatmap=spatial_heatmap,
        label_co_occurrence=label_co_occurrence,
        complexity_scores=complexity_scores,
    )


def _compute_spatial_heatmap(dataset: CVDataset) -> SpatialHeatmap | None:
    grid_size = (10, 10)
    grid = np.zeros(grid_size, dtype=np.float64)
    count = 0

    for img in dataset.images:
        if img.width is None or img.height is None:
            continue
        w, h = img.width, img.height
        if w <= 0 or h <= 0:
            continue
        for ann in img.annotations:
            if ann.bbox is None:
                continue
            x1, y1, x2, y2 = ann.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            gx = min(int(cx / max(w, 1) * grid_size[0]), grid_size[0] - 1)
            gy = min(int(cy / max(h, 1) * grid_size[1]), grid_size[1] - 1)
            if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
                grid[gy, gx] += 1.0
                count += 1

    if count == 0:
        grid.fill(0.0)
    else:
        grid = grid / count * 100.0

    return SpatialHeatmap(grid=grid.tolist(), grid_size=grid_size)


def _compute_co_occurrence(dataset: CVDataset) -> list[LabelCoOccurrence]:
    co_occur: dict[tuple[str, str], int] = defaultdict(int)
    class_counts: Counter[str] = Counter()

    for img in dataset.images:
        labels = sorted(set(ann.label for ann in img.annotations))
        for label in labels:
            class_counts[label] += 1
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pair = (labels[i], labels[j])
                co_occur[pair] += 1

    results: list[LabelCoOccurrence] = []
    for (a, b), cnt in sorted(co_occur.items(), key=lambda x: -x[1]):
        union = class_counts[a] + class_counts[b] - cnt
        jaccard = cnt / max(union, 1)
        results.append(
            LabelCoOccurrence(
                class_a=a,
                class_b=b,
                count=cnt,
                jaccard_index=round(jaccard, 4),
            )
        )

    return results


def _compute_complexity(dataset: CVDataset) -> list[ComplexityMetric]:
    results: list[ComplexityMetric] = []

    for img in dataset.images:
        obj_count = len(img.annotations)
        if obj_count == 0:
            results.append(
                ComplexityMetric(
                    image_id=img.image_id,
                    object_count=0,
                    annotation_density=0.0,
                    size_variance=0.0,
                    complexity_score=0.0,
                )
            )
            continue

        areas = []
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                area = abs((x2 - x1) * (y2 - y1))
                areas.append(area)
            elif ann.area:
                areas.append(ann.area)

        img_area = (img.width or 1) * (img.height or 1)
        density = sum(areas) / img_area if img_area > 0 else 0.0
        size_var = float(np.var(areas)) if len(areas) > 1 else 0.0

        complexity = min(100, (obj_count * 10 + density * 50 + np.log1p(size_var) * 5))
        results.append(
            ComplexityMetric(
                image_id=img.image_id,
                object_count=obj_count,
                annotation_density=round(density, 4),
                size_variance=round(size_var, 2),
                complexity_score=round(complexity, 2),
            )
        )

    return results
