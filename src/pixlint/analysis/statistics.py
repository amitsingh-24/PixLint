from __future__ import annotations

from collections import Counter

import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import get_image_size
from pixlint.utils.schemas import DatasetStatistics


def compute_statistics(dataset: CVDataset) -> DatasetStatistics:
    widths: list[int] = []
    heights: list[int] = []
    ann_counts: list[int] = []
    class_counter: Counter[str] = Counter()

    for img in dataset.images:
        w, h = None, None
        if img.width and img.height:
            w, h = img.width, img.height
        else:
            size = get_image_size(img.path)
            if size:
                w, h = size
        if w and h:
            widths.append(w)
            heights.append(h)
        ann_counts.append(len(img.annotations))
        for ann in img.annotations:
            class_counter[ann.label] += 1

    total_pixels = sum(w * h for w, h in zip(widths, heights)) if widths else 0
    avg_w = float(np.mean(widths)) if widths else 0.0
    avg_h = float(np.mean(heights)) if heights else 0.0
    min_w = min(widths) if widths else 0
    min_h = min(heights) if heights else 0
    max_w = max(widths) if widths else 0
    max_h = max(heights) if heights else 0
    avg_anns = float(np.mean(ann_counts)) if ann_counts else 0.0

    return DatasetStatistics(
        dataset_id=dataset.dataset_id,
        num_images=len(dataset.images),
        num_annotations=dataset.get_num_annotations(),
        num_classes=len(dataset.get_class_names()),
        class_names=dataset.get_class_names(),
        total_pixels=total_pixels,
        avg_image_size=(round(avg_w, 1), round(avg_h, 1)),
        min_image_size=(min_w, min_h),
        max_image_size=(max_w, max_h),
        avg_annotations_per_image=round(avg_anns, 2),
        class_counts=dict(class_counter),
    )


def sample_dataset(
    dataset: CVDataset,
    n: int = 10,
    strategy: str = "random",
    seed: int | None = 42,
) -> list[str]:
    import random
    if seed is not None:
        random.seed(seed)

    indices = list(range(len(dataset.images)))

    if strategy == "random":
        selected = random.sample(indices, min(n, len(indices)))  # nosec B311

    elif strategy == "stratified":
        class_to_indices: dict[str, list[int]] = {}
        for i, img in enumerate(dataset.images):
            for ann in img.annotations:
                class_to_indices.setdefault(ann.label, []).append(i)
        selected = []
        per_class = max(1, n // max(len(class_to_indices), 1))
        for cls, idxs in class_to_indices.items():
            selected.extend(random.sample(idxs, min(per_class, len(idxs))))  # nosec B311
        if len(selected) < n:
            remaining = [i for i in indices if i not in selected]
            selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))  # nosec B311
        selected = selected[:n]

    else:
        selected = random.sample(indices, min(n, len(indices)))  # nosec B311

    return [dataset.images[i].image_id for i in selected]
