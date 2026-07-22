from __future__ import annotations

import random
from collections import defaultdict

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import SplitResult


def split_dataset(
    dataset: CVDataset,
    ratios: dict[str, float] | None = None,
    strategy: str = "stratified",
    group_by: str | None = None,
    seed: int | None = 42,
    min_samples_per_class: int = 2,
) -> SplitResult:
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    if seed is not None:
        random.seed(seed)

    image_ids = [img.image_id for img in dataset.images]

    if strategy == "random":
        splits = _random_split(image_ids, ratios)
    elif strategy == "stratified":
        splits = _stratified_split(dataset, ratios, min_samples_per_class)
    elif strategy == "temporal":
        splits = _temporal_split(dataset, ratios)
    elif strategy == "grouped":
        splits = _grouped_split(dataset, ratios, group_by)
    else:
        splits = _random_split(image_ids, ratios)

    class_dists: dict[str, dict[str, int]] = {}
    for split_name, ids in splits.items():
        counter: dict[str, int] = {}
        for img in dataset.images:
            if img.image_id in ids:
                for ann in img.annotations:
                    counter[ann.label] = counter.get(ann.label, 0) + 1
        class_dists[split_name] = counter

    return SplitResult(
        dataset_id=dataset.dataset_id,
        strategy=strategy,
        ratios=ratios,
        splits=splits,
        class_distributions=class_dists,
        seed=seed,
    )


def _random_split(
    image_ids: list[str], ratios: dict[str, float]
) -> dict[str, list[str]]:
    shuffled = image_ids.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    splits: dict[str, list[str]] = {}
    start = 0
    for split_name, ratio in ratios.items():
        end = start + int(n * ratio)
        splits[split_name] = shuffled[start:end]
        start = end
    remainder = shuffled[start:]
    if remainder:
        last_split = list(ratios.keys())[-1]
        splits[last_split] = splits.get(last_split, []) + remainder
    return splits


def _stratified_split(
    dataset: CVDataset, ratios: dict[str, float], min_per_class: int = 2
) -> dict[str, list[str]]:
    class_to_ids: dict[str, list[str]] = defaultdict(list)
    for img in dataset.images:
        labels = [ann.label for ann in img.annotations]
        if not labels:
            class_to_ids["__background__"].append(img.image_id)
        else:
            for label in set(labels):
                class_to_ids[label].append(img.image_id)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for cls, ids in class_to_ids.items():
        if len(ids) < min_per_class:
            train_ids.extend(ids)
            continue
        random.shuffle(ids)
        cls_train, cls_val, cls_test = _partition_from_ratios(ids, ratios)
        train_ids.extend(cls_train)
        val_ids.extend(cls_val)
        test_ids.extend(cls_test)

    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _temporal_split(
    dataset: CVDataset, ratios: dict[str, float]
) -> dict[str, list[str]]:
    sorted_ids = sorted(
        [img.image_id for img in dataset.images],
        key=lambda x: str(x),
    )
    train, val, test = _partition_from_ratios(sorted_ids, ratios)
    return {"train": train, "val": val, "test": test}


def _grouped_split(
    dataset: CVDataset, ratios: dict[str, float], group_by: str | None
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for img in dataset.images:
        group_key = img.metadata.get(group_by, "") if group_by else img.image_id
        if not group_key:
            group_key = img.image_id
        groups[group_key].append(img.image_id)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)
    train_groups, val_groups, test_groups = _partition_from_ratios(group_ids, ratios)

    train_ids = [img_id for g in train_groups for img_id in groups[g]]
    val_ids = [img_id for g in val_groups for img_id in groups[g]]
    test_ids = [img_id for g in test_groups for img_id in groups[g]]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _partition_from_ratios(
    items: list[str], ratios: dict[str, float]
) -> tuple[list[str], list[str], list[str]]:
    n = len(items)
    train_ratio = ratios.get("train", 0.7)
    val_ratio = ratios.get("val", 0.15)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return items[:train_end], items[train_end:val_end], items[val_end:]


def sample_split(
    split: SplitResult, split_name: str = "train", n: int = 10
) -> list[str]:
    ids = split.splits.get(split_name, [])
    return random.sample(ids, min(n, len(ids)))  # nosec B311
