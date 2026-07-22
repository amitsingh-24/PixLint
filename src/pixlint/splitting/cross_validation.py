from __future__ import annotations

import random
from collections import defaultdict

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import KFoldResult, SplitResult


def generate_kfold_splits(
    dataset: CVDataset,
    k: int = 5,
    strategy: str = "stratified",
    seed: int | None = 42,
) -> KFoldResult:
    if seed is not None:
        random.seed(seed)

    image_ids = [img.image_id for img in dataset.images]
    n = len(image_ids)
    if k > n:
        k = n
    # K-fold cross-validation is only meaningful with k >= 2.
    k = max(2, k) if n >= 2 else 1

    folds: list[SplitResult] = []

    def _make_fold(test_ids: list[str], rest_ids: list[str]) -> SplitResult:
        # Carve a validation slice out of the non-test folds so the training
        # set is never empty (previously a whole fold was reserved for val,
        # which emptied train for small k).
        val_size = len(rest_ids) // k if len(rest_ids) >= k else 0
        val_ids = rest_ids[:val_size]
        train_ids = rest_ids[val_size:]
        total = max(1, n)
        return SplitResult(
            dataset_id=dataset.dataset_id,
            strategy=strategy,
            ratios={
                "train": round(len(train_ids) / total, 4),
                "val": round(len(val_ids) / total, 4),
                "test": round(len(test_ids) / total, 4),
            },
            splits={"train": train_ids, "val": val_ids, "test": test_ids},
            seed=seed,
        )

    if strategy == "stratified":
        class_to_ids: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            labels = [ann.label for ann in img.annotations]
            key = "_".join(sorted(set(labels))) if labels else "__background__"
            class_to_ids[key].append(img.image_id)

        fold_ids: list[list[str]] = [[] for _ in range(k)]
        for cls, ids in class_to_ids.items():
            random.shuffle(ids)
            for i, img_id in enumerate(ids):
                fold_ids[i % k].append(img_id)

        for fold_idx in range(k):
            test_ids = fold_ids[fold_idx]
            rest_ids = [img_id for i in range(k) if i != fold_idx for img_id in fold_ids[i]]
            folds.append(_make_fold(test_ids, rest_ids))
    else:
        random.shuffle(image_ids)
        fold_size = n // k
        for fold_idx in range(k):
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size if fold_idx < k - 1 else n
            test_ids = image_ids[test_start:test_end]
            rest_ids = image_ids[:test_start] + image_ids[test_end:]
            folds.append(_make_fold(test_ids, rest_ids))

    return KFoldResult(
        dataset_id=dataset.dataset_id,
        k=k,
        strategy=strategy,
        folds=folds,
        seed=seed,
    )
