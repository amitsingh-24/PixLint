from __future__ import annotations

from typing import Any

import numpy as np

from pixlint.analysis.embeddings import compute_embeddings
from pixlint.core.loader import CVDataset

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import pairwise_distances
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


def uncertainty_sampling(
    dataset: CVDataset,
    method: str = "entropy",
    n: int = 10,
    model: str = "resnet50",
    label_column: str = "label",
) -> dict[str, Any]:
    if not _SKLEARN_AVAILABLE:
        return {"error": "scikit-learn required for uncertainty sampling"}

    emb_result = compute_embeddings(dataset, model=model)
    if not emb_result.embeddings or len(emb_result.embeddings) < 2:
        return {"error": "Need at least 2 images with embeddings"}

    embeddings = np.array(emb_result.embeddings)
    image_ids = emb_result.image_ids
    n = min(n, len(image_ids))

    labels = _get_labels_for_images(dataset)
    labeled_idx = [i for i, img_id in enumerate(image_ids) if img_id in labels]
    unlabeled_idx = [i for i, img_id in enumerate(image_ids) if img_id not in labels]

    if len(labeled_idx) < 2:
        return {"error": "Need at least 2 labeled images to train classifier", "strategy": "random_fallback", "sampled_ids": image_ids[:n]}

    try:
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        X_labeled = embeddings[labeled_idx]
        y_labeled = np.array([labels[image_ids[i]] for i in labeled_idx])
        clf.fit(X_labeled, y_labeled)

        if unlabeled_idx:
            X_unlabeled = embeddings[unlabeled_idx]
            probs = clf.predict_proba(X_unlabeled)

            if method == "least_confidence":
                scores = 1 - probs.max(axis=1)
            elif method == "margin":
                sorted_probs = np.sort(probs, axis=1)
                scores = sorted_probs[:, -1] - sorted_probs[:, -2]
                scores = 1 - scores
            elif method == "entropy":
                scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                if scores.max() > 0:
                    scores = scores / scores.max()
            else:
                scores = 1 - probs.max(axis=1)

            top_indices = np.argsort(scores)[-n:][::-1]
            sampled_ids = [image_ids[unlabeled_idx[i]] for i in top_indices if unlabeled_idx[i] < len(image_ids)]
        else:
            sampled_ids = image_ids[:n]
    except Exception as e:
        return {"error": f"Uncertainty sampling failed: {e}", "strategy": "random_fallback", "sampled_ids": image_ids[:n]}

    return {
        "strategy": f"uncertainty_{method}",
        "n": len(sampled_ids),
        "sampled_ids": sampled_ids,
        "method": method,
        "num_labeled": len(labeled_idx),
        "num_unlabeled": len(unlabeled_idx),
    }


def diversity_sampling(
    dataset: CVDataset,
    n: int = 10,
    model: str = "resnet50",
) -> dict[str, Any]:
    if not _SKLEARN_AVAILABLE:
        return {"error": "scikit-learn required for diversity sampling"}

    emb_result = compute_embeddings(dataset, model=model)
    if not emb_result.embeddings:
        return {"error": "No embeddings available (install the [torch] extra)"}

    embeddings = np.array(emb_result.embeddings)
    image_ids = emb_result.image_ids
    # Return as many as we can rather than erroring when fewer than n images exist.
    n = min(n, len(image_ids))

    try:
        n_samples = len(embeddings)
        if n >= n_samples:
            return {"strategy": "diversity", "n": n_samples, "sampled_ids": image_ids}

        dist_matrix = pairwise_distances(embeddings, metric="cosine")
        selected: list[int] = [0]
        candidates = list(range(1, n_samples))

        while len(selected) < n and candidates:
            farthest_idx = max(
                candidates,
                key=lambda c: min(dist_matrix[c][s] for s in selected),
            )
            selected.append(farthest_idx)
            candidates.remove(farthest_idx)

        sampled_ids = [image_ids[i] for i in selected]
    except Exception as e:
        return {"error": f"Diversity sampling failed: {e}", "sampled_ids": image_ids[:n]}

    return {
        "strategy": "diversity",
        "n": len(sampled_ids),
        "sampled_ids": sampled_ids,
    }


def query_strategy(
    dataset: CVDataset,
    strategy: str = "uncertainty",
    n: int = 10,
    model: str = "resnet50",
    alpha: float = 0.5,
) -> dict[str, Any]:
    if strategy == "uncertainty":
        return uncertainty_sampling(dataset, method="entropy", n=n, model=model)
    elif strategy == "diversity":
        return diversity_sampling(dataset, n=n, model=model)
    elif strategy == "combined":
        unc_result = uncertainty_sampling(dataset, method="entropy", n=n * 2, model=model)
        if "error" in unc_result:
            return diversity_sampling(dataset, n=n, model=model)

        unc_ids = set(unc_result.get("sampled_ids", []))
        if not unc_ids:
            return diversity_sampling(dataset, n=n, model=model)

        sub_dataset = _filter_dataset_by_ids(dataset, list(unc_ids))
        return diversity_sampling(sub_dataset, n=n, model=model)
    else:
        return {"error": f"Unknown strategy: {strategy}"}


def _get_labels_for_images(dataset: CVDataset) -> dict[str, int]:
    labels: dict[str, int] = {}
    class_names = dataset.get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    for img in dataset.images:
        if img.annotations:
            primary_label = img.annotations[0].label
            if primary_label in class_to_idx:
                labels[img.image_id] = class_to_idx[primary_label]
    return labels


def _filter_dataset_by_ids(dataset: CVDataset, ids: list[str]) -> CVDataset:
    id_set = set(ids)
    filtered = CVDataset.__new__(CVDataset)
    filtered.path = dataset.path
    filtered._name = dataset.name + "_filtered"
    filtered._format = dataset.format
    filtered._dataset_id = dataset.dataset_id + "_filtered"
    filtered._images = [img for img in dataset.images if img.image_id in id_set]
    return filtered
