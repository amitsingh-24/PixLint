from __future__ import annotations

from typing import Any

import numpy as np

from pixlint.analysis.embeddings import compute_embeddings
from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import get_image_size
from pixlint.utils.schemas import OutlierReport

_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


def detect_outliers(
    dataset: CVDataset,
    methods: list[str] | None = None,
    contamination: float = 0.05,
) -> OutlierReport:
    if methods is None:
        methods = ["embedding", "resolution", "annotation"]

    outlier_ids: set[str] = set()
    outlier_scores: dict[str, float] = {}
    method_results: dict[str, list[str]] = {}
    summary: dict[str, Any] = {}

    if "embedding" in methods and _SKLEARN_AVAILABLE:
        emb_outliers, emb_scores = _embedding_outliers(dataset, contamination)
        outlier_ids.update(emb_outliers)
        outlier_scores.update(emb_scores)
        method_results["embedding"] = emb_outliers
        summary["embedding_outliers"] = len(emb_outliers)

    if "resolution" in methods:
        res_outliers, res_scores = _resolution_outliers(dataset, contamination)
        outlier_ids.update(res_outliers)
        outlier_scores.update(res_scores)
        method_results["resolution"] = res_outliers
        summary["resolution_outliers"] = len(res_outliers)

    if "annotation" in methods:
        ann_outliers, ann_scores = _annotation_outliers(dataset, contamination)
        outlier_ids.update(ann_outliers)
        outlier_scores.update(ann_scores)
        method_results["annotation"] = ann_outliers
        summary["annotation_outliers"] = len(ann_outliers)

    return OutlierReport(
        dataset_id=dataset.dataset_id,
        methods_used=methods,
        outlier_ids=sorted(outlier_ids),
        outlier_scores=outlier_scores,
        total_outliers=len(outlier_ids),
        summary=summary,
    )


def _embedding_outliers(
    dataset: CVDataset, contamination: float
) -> tuple[list[str], dict[str, float]]:
    emb_result = compute_embeddings(dataset)
    if not emb_result.embeddings or len(emb_result.embeddings) < 3:
        return [], {}

    embeddings = np.array(emb_result.embeddings)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(embeddings)
    scores = model.score_samples(embeddings)
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1

    outlier_ids: list[str] = []
    outlier_scores: dict[str, float] = {}
    for i, img_id in enumerate(emb_result.image_ids):
        if preds[i] == -1:
            outlier_ids.append(img_id)
            normalized = (scores[i] - min_score) / score_range
            outlier_scores[img_id] = round(normalized, 4)

    return outlier_ids, outlier_scores


def _resolution_outliers(
    dataset: CVDataset, contamination: float
) -> tuple[list[str], dict[str, float]]:
    resolutions: dict[str, float] = {}
    for img in dataset.images:
        size = get_image_size(img.path)
        if size:
            megapixels = (size[0] * size[1]) / 1_000_000
        else:
            megapixels = 0.0
        resolutions[img.image_id] = megapixels

    if len(resolutions) < 3:
        return [], {}

    values = np.array(list(resolutions.values())).reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(values)
    scores = model.score_samples(values)
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1

    outlier_ids: list[str] = []
    outlier_scores: dict[str, float] = {}
    for i, img_id in enumerate(resolutions.keys()):
        if preds[i] == -1:
            outlier_ids.append(img_id)
            normalized = (scores[i] - min_score) / score_range
            outlier_scores[img_id] = round(normalized, 4)

    return outlier_ids, outlier_scores


def _annotation_outliers(
    dataset: CVDataset, contamination: float
) -> tuple[list[str], dict[str, float]]:
    features: dict[str, list[float]] = {}
    for img in dataset.images:
        if not img.annotations:
            features[img.image_id] = [0.0, 0.0, 0.0]
            continue
        bbox_areas = []
        bbox_ratios = []
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                area = abs((x2 - x1) * (y2 - y1))
                bbox_areas.append(area)
                if (y2 - y1) > 0:
                    bbox_ratios.append(abs(x2 - x1) / abs(y2 - y1))
        num_objs = len(img.annotations)
        avg_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0
        avg_ratio = float(np.mean(bbox_ratios)) if bbox_ratios else 1.0
        features[img.image_id] = [float(num_objs), avg_area, avg_ratio]

    if len(features) < 3:
        return [], {}

    values = np.array(list(features.values()))
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(values)
    scores = model.score_samples(values)
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1

    outlier_ids: list[str] = []
    outlier_scores: dict[str, float] = {}
    for i, img_id in enumerate(features.keys()):
        if preds[i] == -1:
            outlier_ids.append(img_id)
            normalized = (scores[i] - min_score) / score_range
            outlier_scores[img_id] = round(normalized, 4)

    return outlier_ids, outlier_scores
