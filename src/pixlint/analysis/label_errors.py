"""Label-error detection — a "data linter" for annotations.

Finds annotations that are *probably wrong* without needing a trained model or
ground-truth, using self-supervised label consistency in embedding space
(a lightweight take on confident-learning / kNN label agreement):

  If an image's label disagrees with the consensus of its nearest neighbors in
  a strong pretrained embedding space, that label is suspect.

This is the kind of capability incumbents gate behind a paid/enterprise tier;
here it is a single MCP tool call.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from pixlint.analysis.embeddings import compute_embeddings
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import LabelErrorItem, LabelErrorReport

_SKLEARN_AVAILABLE = False
try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


def _primary_label(img) -> str | None:
    """The dominant label for an image (most frequent annotation label)."""
    labels = [ann.label for ann in img.annotations]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def find_label_errors(
    dataset: CVDataset,
    k: int = 10,
    threshold: float = 0.6,
    model: str = "resnet50",
) -> LabelErrorReport:
    """Flag likely-mislabeled images.

    Args:
        k: neighbors to consider.
        threshold: minimum neighbor-agreement on a *different* label required to
            flag the image (higher = more conservative / fewer flags).
    """
    notes: list[str] = []
    labeled = [(img, _primary_label(img)) for img in dataset.images]
    labeled = [(img, lab) for img, lab in labeled if lab is not None]

    if len(labeled) < k + 1:
        return LabelErrorReport(
            dataset_id=dataset.dataset_id, method="knn_embedding_consistency",
            num_images_analyzed=len(labeled), num_suspected_errors=0,
            suspected_error_rate=0.0, suspected_errors=[],
            notes=["Not enough labeled images for neighbor analysis."],
        )
    if not _SKLEARN_AVAILABLE:
        return LabelErrorReport(
            dataset_id=dataset.dataset_id, method="knn_embedding_consistency",
            num_images_analyzed=len(labeled), num_suspected_errors=0,
            suspected_error_rate=0.0, suspected_errors=[],
            notes=["scikit-learn is required for label-error detection."],
        )

    emb = compute_embeddings(dataset, model=model)
    if not emb.embeddings:
        return LabelErrorReport(
            dataset_id=dataset.dataset_id, method="knn_embedding_consistency",
            num_images_analyzed=len(labeled), num_suspected_errors=0,
            suspected_error_rate=0.0, suspected_errors=[],
            notes=["Embeddings unavailable (install the [torch] extra)."],
        )

    # Align embeddings with labels by image_id.
    label_map = {img.image_id: lab for img, lab in labeled}
    emb_by_id = {iid: np.asarray(vec, dtype=np.float32)
                 for iid, vec in zip(emb.image_ids, emb.embeddings)}
    ids = [iid for iid in emb.image_ids if iid in label_map]
    if len(ids) < k + 1:
        return LabelErrorReport(
            dataset_id=dataset.dataset_id, method="knn_embedding_consistency",
            num_images_analyzed=len(ids), num_suspected_errors=0,
            suspected_error_rate=0.0, suspected_errors=[],
            notes=["Not enough images with both embeddings and labels."],
        )

    X = np.stack([emb_by_id[i] for i in ids])
    # L2-normalize so neighbor distance reflects cosine geometry.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    n_neighbors = min(k + 1, len(ids))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(X)
    _, indices = nn.kneighbors(X)

    suspects: list[LabelErrorItem] = []
    for row, iid in enumerate(ids):
        given = label_map[iid]
        if given is None:  # unlabeled images can't be label-error suspects
            continue
        # exclude self; drop any None-labeled neighbors from the vote
        neighbor_labels = [
            lbl for j in indices[row][1:] if (lbl := label_map[ids[j]]) is not None
        ]
        if not neighbor_labels:
            continue
        counts = Counter(neighbor_labels)
        majority_label, majority_count = counts.most_common(1)[0]
        agreement = majority_count / len(neighbor_labels)
        # Flag when neighbors strongly agree on a DIFFERENT label.
        if majority_label != given and agreement >= threshold:
            given_frac = counts.get(given, 0) / len(neighbor_labels)
            confidence = round(agreement * (1.0 - given_frac), 4)
            suspects.append(LabelErrorItem(
                image_id=iid,
                given_label=given,
                suggested_label=majority_label,
                confidence=confidence,
                neighbor_agreement=round(agreement, 4),
                reason=(f"{majority_count}/{len(neighbor_labels)} nearest neighbors "
                        f"are '{majority_label}', not '{given}'."),
            ))

    suspects.sort(key=lambda s: -s.confidence)
    return LabelErrorReport(
        dataset_id=dataset.dataset_id,
        method="knn_embedding_consistency",
        num_images_analyzed=len(ids),
        num_suspected_errors=len(suspects),
        suspected_error_rate=round(len(suspects) / len(ids), 4) if ids else 0.0,
        suspected_errors=suspects[:500],
        notes=notes or [f"Analyzed {len(ids)} images with k={k}, threshold={threshold}."],
    )
