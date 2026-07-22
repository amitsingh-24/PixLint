from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pixlint.core.loader import CVDataset
from pixlint.utils.hashing import md5_hash, compute_phash, hamming_distance
from pixlint.utils.image_io import read_image
from pixlint.utils.schemas import DuplicateGroup, DuplicateReport

_CLIP_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    _CLIP_AVAILABLE = True
except ImportError:
    pass


def find_duplicates(
    dataset: CVDataset,
    methods: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    include_pairs: bool = True,
    parallel: bool = True,
) -> DuplicateReport:
    if methods is None:
        methods = ["exact", "perceptual", "ssim"]

    thresholds = thresholds or {}
    phash_threshold = int(thresholds.get("phash", 8))
    ssim_threshold = thresholds.get("ssim", 0.85)
    semantic_threshold = thresholds.get("semantic", 0.92)

    duplicate_groups: list[DuplicateGroup] = []
    group_id = 0
    method_summary: dict[str, int] = {m: 0 for m in methods}

    image_cache: dict[str, np.ndarray | None] = {}

    def _get_image(img_id: str) -> np.ndarray | None:
        if img_id not in image_cache:
            path_map = {img.image_id: img.path for img in dataset.images}
            path = path_map.get(img_id)
            if path:
                image_cache[img_id] = read_image(path)
            else:
                image_cache[img_id] = None
        return image_cache[img_id]

    if "exact" in methods:
        hash_to_ids: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            h = md5_hash(img.path)
            hash_to_ids[h].append(img.image_id)

        for h, ids in hash_to_ids.items():
            if len(ids) > 1:
                method_summary["exact"] += 1
                group_id += 1
                duplicate_groups.append(
                    DuplicateGroup(
                        group_id=group_id,
                        method="exact",
                        image_ids=ids,
                        scores=[1.0] * len(ids),
                        recommended_keeper=ids[0],
                    )
                )

    if "perceptual" in methods:
        phash_map: dict[str, str] = {}
        for img in dataset.images:
            image = _get_image(img.image_id)
            if image is None:
                continue
            ph = compute_phash(image)
            phash_map[img.image_id] = ph

        handled: set[str] = set()
        all_ids = list(phash_map.keys())
        for i in range(len(all_ids)):
            if all_ids[i] in handled:
                continue
            group: list[str] = [all_ids[i]]
            for j in range(i + 1, len(all_ids)):
                if all_ids[j] in handled:
                    continue
                dist = hamming_distance(phash_map[all_ids[i]], phash_map[all_ids[j]])
                if dist <= phash_threshold:
                    group.append(all_ids[j])
            if len(group) > 1:
                method_summary["perceptual"] += 1
                group_id += 1
                handled.update(group)
                scores = [1.0 - hamming_distance(phash_map[g], phash_map[group[0]]) / 64.0 for g in group]
                duplicate_groups.append(
                    DuplicateGroup(
                        group_id=group_id,
                        method="perceptual",
                        image_ids=group,
                        scores=scores,
                        recommended_keeper=group[0],
                    )
                )

    if "ssim" in methods:
        ssim_pairs = _find_ssim_duplicates(dataset, ssim_threshold)
        for pair_ids in ssim_pairs:
            method_summary["ssim"] += 1
            group_id += 1
            duplicate_groups.append(
                DuplicateGroup(
                    group_id=group_id,
                    method="ssim",
                    image_ids=pair_ids,
                    scores=[1.0] * len(pair_ids),
                    recommended_keeper=pair_ids[0],
                )
            )

    if "semantic" in methods:
        semantic_groups = _find_semantic_duplicates(dataset, semantic_threshold)
        for group_ids in semantic_groups:
            method_summary["semantic"] += 1
            group_id += 1
            duplicate_groups.append(
                DuplicateGroup(
                    group_id=group_id,
                    method="semantic",
                    image_ids=group_ids,
                    scores=[1.0] * len(group_ids),
                    recommended_keeper=group_ids[0],
                )
            )

    total_dup = sum(len(g.image_ids) for g in duplicate_groups)
    savings_estimate = _estimate_savings(duplicate_groups)

    return DuplicateReport(
        dataset_id=dataset.dataset_id,
        total_duplicate_groups=len(duplicate_groups),
        total_duplicate_images=total_dup,
        duplicate_groups=duplicate_groups,
        estimated_space_savings=savings_estimate,
        method_summary=method_summary,
    )


def _find_ssim_duplicates(
    dataset: CVDataset, threshold: float
) -> list[list[str]]:
    groups: list[list[str]] = []
    handled: set[str] = set()
    images_list = list(dataset.images)
    _img_cache: dict[str, np.ndarray | None] = {}

    def _get_img(img_id: str) -> np.ndarray | None:
        if img_id not in _img_cache:
            path_map = {img.image_id: img.path for img in dataset.images}
            path = path_map.get(img_id)
            _img_cache[img_id] = read_image(path) if path else None
        return _img_cache[img_id]

    for i in range(len(images_list)):
        if images_list[i].image_id in handled:
            continue
        img1 = _get_img(images_list[i].image_id)
        if img1 is None:
            continue
        group: list[str] = [images_list[i].image_id]
        for j in range(i + 1, len(images_list)):
            if images_list[j].image_id in handled:
                continue
            img2 = _get_img(images_list[j].image_id)
            if img2 is None:
                continue
            try:
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                min_h, min_w = min(h1, h2), min(w1, w2)
                if min_h < 7 or min_w < 7:
                    continue
                r1 = cv2.resize(img1, (min_w, min_h))
                r2 = cv2.resize(img2, (min_w, min_h))
                score = ssim(r1, r2, channel_axis=-1, win_size=min(7, min_h, min_w))
                if score >= threshold:
                    group.append(images_list[j].image_id)
            except (ValueError, RuntimeError):
                continue
        if len(group) > 1:
            handled.update(group)
            groups.append(group)

    return groups


def _find_semantic_duplicates(
    dataset: CVDataset, threshold: float
) -> list[list[str]]:
    if not _CLIP_AVAILABLE:
        return []

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(device).eval()
        transform = weights.transforms()
        preprocess = T.Compose([
            T.ToPILImage(),
            transform,
        ])

        embeddings: dict[str, np.ndarray] = {}
        for img in dataset.images:
            image = read_image(img.path)
            if image is None:
                continue
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy().flatten()
            embeddings[img.image_id] = emb

        groups: list[list[str]] = []
        handled: set[str] = set()
        ids = list(embeddings.keys())
        for i in range(len(ids)):
            if ids[i] in handled:
                continue
            group: list[str] = [ids[i]]
            for j in range(i + 1, len(ids)):
                sim = float(np.dot(embeddings[ids[i]], embeddings[ids[j]]))
                norm_val = np.linalg.norm(embeddings[ids[i]]) * np.linalg.norm(embeddings[ids[j]])
                if norm_val > 0:
                    sim = float(sim / norm_val)
                if sim >= threshold:
                    group.append(ids[j])
            if len(group) > 1:
                handled.update(group)
                groups.append(group)
        return groups
    except Exception:
        return []


def _estimate_savings(groups: list[DuplicateGroup]) -> str:
    total_redundant = sum(len(g.image_ids) - 1 for g in groups)
    if total_redundant == 0:
        return "0 MB"
    est_mb = total_redundant * 0.5
    if est_mb < 1:
        return f"{int(est_mb * 1024)} KB"
    return f"{est_mb:.1f} MB"
