from __future__ import annotations

import os

import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import NormalizeResult


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_dataset(
    dataset: CVDataset,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    output_dir: str | None = None,
    inplace: bool = False,
) -> NormalizeResult:
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    if output_dir is None and not inplace:
        output_dir = os.path.join(os.path.dirname(dataset.path), f"{dataset.name}_normalized")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    for img_record in dataset.images:
        image = cv2.imread(img_record.path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (image_rgb - mean_arr) / std_arr
        normalized = np.clip(normalized, -3, 3).astype(np.float32)

        # Normalized data contains negative floats and is NOT a valid 8-bit
        # image — writing it to .jpg/.png silently fails. Persist the actual
        # normalized array as .npy (the artifact a training pipeline consumes).
        if output_dir:
            out_path = os.path.join(output_dir, f"{img_record.image_id}.npy")
            np.save(out_path, normalized)
        elif inplace:
            base, _ = os.path.splitext(img_record.path)
            np.save(base + ".npy", normalized)

        count += 1

    return NormalizeResult(
        dataset_id=dataset.dataset_id,
        mean=mean,
        std=std,
        num_images_normalized=count,
    )


def compute_channel_stats(
    dataset: CVDataset,
) -> dict[str, list[float] | list[list[float]] | int]:
    all_pixels: list[np.ndarray] = []
    sample_count = 0
    max_samples = min(100, len(dataset))

    for img_record in dataset.images[:max_samples]:
        image = cv2.imread(img_record.path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_pixels.append(image_rgb.reshape(-1, 3).astype(np.float32) / 255.0)
        sample_count += 1

    if not all_pixels:
        return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "per_channel_mean": [[0.0], [0.0], [0.0]]}

    pixels = np.vstack(all_pixels)
    mean = pixels.mean(axis=0).tolist()
    std = pixels.std(axis=0).tolist()

    return {
        "mean": [round(m, 4) for m in mean],
        "std": [round(s, 4) for s in std],
        "samples_used": sample_count,
    }


def normalize_image(
    image: np.ndarray,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    image_float = image.astype(np.float32) / 255.0
    normalized = (image_float - mean_arr) / std_arr
    return np.clip(normalized, -3, 3)
