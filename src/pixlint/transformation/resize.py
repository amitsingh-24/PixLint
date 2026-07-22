from __future__ import annotations

import os

import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ResizeResult


def resize_dataset(
    dataset: CVDataset,
    size: tuple[int, int],
    strategy: str = "letterbox",
    output_dir: str | None = None,
    preserve_annotations: bool = True,
) -> ResizeResult:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset.path), f"{dataset.name}_resized")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    target_w, target_h = size

    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]

        if strategy == "letterbox":
            resized, x_offset, y_offset, scale = _letterbox(image, target_w, target_h)
        elif strategy == "stretch":
            resized = cv2.resize(image, (target_w, target_h))
            scale = 1.0
            x_offset, y_offset = 0, 0
        elif strategy == "crop":
            resized, x_offset, y_offset, scale = _center_crop(image, target_w, target_h)
        else:
            resized = cv2.resize(image, (target_w, target_h))
            x_offset, y_offset, scale = 0, 0, 1.0

        out_path = os.path.join(output_dir, f"{img.image_id}.jpg")
        cv2.imwrite(out_path, resized)

        if preserve_annotations and img.annotations and (x_offset != 0 or y_offset != 0 or scale != 1.0):
            _adjust_annotations(img, x_offset, y_offset, scale, target_w, target_h)

        count += 1

    return ResizeResult(
        dataset_id=dataset.dataset_id,
        target_size=(target_w, target_h),
        strategy=strategy,
        num_images_resized=count,
    )


def _letterbox(
    image: np.ndarray, target_w: int, target_h: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, int, int, float]:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas, x_offset, y_offset, scale


def _center_crop(
    image: np.ndarray, target_w: int, target_h: int,
) -> tuple[np.ndarray, int, int, float]:
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start : y_start + target_h, x_start : x_start + target_w]

    return cropped, -x_start, -y_start, scale


def _adjust_annotations(
    img, x_offset: int, y_offset: int, scale: float,
    target_w: int, target_h: int,
) -> None:
    for ann in img.annotations:
        if ann.bbox is None:
            continue
        x1, y1, x2, y2 = ann.bbox
        x1 = x1 * scale + x_offset
        y1 = y1 * scale + y_offset
        x2 = x2 * scale + x_offset
        y2 = y2 * scale + y_offset
        x1 = max(0, min(target_w, x1))
        y1 = max(0, min(target_h, y1))
        x2 = max(0, min(target_w, x2))
        y2 = max(0, min(target_h, y2))
        ann.bbox = (x1, y1, x2, y2)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
    strategy: str = "letterbox",
) -> np.ndarray:
    target_w, target_h = size
    if strategy == "letterbox":
        resized, _, _, _ = _letterbox(image, target_w, target_h)
    elif strategy == "crop":
        resized, _, _, _ = _center_crop(image, target_w, target_h)
    else:
        resized = cv2.resize(image, (target_w, target_h))
    return resized
