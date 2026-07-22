from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image
from pixlint.utils.schemas import AugmentationResult, DatasetFormat

_ALBUMENTATIONS_AVAILABLE = False
try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    pass


PRESET_PIPELINES: dict[str, list[dict[str, Any]]] = {
    "yolo_detection": [
        {"type": "HorizontalFlip", "p": 0.5},
        {"type": "RandomBrightnessContrast", "p": 0.3, "brightness_limit": 0.2, "contrast_limit": 0.2},
        {"type": "HueSaturationValue", "p": 0.3, "hue_shift_limit": 20, "sat_shift_limit": 30, "val_shift_limit": 20},
        {"type": "RandomScale", "p": 0.3, "scale_limit": 0.15},
        {"type": "GaussianNoise", "p": 0.2, "var_limit": (10, 50)},
        {"type": "MotionBlur", "p": 0.2, "blur_limit": 7},
    ],
    "classification": [
        {"type": "HorizontalFlip", "p": 0.5},
        {"type": "RandomBrightnessContrast", "p": 0.3},
        {"type": "Rotate", "p": 0.3, "limit": 30},
        {"type": "ShiftScaleRotate", "p": 0.3, "shift_limit": 0.1, "scale_limit": 0.1, "rotate_limit": 15},
        {"type": "GaussianBlur", "p": 0.1, "blur_limit": (3, 5)},
    ],
    "segmentation": [
        {"type": "HorizontalFlip", "p": 0.5},
        {"type": "ElasticTransform", "p": 0.3, "alpha": 1, "sigma": 50, "alpha_affine": 50},
        {"type": "RandomBrightnessContrast", "p": 0.3},
        {"type": "RandomGamma", "p": 0.2},
        {"type": "GridDistortion", "p": 0.2},
    ],
}


def _get_bbox_format(dataset: CVDataset) -> str:
    fmt_map = {
        DatasetFormat.COCO: "pascal_voc",
        DatasetFormat.VOC: "pascal_voc",
        DatasetFormat.YOLO: "pascal_voc",
        DatasetFormat.KITTI: "pascal_voc",
        DatasetFormat.FOLDER: "pascal_voc",
    }
    return fmt_map.get(dataset.format, "pascal_voc")


def _build_albumentations_pipeline(
    transform_configs: list[dict[str, Any]],
    bbox_format: str = "pascal_voc",
) -> A.Compose | None:
    if not _ALBUMENTATIONS_AVAILABLE:
        return None

    transforms: list[A.DualTransform | A.ImageOnlyTransform] = []
    for cfg in transform_configs:
        t_type = cfg.get("type", "")
        p = cfg.get("p", 0.5)
        params = {k: v for k, v in cfg.items() if k not in ("type", "p")}
        try:
            transform_cls = getattr(A, t_type)
            transforms.append(transform_cls(p=p, **params))
        except (AttributeError, TypeError):
            pass

    if not transforms:
        return None

    bbox_params = A.BboxParams(format=bbox_format, label_fields=["labels"])
    return A.Compose(transforms, bbox_params=bbox_params)


def augment_dataset(
    dataset: CVDataset,
    pipeline: str = "yolo_detection",
    multiplier: int = 3,
    output_dir: str | None = None,
) -> AugmentationResult:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset.path), f"{dataset.name}_augmented")

    os.makedirs(output_dir, exist_ok=True)

    if pipeline == "custom":
        transform_configs = []
    elif pipeline in PRESET_PIPELINES:
        transform_configs = PRESET_PIPELINES[pipeline]
    else:
        transform_configs = PRESET_PIPELINES.get("yolo_detection", [])

    bbox_format = _get_bbox_format(dataset)
    aug_pipeline = _build_albumentations_pipeline(transform_configs, bbox_format=bbox_format)
    total_generated = 0

    for img in dataset.images:
        image = read_image(img.path)
        if image is None:
            continue

        for copy_idx in range(multiplier):
            if aug_pipeline is not None:
                labels = [ann.label for ann in img.annotations]
                bboxes = []
                for ann in img.annotations:
                    if ann.bbox:
                        x1, y1, x2, y2 = ann.bbox
                        bboxes.append([x1, y1, x2, y2])

                try:
                    augmented = aug_pipeline(image=image, bboxes=bboxes, labels=labels)
                    aug_image = augmented["image"]
                except Exception:
                    aug_image = image.copy()
            else:
                aug_image = image.copy()

            stem = Path(img.path).stem
            ext = Path(img.path).suffix or ".jpg"
            out_name = f"{stem}_aug_{copy_idx}{ext}"
            out_path = os.path.join(output_dir, out_name)

            if aug_image.shape[2] == 3:
                out_img = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, out_img)
            else:
                cv2.imwrite(out_path, aug_image)

            total_generated += 1

    return AugmentationResult(
        dataset_id=dataset.dataset_id,
        pipeline=pipeline,
        multiplier=multiplier,
        output_dir=output_dir,
        total_generated=total_generated,
    )


def preview_augmentation(
    dataset: CVDataset,
    pipeline: str = "yolo_detection",
    n_samples: int = 8,
) -> dict[str, Any]:
    transform_configs = PRESET_PIPELINES.get(pipeline, PRESET_PIPELINES["yolo_detection"])
    bbox_format = _get_bbox_format(dataset)
    aug_pipeline = _build_albumentations_pipeline(transform_configs, bbox_format=bbox_format)

    previews: list[dict[str, Any]] = []
    count = 0

    for img in dataset.images:
        if count >= n_samples:
            break
        image = read_image(img.path)
        if image is None:
            continue

        if aug_pipeline is not None:
            labels = [ann.label for ann in img.annotations]
            bboxes = []
            for ann in img.annotations:
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    bboxes.append([x1, y1, x2, y2])
            try:
                augmented = aug_pipeline(image=image, bboxes=bboxes, labels=labels)
                aug_image = augmented["image"]
            except Exception:
                aug_image = image.copy()
        else:
            aug_image = image.copy()

        previews.append({
            "image_id": img.image_id,
            "original_shape": list(image.shape[:2]),
            "augmented_shape": list(aug_image.shape[:2]),
        })
        count += 1

    return {
        "pipeline": pipeline,
        "n_samples": len(previews),
        "samples": previews,
    }
