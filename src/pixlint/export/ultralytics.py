from __future__ import annotations

import os
from typing import Any

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

_YAML_AVAILABLE = False
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    pass


def export_ultralytics(
    dataset: CVDataset,
    output_dir: str,
    yaml_path: str = "dataset.yaml",
    image_size: int = 640,
    split: dict[str, list[str]] | None = None,
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    class_names = dataset.get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    if split:
        splits_to_export = ["train", "val", "test"]
    else:
        splits_to_export = ["train"]

    export_count = 0
    ann_count = 0
    split_stats: dict[str, dict[str, Any]] = {}

    for split_name in splits_to_export:
        imgs_dir = os.path.join(output_dir, split_name, "images")
        lbls_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(imgs_dir, exist_ok=True)
        os.makedirs(lbls_dir, exist_ok=True)

        selected_ids = set(split.get(split_name, [])) if split else {img.image_id for img in dataset.images}
        split_export_count = 0
        split_ann_count = 0

        for img in dataset.images:
            if img.image_id not in selected_ids:
                continue

            image = cv2.imread(img.path)
            if image is None:
                continue
            h, w = image.shape[:2]
            image = cv2.resize(image, (image_size, image_size))
            filename = f"{img.image_id}.jpg"
            cv2.imwrite(os.path.join(imgs_dir, filename), image)

            label_lines: list[str] = []
            for ann in img.annotations:
                label_idx = class_to_idx.get(ann.label, -1)
                if label_idx < 0:
                    continue
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    cx = (x1 + x2) / (2.0 * w)
                    cy = (y1 + y2) / (2.0 * h)
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    label_lines.append(f"{label_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    ann_count += 1
                    split_ann_count += 1

            if label_lines:
                with open(os.path.join(lbls_dir, f"{img.image_id}.txt"), "w") as f:
                    f.write("\n".join(label_lines))

            export_count += 1
            split_export_count += 1

        split_stats[split_name] = {
            "num_images": split_export_count,
            "num_annotations": split_ann_count,
        }

    yaml_data: dict[str, Any] = {
        "names": class_names,
        "nc": len(class_names),
    }

    for split_name in splits_to_export:
        yaml_data[split_name] = os.path.join(output_dir, split_name, "images")

    yaml_out = os.path.join(output_dir, yaml_path)
    if _YAML_AVAILABLE:
        with open(yaml_out, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
    else:
        with open(yaml_out, "w") as f:
            lines = []
            for k, v in yaml_data.items():
                if isinstance(v, list):
                    items = "\n".join(f"  - {item}" for item in v)
                    lines.append(f"{k}:\n{items}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k}: {v}")
                else:
                    lines.append(f'{k}: "{v}"')
            f.write("\n".join(lines))

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="ultralytics",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )
