from __future__ import annotations

import csv
import json
import os
from typing import Any

import cv2
import numpy as np
import xml.etree.ElementTree as ET

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import FormatConversionResult


CONVERSION_MATRIX: dict[str, list[str]] = {
    "coco": ["voc", "yolo", "kitti", "csv"],
    "voc": ["coco", "yolo", "kitti", "csv"],
    "yolo": ["coco", "voc", "kitti", "csv"],
    "kitti": ["coco", "voc", "yolo", "csv"],
    "csv": ["coco", "voc", "yolo", "kitti"],
}


def convert_format(
    dataset: CVDataset,
    target_format: str,
    output_dir: str,
    image_size: tuple[int, int] | None = None,
    preserve_structure: bool = True,
    class_mapping: dict[str, str] | None = None,
) -> FormatConversionResult:
    os.makedirs(output_dir, exist_ok=True)

    source_format = dataset.format.value
    if target_format not in CONVERSION_MATRIX.get(source_format, []):
        raise ValueError(f"Conversion from {source_format} to {target_format} not supported")

    converters = {
        "voc": _convert_to_voc,
        "coco": _convert_to_coco,
        "yolo": _convert_to_yolo,
        "kitti": _convert_to_kitti,
        "csv": _convert_to_csv,
    }

    converter = converters.get(target_format)
    if converter is None:
        raise ValueError(f"Unknown target format: {target_format}")

    num_converted, class_map = converter(dataset, output_dir, image_size, class_mapping or {})

    return FormatConversionResult(
        dataset_id=dataset.dataset_id,
        source_format=source_format,
        target_format=target_format,
        output_dir=output_dir,
        num_images_converted=num_converted,
        class_mapping=class_map or None,
    )


def _get_class_id_map(
    dataset: CVDataset, user_mapping: dict[str, str]
) -> dict[str, int]:
    class_names = dataset.get_class_names()
    if user_mapping:
        mapped = {user_mapping.get(c, c): i for i, c in enumerate(class_names)}
        return mapped

    return {c: i for i, c in enumerate(class_names)}


def _save_image(image: np.ndarray, output_dir: str, filename: str) -> None:
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _convert_to_voc(
    dataset: CVDataset, output_dir: str,
    image_size: tuple[int, int] | None,
    user_mapping: dict[str, str],
) -> tuple[int, dict[str, str]]:
    ann_dir = os.path.join(output_dir, "Annotations")
    jpg_dir = os.path.join(output_dir, "JPEGImages")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    count = 0
    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]
        sx, sy = 1.0, 1.0
        if image_size:
            sx, sy = image_size[0] / w, image_size[1] / h
            image = cv2.resize(image, image_size)
            w, h = image_size

        filename = f"{img.image_id}.jpg"
        cv2.imwrite(os.path.join(jpg_dir, filename), image)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = os.path.basename(output_dir)
        ET.SubElement(annotation, "filename").text = filename
        size_el = ET.SubElement(annotation, "size")
        ET.SubElement(size_el, "width").text = str(w)
        ET.SubElement(size_el, "height").text = str(h)

        for ann in img.annotations:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = user_mapping.get(ann.label, ann.label)
            bndbox = ET.SubElement(obj, "bndbox")
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                ET.SubElement(bndbox, "xmin").text = str(int(x1))
                ET.SubElement(bndbox, "ymin").text = str(int(y1))
                ET.SubElement(bndbox, "xmax").text = str(int(x2))
                ET.SubElement(bndbox, "ymax").text = str(int(y2))

        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(ann_dir, f"{img.image_id}.xml"))
        count += 1

    return count, user_mapping


def _convert_to_coco(
    dataset: CVDataset, output_dir: str,
    image_size: tuple[int, int] | None,
    user_mapping: dict[str, str],
) -> tuple[int, dict[str, str]]:
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    class_id_map = _get_class_id_map(dataset, user_mapping)
    class_list = sorted(class_id_map.keys(), key=lambda c: class_id_map[c])

    coco_data: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_list)],
    }

    ann_id = 1
    count = 0

    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]
        sx, sy = 1.0, 1.0
        if image_size:
            sx, sy = image_size[0] / w, image_size[1] / h
            image = cv2.resize(image, image_size)
            w, h = image_size

        filename = f"{img.image_id}.jpg"
        cv2.imwrite(os.path.join(images_dir, filename), image)

        img_id = count + 1
        coco_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h,
        })

        for ann in img.annotations:
            cat_name = user_mapping.get(ann.label, ann.label)
            cat_id = class_id_map.get(cat_name, 0)
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                bw = x2 - x1
                bh = y2 - y1
                bbox_coco = [float(x1), float(y1), float(bw), float(bh)]
            else:
                bbox_coco = [0, 0, 10, 10]
                bw, bh = 10, 10

            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox_coco,
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

        count += 1

    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "instances.json"), "w") as f:
        json.dump(coco_data, f, indent=2)

    return count, {name: name for name in class_list}


def _convert_to_yolo(
    dataset: CVDataset, output_dir: str,
    image_size: tuple[int, int] | None,
    user_mapping: dict[str, str],
) -> tuple[int, dict[str, str]]:
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Build the class index from the *mapped* names so a user class_mapping
    # does not collapse every annotation to class 0.
    class_list = sorted(set(
        user_mapping.get(ann.label, ann.label)
        for img in dataset.images for ann in img.annotations
    ))
    class_id_map = {c: i for i, c in enumerate(class_list)}

    names = [c for c in class_list]
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    try:
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump({"names": names, "nc": len(names)}, f)
    except ImportError:
        pass

    count = 0
    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]
        sx, sy = 1.0, 1.0
        if image_size:
            sx, sy = image_size[0] / w, image_size[1] / h
            image = cv2.resize(image, image_size)
            w, h = image_size

        filename = f"{img.image_id}.jpg"
        cv2.imwrite(os.path.join(images_dir, filename), image)

        label_lines: list[str] = []
        for ann in img.annotations:
            cat_name = user_mapping.get(ann.label, ann.label)
            cat_id = class_id_map.get(cat_name, 0)
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                cx = (x1 + x2) / (2.0 * w)
                cy = (y1 + y2) / (2.0 * h)
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                label_lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if label_lines:
            with open(os.path.join(labels_dir, f"{img.image_id}.txt"), "w") as f:
                f.write("\n".join(label_lines))

        count += 1

    return count, {c: c for c in class_list}


def _convert_to_kitti(
    dataset: CVDataset, output_dir: str,
    image_size: tuple[int, int] | None,
    user_mapping: dict[str, str],
) -> tuple[int, dict[str, str]]:
    image_dir = os.path.join(output_dir, "image_2")
    label_dir = os.path.join(output_dir, "label_2")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    count = 0
    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]
        sx, sy = 1.0, 1.0
        if image_size:
            sx, sy = image_size[0] / w, image_size[1] / h
            image = cv2.resize(image, image_size)
            w, h = image_size

        filename = f"{img.image_id}.jpg"
        cv2.imwrite(os.path.join(image_dir, filename), image)

        label_lines: list[str] = []
        for ann in img.annotations:
            cat_name = user_mapping.get(ann.label, ann.label)
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                truncated = 0.0
                occluded = 0
                alpha = 0.0
                dims = f"{h * 0.5:.2f} {w * 0.5:.2f} {w * 0.5:.2f}"
                location = "0.0 0.0 0.0"
                rot_y = 0.0
                line = f"{cat_name} {truncated} {occluded} {alpha} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {dims} {location} {rot_y}"
                label_lines.append(line)

        if label_lines:
            with open(os.path.join(label_dir, f"{img.image_id}.txt"), "w") as f:
                f.write("\n".join(label_lines))

        count += 1

    return count, user_mapping


def _convert_to_csv(
    dataset: CVDataset, output_dir: str,
    image_size: tuple[int, int] | None,
    user_mapping: dict[str, str],
) -> tuple[int, dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    count = 0

    for img in dataset.images:
        for ann in img.annotations:
            cat_name = user_mapping.get(ann.label, ann.label)
            row = {
                "image_id": img.image_id,
                "path": img.path,
                "label": cat_name,
            }
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                row["xmin"] = str(x1)
                row["ymin"] = str(y1)
                row["xmax"] = str(x2)
                row["ymax"] = str(y2)
                row["width"] = str(abs(x2 - x1))
                row["height"] = str(abs(y2 - y1))
            rows.append(row)
            count += 1

    csv_path = os.path.join(output_dir, "annotations.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return count, user_mapping
