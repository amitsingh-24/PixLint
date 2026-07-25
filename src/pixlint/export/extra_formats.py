from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Any

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

_WDS_AVAILABLE = False
try:
    import webdataset as wds
    _WDS_AVAILABLE = True
except ImportError:
    pass

_FIFTYONE_AVAILABLE = False
try:
    import fiftyone as fo  # noqa: F401
    _FIFTYONE_AVAILABLE = True
except ImportError:
    pass


def export_webdataset(
    dataset: CVDataset,
    output_dir: str,
    image_size: tuple[int, int] = (640, 640),
    shard_size: int = 1000,
    compression: str | None = None,
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    export_count = 0
    ann_count = 0
    class_names = dataset.get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    shard_idx = 0
    writer = None

    try:
        for i, img in enumerate(dataset.images):
            if i % shard_size == 0:
                if writer is not None:
                    writer.close()
                shard_idx += 1
                pattern = os.path.join(output_dir, f"shard-{shard_idx:06d}.tar")
                if _WDS_AVAILABLE:
                    writer = wds.TarWriter(pattern, compress=compression is not None)
                else:
                    writer = _SimpleTarWriter(pattern)

            image = cv2.imread(img.path)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            _, img_bytes = cv2.imencode(".jpg", image)

            yolo_anns = []
            h, w = image.shape[:2]
            for ann in img.annotations:
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    cx = (x1 + x2) / (2.0 * w)
                    cy = (y1 + y2) / (2.0 * h)
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    label_idx = class_to_idx.get(ann.label, 0)
                    yolo_anns.append(f"{label_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            key = img.image_id.replace("/", "_")
            sample = {
                "__key__": key,
                # cv2.imencode returns an ndarray; the tar writers need raw bytes.
                "jpg": img_bytes.tobytes(),
                "cls": " ".join(ann.label for ann in img.annotations),
            }
            if yolo_anns:
                sample["txt"] = "\n".join(yolo_anns).encode()

            if writer is not None:
                writer.write(sample)
            export_count += 1
            ann_count += len(img.annotations)

        if writer is not None:
            writer.close()
    except Exception:
        if writer is not None:
            writer.close()
        raise

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="webdataset",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def export_fiftyone(
    dataset: CVDataset,
    output_dir: str,
    export_format: str = "coco",
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    export_count = 0
    ann_count = 0

    if not _FIFTYONE_AVAILABLE:
        return _export_fiftyone_json(dataset, output_dir)

    import fiftyone as fo
    samples = []
    for img in dataset.images:
        sample = fo.Sample(filepath=img.path)
        if img.annotations:
            detections = []
            iw = img.width or 1
            ih = img.height or 1
            for ann in img.annotations:
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    bw = abs(x2 - x1)
                    bh = abs(y2 - y1)
                    # FiftyOne expects [top-left-x, top-left-y, w, h] normalized to [0, 1].
                    detections.append(
                        fo.Detection(
                            label=ann.label,
                            bounding_box=[x1 / iw, y1 / ih, bw / iw, bh / ih],
                        )
                    )
            sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset_fo = fo.Dataset(name=dataset.dataset_id)
    dataset_fo.add_samples(samples)
    dataset_fo.export(
        export_dir=output_dir,
        dataset_type=fo.types.COCODetectionDataset if export_format == "coco" else fo.types.VOCDetectionDataset,
    )
    export_count = len(samples)
    ann_count = dataset.get_num_annotations()

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format=f"fiftyone_{export_format}",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def _export_fiftyone_json(dataset: CVDataset, output_dir: str) -> ExportResult:
    coco_data = _build_coco_json(dataset)
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(coco_data, f, indent=2)
    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="fiftyone_json",
        output_dir=output_dir,
        num_images_exported=len(dataset.images),
        num_annotations_exported=dataset.get_num_annotations(),
    )


def export_cvat_xml(
    dataset: CVDataset,
    output_path: str,
) -> ExportResult:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ann_count = 0

    annotations = ET.Element("annotations")
    for img in dataset.images:
        image_el = ET.SubElement(annotations, "image")
        image_el.set("id", img.image_id)
        image_el.set("name", os.path.basename(img.path))
        if img.width:
            image_el.set("width", str(img.width))
        if img.height:
            image_el.set("height", str(img.height))

        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                box_el = ET.SubElement(image_el, "box")
                box_el.set("label", ann.label)
                box_el.set("xtl", f"{x1:.2f}")
                box_el.set("ytl", f"{y1:.2f}")
                box_el.set("xbr", f"{x2:.2f}")
                box_el.set("ybr", f"{y2:.2f}")
                box_el.set("occluded", "0")
                ann_count += 1

    tree = ET.ElementTree(annotations)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="cvat_xml",
        output_dir=os.path.dirname(output_path) or ".",
        num_images_exported=len(dataset.images),
        num_annotations_exported=ann_count,
    )


def export_labelme_json(
    dataset: CVDataset,
    output_dir: str,
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    export_count = 0
    ann_count = 0

    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        h, w = image.shape[:2]

        shapes: list[dict[str, Any]] = []
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                shapes.append({
                    "label": ann.label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                })
                ann_count += 1

        labelme_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img.path),
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }

        stem = os.path.splitext(os.path.basename(img.path))[0]
        with open(os.path.join(output_dir, f"{stem}.json"), "w") as f:
            json.dump(labelme_data, f, indent=2)
        export_count += 1

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="labelme_json",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def _build_coco_json(dataset: CVDataset) -> dict[str, Any]:
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(dataset.get_class_names())
    ]
    class_to_id = {cat["name"]: cat["id"] for cat in categories}

    images_list = []
    annotations_list = []
    ann_id = 1

    for img in dataset.images:
        images_list.append({
            "id": img.image_id,
            "file_name": os.path.basename(img.path),
            "width": img.width or 0,
            "height": img.height or 0,
        })
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                annotations_list.append({
                    "id": ann_id,
                    "image_id": img.image_id,
                    "category_id": class_to_id.get(ann.label, 0),
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

    return {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }


class _SimpleTarWriter:
    def __init__(self, path: str):
        import tarfile
        self.tar = tarfile.open(path, "w")

    def write(self, sample: dict[str, Any]) -> None:
        import io
        import tarfile
        key = sample["__key__"]
        for suffix, data in sample.items():
            if suffix == "__key__":
                continue
            if isinstance(data, str):
                data = data.encode("utf-8")
            if isinstance(data, bytes):
                info = tarfile.TarInfo(name=f"{key}.{suffix}")
                info.size = len(data)
                self.tar.addfile(info, io.BytesIO(data))

    def close(self) -> None:
        self.tar.close()
