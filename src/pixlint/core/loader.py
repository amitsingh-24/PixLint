from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as DefusedET

from pixlint.utils.image_io import get_image_size, is_supported_image
from pixlint.utils.schemas import (
    Annotation,
    CVDatasetInfo,
    DatasetFormat,
    ImageRecord,
)

_YAML_AVAILABLE = False
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    pass


class CVDataset:
    def __init__(
        self,
        path: str,
        format: DatasetFormat = DatasetFormat.AUTO,
        recursive: bool = True,
        name: str | None = None,
    ):
        self.path = Path(path).resolve()
        self.recursive = recursive
        self._format = format if format != DatasetFormat.AUTO else self._detect_format()
        self._name = name or self.path.name
        try:
            ino = self.path.stat().st_ino
            self._dataset_id = str(self.path.stem) + "_" + str(abs(ino) % 100000)
        except (OSError, AttributeError):
            self._dataset_id = str(self.path.stem) + "_" + str(abs(id(self.path)) % 100000)
        self._images: list[ImageRecord] = []
        self._load()

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def format(self) -> DatasetFormat:
        return self._format

    @property
    def images(self) -> list[ImageRecord]:
        return self._images

    def _detect_format(self) -> DatasetFormat:
        str(self.path)
        if not self.path.is_dir():
            if self.path.suffix.lower() in (".json",):
                return DatasetFormat.COCO
            return DatasetFormat.FOLDER

        ann_dir = self.path / "annotations"
        if ann_dir.exists() and any(ann_dir.glob("*.json")):
            return DatasetFormat.COCO

        voc_ann = self.path / "Annotations"
        if voc_ann.exists() and any(voc_ann.glob("*.xml")):
            return DatasetFormat.VOC

        labels_dir = self.path / "labels"
        if labels_dir.exists() and any(labels_dir.glob("*.txt")):
            return DatasetFormat.YOLO

        label_2 = self.path / "label_2"
        if label_2.exists() and any(label_2.glob("*.txt")):
            return DatasetFormat.KITTI

        images_dir = self.path / "images"
        if images_dir.exists():
            return DatasetFormat.YOLO

        return DatasetFormat.FOLDER

    def _load(self) -> None:
        loaders = {
            DatasetFormat.COCO: self._load_coco,
            DatasetFormat.VOC: self._load_voc,
            DatasetFormat.YOLO: self._load_yolo,
            DatasetFormat.KITTI: self._load_kitti,
            DatasetFormat.FOLDER: self._load_folder,
        }
        loader = loaders.get(self._format, self._load_folder)
        loader()

    def _load_folder(self) -> None:
        pattern = "**/*" if self.recursive else "*"
        for img_path in sorted(self.path.glob(pattern)):
            if not img_path.is_file() or not is_supported_image(str(img_path)):
                continue
            size = get_image_size(str(img_path))
            record = ImageRecord(
                image_id=img_path.stem,
                path=str(img_path),
                width=size[0] if size else None,
                height=size[1] if size else None,
            )
            self._images.append(record)

    def _load_coco(self) -> None:
        # Prefer annotation JSONs under an `annotations/` dir, sorted for
        # deterministic selection (real COCO layouts ship several *.json files,
        # e.g. instances_train + instances_val + captions).
        ann_dir = self.path / "annotations"
        json_files = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []
        if not json_files:
            json_files = sorted(self.path.glob("**/*.json"))
        if not json_files:
            self._load_folder()
            return

        json_file = json_files[0]
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._images = []
            return

        categories = {
            cat["id"]: cat.get("name", "unknown")
            for cat in data.get("categories", [])
            if cat.get("id") is not None
        }

        images_map = {}
        for img in data.get("images", []):
            img_id = img.get("id")
            if img_id is None:
                continue
            file_name = img.get("file_name", "")
            img_path = self.path / file_name
            if not img_path.exists():
                img_path = self.path / "images" / file_name
            if not img_path.exists():
                img_path = self.path / ".." / file_name
            record = ImageRecord(
                image_id=str(img_id),
                path=str(img_path.resolve()) if img_path.exists() else str(self.path / file_name),
                width=img.get("width"),
                height=img.get("height"),
            )
            images_map[img_id] = record

        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_id = ann.get("category_id")
            if img_id is None or cat_id is None:
                continue
            rec = images_map.get(img_id)
            if rec is None:
                continue
            cat_name = categories.get(cat_id, "unknown")
            # COCO stores bbox as [x, y, width, height]; normalize to the
            # (x1, y1, x2, y2) convention every other loader/consumer uses.
            raw_bbox = ann.get("bbox")
            bbox = None
            if raw_bbox and len(raw_bbox) == 4:
                try:
                    x, y, bw, bh = (float(v) for v in raw_bbox)
                    bbox = (x, y, x + bw, y + bh)
                except (TypeError, ValueError):
                    bbox = None
            annotation = Annotation(
                label=cat_name,
                bbox=bbox,
                area=ann.get("area"),
                iscrowd=bool(ann.get("iscrowd", False)),
                segmentation=ann.get("segmentation"),
            )
            rec.annotations.append(annotation)

        self._images = sorted(images_map.values(), key=lambda r: r.image_id)

    def _load_voc(self) -> None:
        ann_dir = self.path / "Annotations"
        if not ann_dir.exists():
            self._load_folder()
            return

        def _num(el: Any, tag: str, default: float = 0.0) -> float:
            # ElementTree.findtext returns "" (not the default) for a present-but-
            # empty element, so int("")/float("500.0") would raise. Parse safely.
            txt = (el.findtext(tag, "") or "").strip()
            try:
                return float(txt)
            except ValueError:
                return default

        for xml_file in sorted(ann_dir.glob("*.xml")):
            try:
                tree = DefusedET.parse(xml_file)
                root = tree.getroot()
            except (ET.ParseError, OSError, ValueError):
                continue
            filename = root.findtext("filename", "") or ""
            img_path = self.path / "JPEGImages" / filename
            if not img_path.exists():
                img_path = self.path / filename
            if not img_path.exists():
                img_path = xml_file.parent / ".." / "JPEGImages" / filename
            size_el = root.find("size")
            width = int(_num(size_el, "width", 0)) if size_el is not None else 0
            height = int(_num(size_el, "height", 0)) if size_el is not None else 0
            record = ImageRecord(
                image_id=xml_file.stem,
                path=str(img_path.resolve()) if img_path.exists() else str(self.path / filename),
                width=width or None,
                height=height or None,
            )
            for obj in root.findall("object"):
                name = obj.findtext("name", "unknown") or "unknown"
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    bbox = (
                        _num(bndbox, "xmin"), _num(bndbox, "ymin"),
                        _num(bndbox, "xmax"), _num(bndbox, "ymax"),
                    )
                else:
                    bbox = None
                record.annotations.append(Annotation(label=name, bbox=bbox))
            self._images.append(record)

    def _load_yolo(self) -> None:
        labels_dir = self.path / "labels"
        images_dir = self.path / "images"
        if not labels_dir.exists() or not images_dir.exists():
            self._load_folder()
            return

        names_file = self.path / "dataset.yaml"
        class_names: dict[int, str] = {}
        if names_file.exists() and _YAML_AVAILABLE:
            try:
                with open(names_file) as f:
                    data = yaml.safe_load(f) or {}
                names = data.get("names", {})
                if isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
            except (yaml.YAMLError, ValueError, OSError):
                class_names = {}

        # Index images by stem once (O(n)) instead of rescanning the dir per label.
        img_by_stem = {p.stem: p for p in sorted(images_dir.iterdir()) if p.is_file()}

        for label_file in sorted(labels_dir.iterdir()):
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            img_path = img_by_stem.get(label_file.stem)
            if img_path is None:
                continue
            size = get_image_size(str(img_path))
            img_w = size[0] if size else None
            img_h = size[1] if size else None
            record = ImageRecord(
                image_id=label_file.stem,
                path=str(img_path),
                width=img_w,
                height=img_h,
            )
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        coords = [float(p) for p in parts[1:]]
                    except ValueError:
                        continue
                    label = class_names.get(cls_id, f"class_{cls_id}")
                    segmentation = None
                    if len(coords) > 4 and len(coords) % 2 == 0:
                        # YOLO-seg polygon row: cls x1 y1 x2 y2 ... (normalized).
                        xs, ys = coords[0::2], coords[1::2]
                        nx1, ny1, nx2, ny2 = min(xs), min(ys), max(xs), max(ys)
                        if img_w and img_h:
                            bbox = (nx1 * img_w, ny1 * img_h, nx2 * img_w, ny2 * img_h)
                            segmentation = [[
                                coords[i] * (img_w if i % 2 == 0 else img_h)
                                for i in range(len(coords))
                            ]]
                        else:
                            bbox = (nx1, ny1, nx2, ny2)
                            segmentation = [list(coords)]
                    else:
                        cx, cy, w_norm, h_norm = coords[:4]
                        if img_w and img_h:
                            abs_w, abs_h = w_norm * img_w, h_norm * img_h
                            x1 = cx * img_w - abs_w / 2
                            y1 = cy * img_h - abs_h / 2
                            bbox = (x1, y1, x1 + abs_w, y1 + abs_h)
                        else:
                            bbox = (cx, cy, w_norm, h_norm)
                    record.annotations.append(
                        Annotation(label=label, bbox=bbox, segmentation=segmentation)
                    )
            self._images.append(record)

    def _load_kitti(self) -> None:
        label_dir = self.path / "label_2"
        image_dir = self.path / "image_2"
        if not label_dir.exists() or not image_dir.exists():
            self._load_folder()
            return

        img_by_stem = {p.stem: p for p in sorted(image_dir.iterdir()) if p.is_file()}

        for label_file in sorted(label_dir.iterdir()):
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            img_path = img_by_stem.get(label_file.stem)
            if img_path is None:
                continue
            size = get_image_size(str(img_path))
            record = ImageRecord(
                image_id=label_file.stem,
                path=str(img_path),
                width=size[0] if size else None,
                height=size[1] if size else None,
            )
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    label = parts[0]
                    # 'DontCare'/'Misc' are ignore-regions, not real objects —
                    # exclude them so they don't pollute class lists/stats.
                    if label in ("DontCare", "Misc"):
                        continue
                    try:
                        x1, y1, x2, y2 = (float(v) for v in parts[4:8])
                    except ValueError:
                        continue
                    record.annotations.append(Annotation(label=label, bbox=(x1, y1, x2, y2)))
            self._images.append(record)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> ImageRecord:
        return self._images[idx]

    def __iter__(self):
        return iter(self._images)

    def get_class_names(self) -> list[str]:
        names: set[str] = set()
        for img in self._images:
            for ann in img.annotations:
                names.add(ann.label)
        return sorted(names)

    def get_num_annotations(self) -> int:
        return sum(len(img.annotations) for img in self._images)

    def to_info(self) -> CVDatasetInfo:
        return CVDatasetInfo(
            dataset_id=self._dataset_id,
            name=self._name,
            path=str(self.path),
            format=self._format,
            num_images=len(self._images),
            num_annotations=self.get_num_annotations(),
            class_names=self.get_class_names(),
        )


def load_dataset(
    path: str,
    format: str = "auto",
    recursive: bool = True,
    name: str | None = None,
) -> CVDataset:
    fmt = DatasetFormat(format)
    dataset = CVDataset(path=path, format=fmt, recursive=recursive, name=name)
    return dataset


def detect_format(path: str) -> str:
    dummy = CVDataset(path, format=DatasetFormat.AUTO, recursive=False)
    return dummy.format.value
