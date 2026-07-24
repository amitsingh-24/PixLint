from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

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
        json_files = list(self.path.glob("**/*.json"))
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

        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        images_map = {}
        for img in data.get("images", []):
            img_path = self.path / img.get("file_name", "")
            if not img_path.exists():
                img_path = self.path / "images" / img.get("file_name", "")
            if not img_path.exists():
                img_path = self.path / ".." / img.get("file_name", "")
            record = ImageRecord(
                image_id=str(img["id"]),
                path=str(img_path.resolve()) if img_path.exists() else str(self.path / img.get("file_name", "")),
                width=img.get("width"),
                height=img.get("height"),
            )
            images_map[img["id"]] = record

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            record = images_map.get(img_id)
            if record is None:
                continue
            cat_name = categories.get(ann["category_id"], "unknown")
            bbox = ann.get("bbox")
            if bbox:
                bbox = tuple(bbox)
            annotation = Annotation(
                label=cat_name,
                bbox=bbox,
                area=ann.get("area"),
                iscrowd=ann.get("iscrowd", False),
                segmentation=ann.get("segmentation"),
            )
            record.annotations.append(annotation)

        self._images = sorted(images_map.values(), key=lambda r: r.image_id)

    def _load_voc(self) -> None:
        ann_dir = self.path / "Annotations"
        if not ann_dir.exists():
            self._load_folder()
            return

        for xml_file in sorted(ann_dir.glob("*.xml")):
            try:
                tree = DefusedET.parse(xml_file)
                root = tree.getroot()
            except (ET.ParseError, OSError):
                continue
            filename = root.findtext("filename", "")
            img_path = self.path / "JPEGImages" / filename
            if not img_path.exists():
                img_path = self.path / filename
            if not img_path.exists():
                img_path = xml_file.parent / ".." / "JPEGImages" / filename
            size_el = root.find("size")
            width = int(size_el.findtext("width", "0")) if size_el is not None else None
            height = int(size_el.findtext("height", "0")) if size_el is not None else None
            record = ImageRecord(
                image_id=xml_file.stem,
                path=str(img_path.resolve()) if img_path.exists() else str(self.path / filename),
                width=width,
                height=height,
            )
            for obj in root.findall("object"):
                name = obj.findtext("name", "unknown")
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.findtext("xmin", "0"))
                    ymin = float(bndbox.findtext("ymin", "0"))
                    xmax = float(bndbox.findtext("xmax", "0"))
                    ymax = float(bndbox.findtext("ymax", "0"))
                    bbox = (xmin, ymin, xmax, ymax)
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
        class_names = {}
        if names_file.exists() and _YAML_AVAILABLE:
            with open(names_file) as f:
                data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
                else:
                    class_names = {int(k): v for k, v in names.items()}

        for label_file in sorted(labels_dir.iterdir()):
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            stem = label_file.stem
            img_candidates = [p for p in images_dir.iterdir() if p.stem == stem and p.is_file()]
            if not img_candidates:
                continue
            img_path = img_candidates[0]
            size = get_image_size(str(img_path))
            img_w = size[0] if size else None
            img_h = size[1] if size else None
            record = ImageRecord(
                image_id=stem,
                path=str(img_path),
                width=img_w,
                height=img_h,
            )
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        label = class_names.get(cls_id, f"class_{cls_id}")
                        cx, cy, w_norm, h_norm = map(float, parts[1:5])
                        if img_w and img_h:
                            abs_w = w_norm * img_w
                            abs_h = h_norm * img_h
                            x1 = cx * img_w - abs_w / 2
                            y1 = cy * img_h - abs_h / 2
                            x2 = x1 + abs_w
                            y2 = y1 + abs_h
                            bbox = (x1, y1, x2, y2)
                        else:
                            bbox = (cx, cy, w_norm, h_norm)
                        record.annotations.append(Annotation(label=label, bbox=bbox))
            self._images.append(record)

    def _load_kitti(self) -> None:
        label_dir = self.path / "label_2"
        image_dir = self.path / "image_2"
        if not label_dir.exists() or not image_dir.exists():
            self._load_folder()
            return

        for label_file in sorted(label_dir.iterdir()):
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            stem = label_file.stem
            img_candidates = [p for p in image_dir.iterdir() if p.stem == stem and p.is_file()]
            if not img_candidates:
                continue
            img_path = img_candidates[0]
            size = get_image_size(str(img_path))
            record = ImageRecord(
                image_id=stem,
                path=str(img_path),
                width=size[0] if size else None,
                height=size[1] if size else None,
            )
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        label = parts[0]
                        h, w, l_dim = map(float, parts[8:11])
                        x1, y1, x2, y2 = map(float, parts[4:8])
                        bbox = (x1, y1, x2, y2)
                        record.annotations.append(Annotation(label=label, bbox=bbox))
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
