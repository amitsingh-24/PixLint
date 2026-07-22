from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DatasetFormat


@pytest.fixture
def temp_image_dir(tmp_path):
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(tmp_path / f"image_{i}.jpg")
    return tmp_path


@pytest.fixture
def temp_coco_dir(tmp_path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for i in range(4):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(images_dir / f"img_{i}.jpg")

    coco_data = {
        "images": [
            {"id": 1, "file_name": "img_0.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img_1.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "img_2.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "img_3.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [5, 5, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 2, "bbox": [15, 25, 45, 55], "area": 2475, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
    }
    with open(ann_dir / "instances.json", "w") as f:
        json.dump(coco_data, f)

    return tmp_path


@pytest.fixture
def temp_voc_dir(tmp_path):
    ann_dir = tmp_path / "Annotations"
    ann_dir.mkdir()
    jpg_dir = tmp_path / "JPEGImages"
    jpg_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(jpg_dir / f"img_{i}.jpg")

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = f"img_{i}.jpg"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = "100"
        ET.SubElement(size, "height").text = "100"
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "cat" if i % 2 == 0 else "dog"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "10"
        ET.SubElement(bndbox, "ymin").text = "10"
        ET.SubElement(bndbox, "xmax").text = "50"
        ET.SubElement(bndbox, "ymax").text = "50"
        tree = ET.ElementTree(annotation)
        tree.write(ann_dir / f"img_{i}.xml")

    return tmp_path


@pytest.fixture
def temp_yolo_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    for i in range(4):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(images_dir / f"img_{i}.jpg")

        with open(labels_dir / f"img_{i}.txt", "w") as f:
            f.write(f"{i % 2} 0.5 0.5 0.8 0.8\n")

    yaml_data = {"names": ["cat", "dog"], "nc": 2}
    try:
        import yaml
        with open(tmp_path / "dataset.yaml", "w") as f:
            yaml.dump(yaml_data, f)
    except ImportError:
        pass

    return tmp_path


@pytest.fixture
def folder_dataset(temp_image_dir):
    return CVDataset(str(temp_image_dir), format=DatasetFormat.FOLDER)


@pytest.fixture
def coco_dataset(temp_coco_dir):
    return CVDataset(str(temp_coco_dir), format=DatasetFormat.COCO)


@pytest.fixture
def voc_dataset(temp_voc_dir):
    return CVDataset(str(temp_voc_dir), format=DatasetFormat.VOC)


@pytest.fixture
def yolo_dataset(temp_yolo_dir):
    return CVDataset(str(temp_yolo_dir), format=DatasetFormat.YOLO)
