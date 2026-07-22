from __future__ import annotations

import os

import numpy as np

from pixlint.transformation.format_converter import convert_format
from pixlint.transformation.resize import resize_dataset, resize_image, _letterbox
from pixlint.transformation.normalize import normalize_dataset, compute_channel_stats, normalize_image


class TestFormatConverter:
    def test_convert_to_voc(self, coco_dataset, tmp_path):
        result = convert_format(coco_dataset, target_format="voc", output_dir=str(tmp_path / "voc"))
        assert result.source_format == "coco"
        assert result.target_format == "voc"
        assert result.num_images_converted > 0
        assert os.path.isdir(os.path.join(str(tmp_path / "voc"), "Annotations"))
        assert os.path.isdir(os.path.join(str(tmp_path / "voc"), "JPEGImages"))

    def test_convert_to_yolo(self, coco_dataset, tmp_path):
        result = convert_format(coco_dataset, target_format="yolo", output_dir=str(tmp_path / "yolo"))
        assert result.target_format == "yolo"
        assert result.num_images_converted > 0

    def test_convert_to_csv(self, coco_dataset, tmp_path):
        result = convert_format(coco_dataset, target_format="csv", output_dir=str(tmp_path / "csv"))
        assert result.target_format == "csv"
        assert result.num_images_converted > 0

    def test_convert_from_voc(self, voc_dataset, tmp_path):
        result = convert_format(voc_dataset, target_format="coco", output_dir=str(tmp_path / "voc2coco"))
        assert result.num_images_converted > 0

    def test_convert_from_yolo(self, yolo_dataset, tmp_path):
        result = convert_format(yolo_dataset, target_format="voc", output_dir=str(tmp_path / "yolo2voc"))
        assert result.num_images_converted > 0


class TestResize:
    def test_resize_letterbox(self, folder_dataset, tmp_path):
        result = resize_dataset(folder_dataset, size=(224, 224), strategy="letterbox", output_dir=str(tmp_path / "resized"))
        assert result.num_images_resized > 0
        assert result.target_size == (224, 224)
        assert result.strategy == "letterbox"

    def test_resize_stretch(self, folder_dataset, tmp_path):
        result = resize_dataset(folder_dataset, size=(224, 224), strategy="stretch", output_dir=str(tmp_path / "stretch"))
        assert result.num_images_resized > 0

    def test_resize_single_image(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        resized = resize_image(img, (100, 100), strategy="letterbox")
        assert resized.shape[:2] == (100, 100)

    def test_letterbox_function(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result, x_off, y_off, scale = _letterbox(img, 224, 224)
        assert result.shape[:2] == (224, 224)


class TestNormalize:
    def test_normalize_dataset(self, folder_dataset, tmp_path):
        result = normalize_dataset(folder_dataset, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], output_dir=str(tmp_path / "norm"))
        assert result.num_images_normalized > 0
        assert result.mean == [0.5, 0.5, 0.5]
        assert result.std == [0.5, 0.5, 0.5]

    def test_normalize_default(self, folder_dataset, tmp_path):
        result = normalize_dataset(folder_dataset, output_dir=str(tmp_path / "norm2"))
        assert result.num_images_normalized > 0

    def test_channel_stats(self, folder_dataset):
        stats = compute_channel_stats(folder_dataset)
        assert "mean" in stats
        assert "std" in stats
        assert len(stats["mean"]) == 3
        assert len(stats["std"]) == 3

    def test_normalize_single_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert normalized.shape == img.shape
