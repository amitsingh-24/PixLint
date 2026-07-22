from __future__ import annotations

import os

import numpy as np

from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.augmentation.transforms import (
    apply_cutmix,
    apply_mixup,
    apply_mosaic,
    apply_copypaste,
)
from pixlint.augmentation.auto_augment import (
    apply_randaugment,
    apply_trivialaugment,
    apply_autoaugment,
)


class TestAugmentationPipeline:
    def test_augment_dataset_structure(self, folder_dataset, tmp_path):
        result = augment_dataset(folder_dataset, pipeline="classification", multiplier=2, output_dir=str(tmp_path / "aug"))
        assert result.dataset_id == folder_dataset.dataset_id
        assert result.pipeline == "classification"
        assert result.multiplier == 2
        assert result.total_generated > 0
        assert os.path.isdir(result.output_dir)

    def test_preview_augmentation(self, folder_dataset):
        result = preview_augmentation(folder_dataset, pipeline="yolo_detection", n_samples=3)
        assert result["pipeline"] == "yolo_detection"
        assert result["n_samples"] <= 3

    def test_augment_default_pipeline(self, folder_dataset, tmp_path):
        result = augment_dataset(folder_dataset, output_dir=str(tmp_path / "aug2"))
        assert result.total_generated > 0


class TestCutMix:
    def test_cutmix(self):
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mixed, bboxes, labels = apply_cutmix(
            img1, img2,
            [[10, 10, 50, 50]], [[20, 20, 60, 60]],
            ["cat"], ["dog"],
        )
        assert mixed.shape == (100, 100, 3)
        assert isinstance(bboxes, list)
        assert isinstance(labels, list)


class TestMixUp:
    def test_mixup(self):
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mixed, bboxes, labels = apply_mixup(
            img1, img2,
            [[10, 10, 50, 50]], [[20, 20, 60, 60]],
            ["cat"], ["dog"],
        )
        assert mixed.shape == (100, 100, 3)
        assert len(bboxes) == 2
        assert len(labels) == 2


class TestMosaic:
    def test_mosaic(self):
        imgs = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
        bboxes_list = [[[10, 10, 50, 50]] for _ in range(4)]
        labels_list = [["cat"] for _ in range(4)]
        mosaic, bboxes, labels = apply_mosaic(imgs, bboxes_list, labels_list, (200, 200))
        assert mosaic.shape == (200, 200, 3)
        assert isinstance(bboxes, list)
        assert isinstance(labels, list)


class TestCopyPaste:
    def test_copypaste(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        src_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result, bboxes, labels = apply_copypaste(
            img, src_img,
            [[10, 10, 30, 30]], ["cat"],
            [[50, 50, 80, 80]], ["dog"],
        )
        assert result.shape == (100, 100, 3)


class TestAutoAugment:
    def test_randaugment(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_randaugment(img, n_transforms=2, magnitude=9.0)
        assert result.shape == img.shape

    def test_trivialaugment(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_trivialaugment(img)
        assert result.shape == img.shape

    def test_autoaugment(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_autoaugment(img, policy="imagenet")
        assert result.shape == img.shape
