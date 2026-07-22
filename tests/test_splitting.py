from __future__ import annotations

from pixlint.splitting.splitter import split_dataset, sample_split
from pixlint.splitting.cross_validation import generate_kfold_splits
from pixlint.splitting.leakage import detect_leakage


class TestSplitter:
    def test_random_split(self, folder_dataset):
        result = split_dataset(folder_dataset, strategy="random", ratios={"train": 0.6, "val": 0.2, "test": 0.2})
        assert result.dataset_id == folder_dataset.dataset_id
        assert result.strategy == "random"
        assert len(result.splits["train"]) > 0
        assert len(result.splits["val"]) > 0
        assert len(result.splits["test"]) > 0
        total = sum(len(v) for v in result.splits.values())
        assert total == len(folder_dataset)

    def test_stratified_split(self, coco_dataset):
        result = split_dataset(coco_dataset, strategy="stratified")
        assert result.strategy == "stratified"
        assert len(result.splits["train"]) > 0
        assert result.class_distributions is not None

    def test_temporal_split(self, folder_dataset):
        result = split_dataset(folder_dataset, strategy="temporal")
        assert result.strategy == "temporal"

    def test_split_structure(self, folder_dataset):
        result = split_dataset(folder_dataset)
        assert hasattr(result, "splits")
        assert hasattr(result, "ratios")
        assert hasattr(result, "strategy")
        assert "train" in result.splits
        assert "val" in result.splits
        assert "test" in result.splits

    def test_sample_split(self, folder_dataset):
        result = split_dataset(folder_dataset)
        sampled = sample_split(result, split_name="train", n=3)
        assert len(sampled) <= 3
        for s in sampled:
            assert s in result.splits["train"]

    def test_deterministic(self, folder_dataset):
        r1 = split_dataset(folder_dataset, seed=42)
        r2 = split_dataset(folder_dataset, seed=42)
        assert r1.splits["train"] == r2.splits["train"]


class TestKFold:
    def test_kfold_generation(self, coco_dataset):
        result = generate_kfold_splits(coco_dataset, k=3, strategy="stratified")
        assert result.k == 3
        assert len(result.folds) == 3
        assert result.strategy == "stratified"

    def test_kfold_structure(self, coco_dataset):
        result = generate_kfold_splits(coco_dataset, k=2)
        for fold in result.folds:
            assert "train" in fold.splits
            assert "test" in fold.splits
            assert "val" in fold.splits


class TestLeakage:
    def test_leakage_detection(self, folder_dataset):
        report = detect_leakage(folder_dataset)
        assert report.dataset_id == folder_dataset.dataset_id
        assert report.total_leaks >= 0
        assert hasattr(report, "leakage_pairs")

    def test_leakage_with_split(self, folder_dataset):
        split_result = split_dataset(folder_dataset)
        report = detect_leakage(folder_dataset, split_result=split_result.splits)
        assert report.total_leaks >= 0
