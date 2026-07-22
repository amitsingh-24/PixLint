from __future__ import annotations

from pixlint.analysis.duplicates import find_duplicates


class TestSSIMDuplicates:
    def test_ssim_detection(self, folder_dataset):
        report = find_duplicates(folder_dataset, methods=["ssim"], thresholds={"ssim": 0.9})
        assert report.dataset_id == folder_dataset.dataset_id
        assert isinstance(report.total_duplicate_groups, int)

    def test_all_methods(self, coco_dataset):
        report = find_duplicates(coco_dataset, methods=["exact", "perceptual", "ssim"])
        assert "exact" in report.method_summary
        assert "perceptual" in report.method_summary
        assert "ssim" in report.method_summary

    def test_empty_methods_default(self, folder_dataset):
        report = find_duplicates(folder_dataset)
        assert report.dataset_id == folder_dataset.dataset_id


class TestSemanticDuplicates:
    def test_semantic_method_exists(self, folder_dataset):
        report = find_duplicates(folder_dataset, methods=["semantic"])
        assert "semantic" in report.method_summary
