from __future__ import annotations

from pixlint.analysis.integrity import check_integrity


class TestIntegrityChecks:
    def test_integrity_corrupt(self, folder_dataset):
        report = check_integrity(folder_dataset, checks=["corrupt"])
        assert report.dataset_id == folder_dataset.dataset_id
        assert "corrupt_images" in report.summary

    def test_integrity_missing_labels(self, folder_dataset):
        report = check_integrity(folder_dataset, checks=["missing_labels"])
        assert report.dataset_id == folder_dataset.dataset_id
        assert "missing_labels" in report.summary
        assert report.summary["missing_labels"] == len(folder_dataset)

    def test_integrity_with_labels(self, coco_dataset):
        report = check_integrity(coco_dataset, checks=["missing_labels"])
        assert report.summary.get("missing_labels", 0) == 0

    def test_integrity_report_structure(self, folder_dataset):
        report = check_integrity(folder_dataset)
        assert hasattr(report, "total_issues")
        assert hasattr(report, "issues")
        assert hasattr(report, "summary")
