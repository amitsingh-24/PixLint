from __future__ import annotations

from pixlint.analysis.outliers import detect_outliers


class TestOutlierDetection:
    def test_outlier_detection(self, folder_dataset):
        report = detect_outliers(folder_dataset, methods=["resolution"])
        assert report.dataset_id == folder_dataset.dataset_id
        assert report.total_outliers >= 0
        assert "resolution" in report.methods_used

    def test_outlier_all_methods(self, coco_dataset):
        report = detect_outliers(coco_dataset, methods=["resolution", "annotation"])
        assert len(report.methods_used) == 2
        assert hasattr(report, "outlier_ids")
        assert hasattr(report, "outlier_scores")

    def test_outlier_report_structure(self, folder_dataset):
        report = detect_outliers(folder_dataset)
        assert hasattr(report, "methods_used")
        assert hasattr(report, "outlier_ids")
        assert hasattr(report, "outlier_scores")
        assert hasattr(report, "total_outliers")
        assert hasattr(report, "summary")
