from __future__ import annotations

from pixlint.analysis.distribution import analyze_distribution


class TestDistributionAnalysis:
    def test_distribution_empty(self, folder_dataset):
        report = analyze_distribution(folder_dataset, analyses=["class_counts"])
        assert report.dataset_id == folder_dataset.dataset_id
        assert report.total_images == 5
        assert report.total_annotations == 0

    def test_distribution_with_classes(self, coco_dataset):
        report = analyze_distribution(coco_dataset, analyses=["class_counts"])
        assert report.total_images == 4
        assert report.total_annotations == 4

    def test_distribution_class_names(self, coco_dataset):
        report = analyze_distribution(coco_dataset)
        class_names = [d.class_name for d in report.class_distribution]
        assert "cat" in class_names
        assert "dog" in class_names

    def test_distribution_metrics(self, coco_dataset):
        report = analyze_distribution(coco_dataset)
        for d in report.class_distribution:
            assert d.percentage > 0
            assert d.annotation_count > 0

    def test_distribution_report_structure(self, folder_dataset):
        report = analyze_distribution(folder_dataset)
        assert hasattr(report, "class_distribution")
        assert hasattr(report, "total_images")
        assert hasattr(report, "total_annotations")

    def test_spatial_heatmap(self, coco_dataset):
        report = analyze_distribution(coco_dataset, analyses=["spatial"])
        assert report.spatial_heatmap is not None
        assert report.spatial_heatmap.grid_size == (10, 10)

    def test_co_occurrence(self, coco_dataset):
        report = analyze_distribution(coco_dataset, analyses=["co_occurrence"])
        assert report.label_co_occurrence is not None

    def test_complexity(self, coco_dataset):
        report = analyze_distribution(coco_dataset, analyses=["complexity"])
        assert report.complexity_scores is not None
        assert len(report.complexity_scores) == report.total_images
