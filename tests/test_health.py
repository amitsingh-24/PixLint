from __future__ import annotations

from pixlint.analysis.health import dataset_health_score


class TestHealthScore:
    def test_health_score_structure(self, folder_dataset):
        score = dataset_health_score(folder_dataset)
        assert score.dataset_id == folder_dataset.dataset_id
        assert 0 <= score.overall <= 100
        assert hasattr(score, "breakdown")
        assert hasattr(score, "recommendations")

    def test_health_score_breakdown(self, folder_dataset):
        score = dataset_health_score(folder_dataset)
        breakdown = score.breakdown
        assert 0 <= breakdown.class_balance <= 100
        assert 0 <= breakdown.image_quality <= 100
        assert 0 <= breakdown.integrity <= 100
        assert 0 <= breakdown.annotation_consistency <= 100
        assert 0 <= breakdown.diversity <= 100

    def test_health_score_with_annotations(self, coco_dataset):
        score = dataset_health_score(coco_dataset)
        assert score.overall > 0
        assert len(score.recommendations) >= 0
