from __future__ import annotations

from pixlint.analysis.label_errors import _primary_label, find_label_errors
from pixlint.analysis.query import query_dataset
from pixlint.analysis.readiness import dataset_readiness_report
from pixlint.analysis.slices import discover_slices


class TestQueryEngine:
    def test_filter_by_class(self, coco_dataset):
        res = query_dataset(coco_dataset, {"classes": ["cat"]})
        assert res.num_matched >= 1
        assert res.num_matched <= res.total_images
        assert "classes" in res.applied_filters

    def test_exclude_class(self, coco_dataset):
        res = query_dataset(coco_dataset, {"exclude_classes": ["cat"]})
        # every matched image must not contain a cat
        matched = set(res.matched_image_ids)
        for img in coco_dataset.images:
            if img.image_id in matched:
                assert all(a.label != "cat" for a in img.annotations)

    def test_min_annotations(self, coco_dataset):
        res = query_dataset(coco_dataset, {"min_annotations": 1})
        assert res.num_matched == len(coco_dataset.images)  # all have 1 ann

    def test_min_area(self, coco_dataset):
        res = query_dataset(coco_dataset, {"min_area": 100000})
        assert res.num_matched == 0  # no boxes that large in fixture

    def test_position(self, coco_dataset):
        res = query_dataset(coco_dataset, {"position": "left"})
        assert isinstance(res.matched_image_ids, list)

    def test_limit(self, coco_dataset):
        res = query_dataset(coco_dataset, {"min_annotations": 0, "limit": 2})
        assert res.num_matched <= 2


class TestReadiness:
    def test_report_structure(self, coco_dataset):
        r = dataset_readiness_report(coco_dataset)
        assert 0 <= r.readiness_score <= 100
        assert r.readiness_level in ("not_ready", "needs_work", "training_ready")
        assert len(r.recommendations) >= 1
        # every rec has an actionable next step
        assert all(rec.action for rec in r.recommendations)
        # a runnable remediation pipeline is attached
        assert r.remediation_pipeline is not None
        assert "steps" in r.remediation_pipeline

    def test_empty_dataset(self, folder_dataset):
        # folder_dataset has images but no annotations -> should still produce a report
        r = dataset_readiness_report(folder_dataset)
        assert r.readiness_level in ("not_ready", "needs_work", "training_ready")


class TestSlices:
    def test_discovers_slices(self, coco_dataset):
        r = discover_slices(coco_dataset)
        assert r.num_images == len(coco_dataset.images)
        # class + size + aspect dimensions should all appear
        dims = {s.dimension for s in r.all_slices}
        assert {"class", "size", "aspect"}.issubset(dims)
        assert len(r.recommendations) >= 1


class TestLabelErrors:
    def test_primary_label(self, coco_dataset):
        for img in coco_dataset.images:
            lab = _primary_label(img)
            assert lab in (None, "cat", "dog")

    def test_graceful_small_dataset(self, coco_dataset):
        # only 4 images < k+1 -> returns a report with a note, no crash
        report = find_label_errors(coco_dataset, k=10)
        assert report.num_suspected_errors == 0
        assert report.notes
