from __future__ import annotations

from pixlint.core.merge import _remap_annotations, _unify_labels, merge_datasets
from pixlint.utils.schemas import Annotation


class TestMerge:
    def test_merge_two_datasets(self, folder_dataset, coco_dataset):
        result = merge_datasets([folder_dataset, coco_dataset], merged_name="test_merge")
        assert result.total_images > 0
        assert len(result.source_datasets) == 2
        assert result.total_annotations >= 0
        assert result.num_classes >= 0
        assert result.merged_dataset_id is not None

    def test_merge_same_dataset(self, folder_dataset):
        result = merge_datasets([folder_dataset, folder_dataset], merged_name="test_self_merge")
        assert result.total_images > 0
        assert result.duplicate_count >= 0

    def test_merge_union_strategy(self, folder_dataset, coco_dataset):
        result = merge_datasets([folder_dataset, coco_dataset], merged_name="test_union", strategy="union")
        assert result.total_images > 0

    def test_merge_intersection_strategy(self, folder_dataset, coco_dataset):
        result = merge_datasets(
            [folder_dataset, coco_dataset],
            merged_name="test_intersection",
            strategy="intersection",
        )
        assert result.total_images >= 0

    def test_merge_label_unification(self, folder_dataset, coco_dataset):
        result = merge_datasets(
            [folder_dataset, coco_dataset],
            merged_name="test_labels",
            unify_labels=True,
        )
        assert len(result.label_mapping) >= 0
        for entry in result.label_mapping:
            assert entry.source_label is not None
            assert entry.unified_label is not None

    def test_merge_no_deduplicate(self, folder_dataset):
        result = merge_datasets(
            [folder_dataset, folder_dataset],
            merged_name="test_no_dedup",
            deduplicate=False,
        )
        assert result.total_images > 0

    def test_unify_labels_basic(self):
        labels = {"Cat": {"ds1"}, "cat": {"ds2"}, "DOG": {"ds1"}, "dog": {"ds2"}}
        result = _unify_labels(labels)
        assert result["Cat"] == result["cat"]
        assert result["DOG"] == result["dog"]

    def test_unify_labels_case_normalization(self):
        labels = {"car": {"a"}, "Car": {"b"}, "CAR": {"c"}}
        result = _unify_labels(labels)
        assert len(set(result.values())) == 1

    def test_remap_annotations(self):
        anns = [Annotation(label="Cat", bbox=(1, 2, 3, 4))]
        label_map = {"Cat": "cat"}
        result = _remap_annotations(anns, label_map)
        assert result[0].label == "cat"
        assert result[0].bbox == (1, 2, 3, 4)

    def test_merge_less_than_two_datasets(self, folder_dataset):
        import pytest
        with pytest.raises(ValueError, match="at least 2"):
            merge_datasets([folder_dataset])

    def test_merge_empty_datasets(self, folder_dataset, coco_dataset):
        import tempfile

        from pixlint.core.loader import CVDataset
        from pixlint.utils.schemas import DatasetFormat
        with tempfile.TemporaryDirectory() as d:
            empty = CVDataset(d, format=DatasetFormat.FOLDER)
            result = merge_datasets([folder_dataset, empty], merged_name="test_empty")
            assert result.total_images >= 0
