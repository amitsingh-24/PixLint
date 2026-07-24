from __future__ import annotations

import numpy as np
from PIL import Image

from pixlint.core.curation import (
    clean_dataset,
    filter_dataset,
    materialize_coco,
    remap_classes,
)
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DatasetFormat


class TestFilterDataset:
    def test_filter_by_class(self, coco_dataset):
        new_ds, result = filter_dataset(coco_dataset, classes=["cat"])
        assert result.operation == "filter"
        # only images that still have a "cat" annotation survive
        labels = {ann.label for img in new_ds.images for ann in img.annotations}
        assert labels == {"cat"}
        assert result.num_images_out <= result.num_images_in
        assert result.new_dataset_id != coco_dataset.dataset_id

    def test_filter_by_image_ids(self, coco_dataset):
        keep = [coco_dataset.images[0].image_id]
        new_ds, result = filter_dataset(coco_dataset, image_ids=keep)
        assert result.num_images_out == 1
        assert new_ds.images[0].image_id == keep[0]

    def test_filter_by_area(self, coco_dataset):
        # areas in fixture: 2500, 3600, 1200, 2475 -> min_area 2000 keeps 3
        new_ds, result = filter_dataset(coco_dataset, min_area=2000)
        assert result.num_annotations_out == 3

    def test_filter_has_annotations_false(self, folder_dataset):
        # folder dataset has no annotations -> keeping images WITHOUT anns keeps all
        new_ds, result = filter_dataset(folder_dataset, has_annotations=False)
        assert result.num_images_out == result.num_images_in


class TestCleanDataset:
    def test_clip_out_of_bounds(self, tmp_path):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(tmp_path / "a.jpg")
        ds = CVDataset(str(tmp_path), format=DatasetFormat.FOLDER)
        # inject an out-of-bounds + a degenerate box
        from pixlint.utils.schemas import Annotation
        ds.images[0].width, ds.images[0].height = 100, 100
        ds.images[0].annotations = [
            Annotation(label="x", bbox=(50, 50, 200, 200)),   # OOB -> clipped to (50,50,100,100)
            Annotation(label="y", bbox=(10, 10, 10, 20)),     # zero-width -> dropped
        ]
        new_ds, result = clean_dataset(ds, drop_corrupt=False)
        anns = new_ds.images[0].annotations
        assert len(anns) == 1  # degenerate dropped
        assert anns[0].bbox == (50, 50, 100, 100)  # clipped

    def test_drop_duplicates(self, tmp_path):
        base = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(base).save(tmp_path / "a.jpg")
        Image.fromarray(base).save(tmp_path / "b.jpg")  # exact duplicate
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "c.jpg")
        ds = CVDataset(str(tmp_path), format=DatasetFormat.FOLDER)
        new_ds, result = clean_dataset(ds, drop_corrupt=False, drop_duplicates=True)
        assert result.num_images_out == 2  # one dup removed


class TestRemapClasses:
    def test_merge_classes(self, coco_dataset):
        new_ds, result = remap_classes(coco_dataset, {"cat": "animal", "dog": "animal"})
        labels = {ann.label for img in new_ds.images for ann in img.annotations}
        assert labels == {"animal"}

    def test_drop_class_via_empty_mapping(self, coco_dataset):
        new_ds, result = remap_classes(coco_dataset, {"cat": ""})
        labels = {ann.label for img in new_ds.images for ann in img.annotations}
        assert "cat" not in labels
        assert "dog" in labels


class TestMaterialize:
    def test_writes_coco_on_disk(self, coco_dataset, tmp_path):
        out = tmp_path / "out"
        n_imgs, n_anns = materialize_coco(coco_dataset.images, str(out))
        assert (out / "annotations" / "instances.json").exists()
        assert len(list((out / "images").glob("*"))) == n_imgs
        # the written dataset re-loads cleanly
        reloaded = CVDataset(str(out), format=DatasetFormat.COCO)
        assert len(reloaded.images) == n_imgs
