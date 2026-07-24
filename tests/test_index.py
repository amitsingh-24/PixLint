from __future__ import annotations

import tempfile
from pathlib import Path

from pixlint.core.index import ImageIndex


class TestImageIndex:
    def test_index_build(self, folder_dataset):
        index = ImageIndex()
        image_ids = [img.image_id for img in folder_dataset.images]
        image_paths = [img.path for img in folder_dataset.images]
        index.build(image_ids, image_paths)
        assert index.size == len(folder_dataset.images)

    def test_get_path(self, folder_dataset):
        index = ImageIndex()
        image_ids = [img.image_id for img in folder_dataset.images]
        image_paths = [img.path for img in folder_dataset.images]
        index.build(image_ids, image_paths)
        path = index.get_path(image_ids[0])
        assert path is not None

    def test_find_similar_phash(self, folder_dataset):
        index = ImageIndex()
        image_ids = [img.image_id for img in folder_dataset.images]
        image_paths = [img.path for img in folder_dataset.images]
        index.build(image_ids, image_paths)
        similar = index.find_similar_phash(image_ids[0])
        assert isinstance(similar, list)

    def test_save_and_load(self, folder_dataset):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        index = ImageIndex(db_path=db_path)
        image_ids = [img.image_id for img in folder_dataset.images]
        image_paths = [img.path for img in folder_dataset.images]
        index.build(image_ids, image_paths)

        save_path = db_path + ".save"
        index.save(save_path)

        index2 = ImageIndex(db_path=db_path + ".load")
        index2.load(save_path)
        assert index2.size == index.size

        Path(save_path).unlink(missing_ok=True)
        Path(db_path).unlink(missing_ok=True)
        Path(db_path + ".load").unlink(missing_ok=True)

    def test_search_by_path(self, folder_dataset):
        index = ImageIndex()
        image_ids = [img.image_id for img in folder_dataset.images]
        image_paths = [img.path for img in folder_dataset.images]
        index.build(image_ids, image_paths)
        results = index.search_by_path("image")
        assert len(results) > 0
