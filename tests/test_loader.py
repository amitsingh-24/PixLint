from __future__ import annotations

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DatasetFormat


class TestFolderLoader:
    def test_load_folder(self, temp_image_dir):
        dataset = CVDataset(str(temp_image_dir), format=DatasetFormat.FOLDER)
        assert len(dataset) == 5
        assert dataset.format == DatasetFormat.FOLDER

    def test_load_folder_auto_detect(self, temp_image_dir):
        dataset = CVDataset(str(temp_image_dir), format=DatasetFormat.AUTO)
        assert len(dataset) == 5

    def test_dataset_iteration(self, folder_dataset):
        count = 0
        for img in folder_dataset:
            assert img.image_id is not None
            assert img.path is not None
            count += 1
        assert count == len(folder_dataset)

    def test_dataset_getitem(self, folder_dataset):
        img = folder_dataset[0]
        assert img.image_id is not None
        assert img.path is not None

    def test_get_class_names(self, folder_dataset):
        names = folder_dataset.get_class_names()
        assert names == []

    def test_to_info(self, folder_dataset):
        info = folder_dataset.to_info()
        assert info.dataset_id is not None
        assert info.num_images == 5


class TestCocoLoader:
    def test_load_coco(self, temp_coco_dir):
        dataset = CVDataset(str(temp_coco_dir), format=DatasetFormat.COCO)
        assert len(dataset) == 4
        assert dataset.format == DatasetFormat.COCO

    def test_coco_annotations(self, coco_dataset):
        total_anns = 0
        for img in coco_dataset:
            total_anns += len(img.annotations)
        assert total_anns == 4

    def test_coco_class_names(self, coco_dataset):
        names = coco_dataset.get_class_names()
        assert "cat" in names
        assert "dog" in names


class TestVocLoader:
    def test_load_voc(self, temp_voc_dir):
        dataset = CVDataset(str(temp_voc_dir), format=DatasetFormat.VOC)
        assert len(dataset) == 3
        assert dataset.format == DatasetFormat.VOC

    def test_voc_annotations(self, voc_dataset):
        for img in voc_dataset:
            assert len(img.annotations) == 1

    def test_voc_class_names(self, voc_dataset):
        names = voc_dataset.get_class_names()
        assert "cat" in names
        assert "dog" in names


class TestYoloLoader:
    def test_load_yolo(self, temp_yolo_dir):
        dataset = CVDataset(str(temp_yolo_dir), format=DatasetFormat.YOLO)
        assert len(dataset) == 4
        assert dataset.format == DatasetFormat.YOLO

    def test_yolo_annotations(self, yolo_dataset):
        for img in yolo_dataset:
            assert len(img.annotations) == 1

    def test_yolo_class_names(self, yolo_dataset):
        names = yolo_dataset.get_class_names()
        assert len(names) > 0


class TestDetectFormat:
    def test_detect_folder(self, temp_image_dir):
        from pixlint.core.loader import detect_format
        fmt = detect_format(str(temp_image_dir))
        assert fmt == "folder"

    def test_detect_coco(self, temp_coco_dir):
        from pixlint.core.loader import detect_format
        fmt = detect_format(str(temp_coco_dir))
        assert fmt == "coco"

    def test_detect_voc(self, temp_voc_dir):
        from pixlint.core.loader import detect_format
        fmt = detect_format(str(temp_voc_dir))
        assert fmt == "voc"

    def test_detect_yolo(self, temp_yolo_dir):
        from pixlint.core.loader import detect_format
        fmt = detect_format(str(temp_yolo_dir))
        assert fmt == "yolo"
