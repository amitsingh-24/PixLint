from __future__ import annotations

import importlib.util
import os

import pytest

from pixlint.export.pytorch import export_pytorch
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.ultralytics import export_ultralytics
from pixlint.export.hdf5 import export_hdf5

_HAS_H5PY = importlib.util.find_spec("h5py") is not None


class TestExportPyTorch:
    def test_pytorch_export(self, folder_dataset, tmp_path):
        result = export_pytorch(folder_dataset, output_dir=str(tmp_path / "pt"), image_size=(224, 224))
        assert result.export_format == "pytorch"
        assert result.num_images_exported > 0
        assert os.path.isdir(str(tmp_path / "pt"))
        assert os.path.isfile(str(tmp_path / "pt/metadata.json"))
        assert os.path.isfile(str(tmp_path / "pt/dataset.py"))

    def test_pytorch_export_with_annotations(self, coco_dataset, tmp_path):
        result = export_pytorch(coco_dataset, output_dir=str(tmp_path / "pt2"))
        assert result.num_annotations_exported > 0


class TestExportTensorFlow:
    def test_tensorflow_export_metadata(self, coco_dataset, tmp_path):
        result = export_tensorflow(coco_dataset, output_dir=str(tmp_path / "tf"), tfrecord_shards=0)
        assert result.export_format == "tensorflow"
        assert result.num_images_exported > 0
        assert os.path.isfile(str(tmp_path / "tf/metadata.json"))

    def test_tensorflow_dataset_script(self, coco_dataset, tmp_path):
        export_tensorflow(coco_dataset, output_dir=str(tmp_path / "tf2"), tfrecord_shards=0)
        assert os.path.isfile(str(tmp_path / "tf2/dataset.py"))


class TestExportUltralytics:
    def test_ultralytics_export(self, coco_dataset, tmp_path):
        result = export_ultralytics(coco_dataset, output_dir=str(tmp_path / "yolo"), image_size=320)
        assert result.export_format == "ultralytics"
        assert result.num_images_exported > 0
        assert result.num_annotations_exported > 0
        assert os.path.isfile(str(tmp_path / "yolo/dataset.yaml"))

    def test_ultralytics_with_split(self, coco_dataset, tmp_path):
        split = {"train": ["1"], "val": ["2"], "test": ["3", "4"]}
        result = export_ultralytics(coco_dataset, output_dir=str(tmp_path / "yolo_split"), split=split)
        assert result.num_images_exported > 0


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed (install the [hdf5] extra)")
class TestExportHDF5:
    def test_hdf5_export_structure(self, folder_dataset, tmp_path):
        out = str(tmp_path / "dataset.h5")
        result = export_hdf5(folder_dataset, output_path=out, image_size=(64, 64))
        if result.num_images_exported > 0:
            assert os.path.isfile(out)
            assert result.export_format == "hdf5"

    def test_hdf5_export_with_annotations(self, coco_dataset, tmp_path):
        out = str(tmp_path / "coco.h5")
        result = export_hdf5(coco_dataset, output_path=out, image_size=(64, 64))
        if result.num_images_exported > 0:
            assert os.path.isfile(out)
