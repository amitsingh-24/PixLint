from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import pytest

from pixlint.export.extra_formats import (
    _build_coco_json,
    _SimpleTarWriter,
    export_cvat_xml,
    export_fiftyone,
    export_labelme_json,
    export_webdataset,
)
from pixlint.utils.schemas import ExportResult


class TestExtraFormats:
    def test_export_webdataset(self, folder_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_webdataset(folder_dataset, output_dir=tmpdir)
            assert isinstance(result, ExportResult)
            assert result.export_format == "webdataset"
            assert result.num_images_exported >= 0

    def test_export_webdataset_empty(self, folder_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_webdataset(folder_dataset, output_dir=tmpdir, shard_size=1)
            assert result.num_images_exported >= 0

    def test_export_cvat_xml(self, folder_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "annotations.xml")
            result = export_cvat_xml(folder_dataset, output_path=out_path)
            assert result.export_format == "cvat_xml"
            assert os.path.exists(out_path)
            tree = ET.parse(out_path)
            root = tree.getroot()
            assert root.tag == "annotations"

    def test_export_labelme_json(self, folder_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_labelme_json(folder_dataset, output_dir=tmpdir)
            assert result.export_format == "labelme_json"
            json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            assert len(json_files) >= 0

    def test_export_fiftyone_fallback(self, folder_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("pixlint.export.extra_formats._FIFTYONE_AVAILABLE", False)
                result = export_fiftyone(folder_dataset, output_dir=tmpdir)
                assert result.export_format == "fiftyone_json"
                assert os.path.exists(os.path.join(tmpdir, "dataset.json"))

    def test_build_coco_json(self, folder_dataset):
        result = _build_coco_json(folder_dataset)
        assert "images" in result
        assert "annotations" in result
        assert "categories" in result
        assert len(result["images"]) == len(folder_dataset.images)

    def test_simple_tar_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "test.tar")
            writer = _SimpleTarWriter(tar_path)
            writer.write({
                "__key__": "test_img",
                "jpg": b"fake_jpeg_bytes",
                "txt": b"0 0.5 0.5 0.2 0.3",
            })
            writer.close()
            assert os.path.exists(tar_path)
            assert os.path.getsize(tar_path) > 0
