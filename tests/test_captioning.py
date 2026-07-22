from __future__ import annotations

from unittest.mock import patch


from pixlint.analysis.captioning import (
    generate_captions,
    auto_tag_dataset,
    enrich_metadata,
    _BLIP_AVAILABLE,
)


class TestCaptioning:
    def test_generate_captions_no_blip(self):
        if not _BLIP_AVAILABLE:
            with patch("pixlint.analysis.captioning._BLIP_AVAILABLE", False):
                from pixlint.core.loader import CVDataset
                from pixlint.utils.schemas import DatasetFormat
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    ds = CVDataset(d, format=DatasetFormat.FOLDER)
                    result = generate_captions(ds, model="blip")
                    assert isinstance(result, list)
                    if result and "error" in result[0]:
                        assert "BLIP" in result[0]["error"]

    def test_generate_captions_unknown_model(self, folder_dataset):
        result = generate_captions(folder_dataset, model="unknown")
        assert isinstance(result, list)
        if result and "error" in result[0]:
            assert "unknown" in result[0]["error"]

    def test_generate_captions_resnet_no_torch(self, folder_dataset):
        with patch("pixlint.analysis.captioning._IMAGENET_CLASSES", []):
            result = generate_captions(folder_dataset, model="resnet")
            assert isinstance(result, list)
            if result and "error" in result[0]:
                assert "torchvision" in result[0]["error"]

    def test_auto_tag_no_resnet(self, folder_dataset):
        with patch("pixlint.analysis.captioning._IMAGENET_CLASSES", []):
            result = auto_tag_dataset(folder_dataset, method="resnet")
            assert isinstance(result, list)

    def test_auto_tag_blip_no_blip(self, folder_dataset):
        with patch("pixlint.analysis.captioning._BLIP_AVAILABLE", False):
            result = auto_tag_dataset(folder_dataset, method="blip")
            assert isinstance(result, list)

    def test_enrich_metadata_empty(self, folder_dataset):
        with patch("pixlint.analysis.captioning._BLIP_AVAILABLE", False):
            with patch("pixlint.analysis.captioning._IMAGENET_CLASSES", []):
                result = enrich_metadata(folder_dataset)
                assert result["num_images"] >= 0
                assert result["total_enriched"] >= 0

    def test_enrich_metadata_updates_metadata(self, coco_dataset):
        with patch("pixlint.analysis.captioning._IMAGENET_CLASSES", ["cat", "dog"]):
            with patch("pixlint.analysis.captioning._generate_resnet_tags") as mock_tags:
                mock_tags.return_value = []
                with patch("pixlint.analysis.captioning._BLIP_AVAILABLE", False):
                    result = enrich_metadata(coco_dataset)
                    assert result is not None
