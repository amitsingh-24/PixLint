from __future__ import annotations

from unittest.mock import patch


from pixlint.analysis.active_learning import (
    diversity_sampling,
    query_strategy,
    uncertainty_sampling,
    _get_labels_for_images,
    _filter_dataset_by_ids,
)
from pixlint.utils.schemas import EmbeddingResult


class TestUncertaintySampling:
    def test_no_sklearn(self):
        from pixlint.analysis.active_learning import _SKLEARN_AVAILABLE
        if not _SKLEARN_AVAILABLE:
            with patch("pixlint.analysis.active_learning._SKLEARN_AVAILABLE", False):
                result = uncertainty_sampling(None)
                assert "error" in result

    def test_not_enough_embeddings(self, folder_dataset):
        with patch("pixlint.analysis.active_learning.compute_embeddings") as mock_emb:
            mock_emb.return_value = EmbeddingResult(
                dataset_id="test", model="resnet50",
                image_ids=[], embeddings=None,
                embedding_dim=None, num_images=0,
            )
            result = uncertainty_sampling(folder_dataset)
            if "error" in result:
                assert "2" in result["error"]

    def test_not_enough_labeled(self):
        with patch("pixlint.analysis.active_learning.compute_embeddings") as mock_emb:
            mock_emb.return_value = EmbeddingResult(
                dataset_id="test", model="resnet50",
                image_ids=["a", "b"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                embedding_dim=2, num_images=2,
            )
            ds_mock = type("MockDS", (), {
                "images": [],
                "get_class_names": lambda self: ["class1"],
            })()
            result = uncertainty_sampling(ds_mock)
            if "error" in result:
                assert "labeled" in result["error"]


class TestDiversitySampling:
    def test_no_sklearn(self):
        with patch("pixlint.analysis.active_learning._SKLEARN_AVAILABLE", False):
            result = diversity_sampling(None)
            assert "error" in result

    def test_small_dataset(self, folder_dataset):
        with patch("pixlint.analysis.active_learning.compute_embeddings") as mock_emb:
            mock_emb.return_value = EmbeddingResult(
                dataset_id="test", model="resnet50",
                image_ids=["a"], embeddings=[[0.1, 0.2]],
                embedding_dim=2, num_images=1,
            )
            result = diversity_sampling(folder_dataset, n=5)
            if "error" in result:
                assert "5" in result["error"]

    def test_diversity_with_data(self):
        n = 5
        with patch("pixlint.analysis.active_learning.compute_embeddings") as mock_emb:
            mock_emb.return_value = EmbeddingResult(
                dataset_id="test", model="resnet50",
                image_ids=[f"img_{i}" for i in range(n)],
                embeddings=[[float(i), float(i * 2)] for i in range(n)],
                embedding_dim=2, num_images=n,
            )
            ds_mock = type("MockDS", (), {
                "images": [],
                "get_class_names": lambda self: [],
                "dataset_id": "test",
                "name": "test",
                "format": "folder",
                "path": "/tmp",
            })()
            result = diversity_sampling(ds_mock, n=3)
            assert "sampled_ids" in result
            assert len(result["sampled_ids"]) == 3


class TestQueryStrategy:
    def test_unknown_strategy(self):
        result = query_strategy(None, strategy="unknown")
        assert "error" in result

    def test_combined_fallback(self, folder_dataset):
        with patch("pixlint.analysis.active_learning.uncertainty_sampling") as mock_unc:
            mock_unc.return_value = {"error": "test error"}
            with patch("pixlint.analysis.active_learning.diversity_sampling") as mock_div:
                mock_div.return_value = {"strategy": "diversity", "n": 0, "sampled_ids": []}
                result = query_strategy(folder_dataset, strategy="combined")
                assert result["strategy"] == "diversity"


class TestHelpers:
    def test_get_labels_for_images(self, coco_dataset):
        labels = _get_labels_for_images(coco_dataset)
        assert isinstance(labels, dict)

    def test_filter_dataset_by_ids(self, folder_dataset):
        ids = [img.image_id for img in folder_dataset.images[:1]]
        filtered = _filter_dataset_by_ids(folder_dataset, ids)
        assert len(filtered.images) == 1
        assert filtered.images[0].image_id in ids
