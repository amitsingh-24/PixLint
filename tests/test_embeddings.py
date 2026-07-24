from __future__ import annotations

from pixlint.analysis.embeddings import cluster_dataset, compute_embeddings, semantic_search


class TestEmbeddings:
    def test_compute_embeddings(self, folder_dataset):
        result = compute_embeddings(folder_dataset, model="resnet50")
        assert result.dataset_id == folder_dataset.dataset_id
        assert result.num_images >= 0
        assert result.model == "resnet50"

    def test_compute_embeddings_structure(self, folder_dataset):
        result = compute_embeddings(folder_dataset)
        assert hasattr(result, "image_ids")
        assert hasattr(result, "embedding_dim")
        assert hasattr(result, "num_images")

    def test_cluster_dataset(self, folder_dataset):
        result = cluster_dataset(folder_dataset, method="kmeans", n_clusters=2)
        assert result.dataset_id == folder_dataset.dataset_id
        assert result.method == "kmeans"

    def test_cluster_dataset_structure(self, folder_dataset):
        result = cluster_dataset(folder_dataset)
        assert hasattr(result, "n_clusters")
        assert hasattr(result, "labels")
        assert hasattr(result, "cluster_sizes")

    def test_semantic_search(self, folder_dataset):
        results = semantic_search(folder_dataset, query="test image", top_k=5)
        assert isinstance(results, list)
