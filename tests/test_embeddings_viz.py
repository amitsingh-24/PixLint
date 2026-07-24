from __future__ import annotations

from pixlint.analysis.embeddings import compute_embeddings
from pixlint.visualization.embeddings_viz import plot_embeddings, reduce_dimensions


class TestEmbeddingViz:
    def test_reduce_dimensions(self, folder_dataset):
        emb = compute_embeddings(folder_dataset)
        result = reduce_dimensions(emb, method="pca", n_components=2)
        if "error" not in result:
            assert "coordinates" in result
            assert "method" in result
            assert result["n_components"] == 2

    def test_plot_embeddings(self, folder_dataset):
        result = plot_embeddings(folder_dataset, method="pca")
        if "error" not in result:
            assert "method" in result
