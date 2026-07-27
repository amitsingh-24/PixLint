from __future__ import annotations

from typing import Any

import numpy as np

from pixlint.analysis.embeddings import compute_embeddings
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import EmbeddingResult

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
try:
    from sklearn.manifold import TSNE
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

_UMAP_AVAILABLE = False
try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    pass


def reduce_dimensions(
    embeddings: EmbeddingResult,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    if not embeddings.embeddings or len(embeddings.embeddings) < 3:
        return {"error": "Not enough embeddings for dimensionality reduction"}

    data = np.array(embeddings.embeddings)
    n_samples = data.shape[0]
    perplexity = min(30, n_samples - 1)

    if method == "tsne":
        if not _SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available for t-SNE"}
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
        )
    elif method == "umap":
        if not _UMAP_AVAILABLE:
            return _fallback_tsne(data, n_components, random_state, n_samples)
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
        )
    else:
        if not _SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available for PCA"}
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)

    try:
        reduced = reducer.fit_transform(data)
    except Exception as e:
        return {"error": str(e)}

    result: dict[str, Any] = {
        "method": method,
        "n_components": n_components,
        "image_ids": embeddings.image_ids,
        "coordinates": reduced.tolist(),
    }

    return result


def _fallback_tsne(
    data: np.ndarray, n_components: int, random_state: int, n_samples: int
) -> dict[str, Any]:
    if not _SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available for dimensionality reduction"}

    perplexity = min(30, n_samples - 1)
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    )
    reduced = reducer.fit_transform(data)
    return {
        "method": "tsne",
        "n_components": n_components,
        "image_ids": [],
        "coordinates": reduced.tolist(),
    }


def plot_embeddings(
    dataset: CVDataset,
    method: str = "umap",
    color_by: str | None = "class",
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available for plotting"}

    emb_result = compute_embeddings(dataset)
    if not emb_result.embeddings:
        return {"error": "No embeddings computed"}

    reduced = reduce_dimensions(emb_result, method=method)
    if "error" in reduced:
        return reduced

    coords = np.array(reduced["coordinates"])
    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by == "class" and dataset.get_class_names():
        class_map: dict[str, list[int]] = {}
        image_map = {img.image_id: img for img in dataset.images}
        for i, img_id in enumerate(emb_result.image_ids):
            img = image_map.get(img_id)
            if img:
                for ann in img.annotations:
                    class_map.setdefault(ann.label, []).append(i)

        colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, max(len(class_map), 1)))
        for (cls, indices), color in zip(class_map.items(), colors):
            if indices:
                ax.scatter(coords[indices, 0], coords[indices, 1], label=cls, color=color, alpha=0.7, s=30)
        ax.legend(fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=20, c="steelblue")

    ax.set_title(f"{method.upper()} Embedding Visualization")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"method": method, "save_path": save_path, "n_points": len(coords)}

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {
        "method": method,
        "n_points": len(coords),
        "image_base64": img_b64,
    }
