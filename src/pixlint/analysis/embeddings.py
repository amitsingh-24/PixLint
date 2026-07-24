from __future__ import annotations

import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image, read_image_pil
from pixlint.utils.schemas import (
    ClusterResult,
    EmbeddingResult,
    SearchResult,
)

_TORCH_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import ResNet50_Weights, resnet50
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_TRANSFORMERS_AVAILABLE = False
try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_CLIP_AVAILABLE = False
try:
    import clip  # noqa: F401
    _CLIP_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


def _get_device() -> str:
    if _TORCH_AVAILABLE:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def compute_embeddings(
    dataset: CVDataset,
    model: str = "resnet50",
    layers: list[str] | None = None,
) -> EmbeddingResult:
    image_ids: list[str] = []
    embeddings_list: list[list[float]] = []
    embedding_dim: int | None = None

    if not _TORCH_AVAILABLE:
        return EmbeddingResult(
            dataset_id=dataset.dataset_id,
            model=model,
            image_ids=[],
            embeddings=None,
            embedding_dim=None,
            num_images=0,
        )

    device = _get_device()
    try:
        if model == "clip" and _CLIP_AVAILABLE:
            import clip
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            for img in dataset.images:
                image = read_image(img.path)
                if image is None:
                    continue
                pil_img = T.ToPILImage()(image)
                tensor = preprocess(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clip_model.encode_image(tensor).cpu().numpy().flatten()
                image_ids.append(img.image_id)
                embeddings_list.append(emb.tolist())
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
        else:
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights).to(device).eval()
            transform = weights.transforms()
            preprocess = T.Compose([
                T.ToPILImage(),
                transform,
            ])
            for img in dataset.images:
                image = read_image(img.path)
                if image is None:
                    continue
                tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = resnet(tensor).cpu().numpy().flatten()
                image_ids.append(img.image_id)
                embeddings_list.append(emb.tolist())
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Embedding computation failed: {e}")

    return EmbeddingResult(
        dataset_id=dataset.dataset_id,
        model=model,
        image_ids=image_ids,
        embeddings=embeddings_list if embeddings_list else None,
        embedding_dim=embedding_dim,
        num_images=len(image_ids),
    )


def semantic_search(
    dataset: CVDataset,
    query: str,
    top_k: int = 10,
    model: str = "resnet50",
) -> list[SearchResult]:
    if not _TORCH_AVAILABLE:
        return []

    device = _get_device()
    results: list[SearchResult] = []

    # Real text-to-image semantic search requires CLIP so that the text query
    # and the images live in the *same* embedding space. ResNet has no text
    # encoder, so without CLIP we cannot honour a text query meaningfully.
    if not _CLIP_AVAILABLE:
        import logging
        logging.getLogger(__name__).warning(
            "semantic_search requires CLIP for text queries; install with "
            "'pip install pixlint[clip]'. Returning no results."
        )
        return []

    try:
        import clip
        import torch

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        # Encode the text query into a unit-norm CLIP embedding.
        text_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_emb = clip_model.encode_text(text_tokens).cpu().numpy().flatten()
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Encode every dataset image with the SAME CLIP image encoder so the
        # dot product is a genuine cosine similarity in [-1, 1].
        dataset_embs: dict[str, np.ndarray] = {}
        for img in dataset.images:
            pil = read_image_pil(img.path)
            if pil is None:
                continue
            tensor = clip_preprocess(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(tensor).cpu().numpy().flatten()
            n = np.linalg.norm(emb)
            dataset_embs[img.image_id] = emb / n if n > 0 else emb

        if not dataset_embs:
            return []

        img_ids = list(dataset_embs.keys())
        all_embs = np.stack([dataset_embs[i] for i in img_ids])
        scores = all_embs @ query_emb  # cosine similarity, range [-1, 1]

        k = min(top_k, len(img_ids))
        top_indices = np.argsort(scores)[-k:][::-1]
        path_map = {img.image_id: img.path for img in dataset.images}

        for idx in top_indices:
            img_id = img_ids[idx]
            path = path_map.get(img_id, "")
            score = float(scores[idx])
            score = max(0.0, min(1.0, (score + 1) / 2))  # map cosine [-1,1] -> [0,1]
            results.append(SearchResult(image_id=img_id, path=path, score=round(score, 4)))

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Semantic search failed: {e}")

    return results


def _get_text_embedding(text: str, device: str) -> np.ndarray | None:
    if not _CLIP_AVAILABLE:
        return None
    try:
        import clip
        import torch
        model, _ = clip.load("ViT-B/32", device=device)
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(text_tokens).cpu().numpy().flatten()
        return text_emb / np.linalg.norm(text_emb)
    except Exception:
        return None


def cluster_dataset(
    dataset: CVDataset | EmbeddingResult,
    method: str = "kmeans",
    n_clusters: int | None = None,
) -> ClusterResult:
    if not _SKLEARN_AVAILABLE:
        dataset_id = dataset.dataset_id if hasattr(dataset, "dataset_id") else "unknown"
        return ClusterResult(
            dataset_id=dataset_id,
            method=method,
            n_clusters=0,
            labels=[],
            image_ids=[],
            cluster_sizes={},
            silhouette_score=None,
        )

    if isinstance(dataset, EmbeddingResult):
        emb_result = dataset
        dataset_id = emb_result.dataset_id
        image_ids = emb_result.image_ids
        if not emb_result.embeddings:
            return ClusterResult(
                dataset_id=dataset_id, method=method, n_clusters=0,
                labels=[], image_ids=[], cluster_sizes={}, silhouette_score=None,
            )
        embeddings = np.array(emb_result.embeddings)
    else:
        dataset_id = dataset.dataset_id
        emb_result = compute_embeddings(dataset)
        if not emb_result.embeddings:
            return ClusterResult(
                dataset_id=dataset_id, method=method, n_clusters=0,
                labels=[], image_ids=[], cluster_sizes={}, silhouette_score=None,
            )
        embeddings = np.array(emb_result.embeddings)
        image_ids = emb_result.image_ids

    n_samples = embeddings.shape[0]
    if n_clusters is None:
        n_clusters = min(10, max(2, n_samples // 10))

    labels: np.ndarray
    if method == "dbscan":
        eps = 0.5
        clusterer = DBSCAN(eps=eps, min_samples=2)
        labels = clusterer.fit_predict(embeddings)
        n_unique = len(set(labels) - {-1})
    else:
        n_clusters = min(n_clusters, n_samples)
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = clusterer.fit_predict(embeddings)
        n_unique = n_clusters

    sil_score = None
    if n_unique > 1 and n_samples > n_unique:
        try:
            sil_score = round(silhouette_score(embeddings, labels), 4)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Silhouette score computation failed: {e}")

    cluster_sizes: dict[str, int] = {}
    for label_id in sorted(set(labels)):
        key = "noise" if label_id == -1 else f"cluster_{label_id}"
        cluster_sizes[key] = int((labels == label_id).sum())

    return ClusterResult(
        dataset_id=dataset_id,
        method=method,
        n_clusters=n_unique,
        labels=labels.tolist(),
        image_ids=image_ids,
        cluster_sizes=cluster_sizes,
        silhouette_score=sil_score,
    )
