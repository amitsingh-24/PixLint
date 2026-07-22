#!/usr/bin/env python3
"""
PixLint — Semantic Search & Embeddings

Uses ResNet50 or CLIP embeddings for image similarity search.
"""

import os
from pixlint.core.loader import load_dataset
from pixlint.analysis.embeddings import compute_embeddings, semantic_search, cluster_dataset

DATA_DIR = os.environ.get("CV_DATA_DIR", "/tmp/demo_data")

def main():
    ds = load_dataset(os.path.join(DATA_DIR, "your_dataset"))
    print(f"Loaded: {ds.name} ({len(ds)} images)")

    # Compute embeddings (ResNet50 by default)
    print("\nComputing ResNet50 embeddings...")
    emb = compute_embeddings(ds, model="resnet50")
    print(f"  Embeddings: {emb.embeddings.shape} ({emb.embeddings.shape[0]} images, {emb.embeddings.shape[1]} dims)")

    # Semantic search by text query
    print("\nSemantic search: 'person riding bicycle'")
    results = semantic_search(ds, query="person riding bicycle", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.image_id} — score: {r.similarity:.4f}")

    # Cluster dataset
    print("\nClustering into 5 groups (KMeans)...")
    clusters = cluster_dataset(ds, n_clusters=5, model="resnet50")
    print(f"  Cluster sizes: {clusters.cluster_sizes}")

    # Also try CLIP
    print("\nComputing CLIP embeddings...")
    emb_clip = compute_embeddings(ds, model="clip-vit-base")
    print(f"  CLIP dims: {emb_clip.embeddings.shape[1]}")

if __name__ == "__main__":
    main()