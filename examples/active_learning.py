#!/usr/bin/env python3
"""
PixLint — Active Learning Loop

Iteratively select most informative samples for labeling.
"""

import os

from pixlint.analysis.active_learning import (
    diversity_sampling,
    query_strategy,
    uncertainty_sampling,
)
from pixlint.analysis.embeddings import compute_embeddings
from pixlint.core.loader import load_dataset

DATA_DIR = os.environ.get("CV_DATA_DIR", "/tmp/demo_data")

def main():
    # Load unlabeled pool
    pool = load_dataset(os.path.join(DATA_DIR, "unlabeled_pool"))
    print(f"Pool: {len(pool)} unlabeled images")

    # Compute embeddings once to warm the cache used by the sampling strategies.
    emb = compute_embeddings(pool, model="resnet50")
    print(f"Computed {len(emb.embeddings)} embeddings")

    for round_num in range(1, 6):
        print(f"\n=== Round {round_num} ===")

        # Strategy 1: Uncertainty (entropy)
        unc = uncertainty_sampling(pool, method="entropy", n=10, model="resnet50")
        print(f"  Uncertainty picks: {[r.image_id for r in unc.selected]}")

        # Strategy 2: Diversity (k-means++ on embeddings)
        div = diversity_sampling(pool, n=10, model="resnet50")
        print(f"  Diversity picks: {[r.image_id for r in div.selected]}")

        # Strategy 3: Combined (alpha=0.5)
        comb = query_strategy(pool, strategy="combined", n=10, model="resnet50", alpha=0.5)
        print(f"  Combined picks: {[r.image_id for r in comb.selected]}")

        # Human labels these → move to labeled set → retrain model
        # ... labeling workflow ...

        # Remove from pool (simplified)
        labeled_ids = {r.image_id for r in comb.selected}
        pool = pool.filter(lambda r: r.image_id not in labeled_ids)
        print(f"  Remaining pool: {len(pool)}")

if __name__ == "__main__":
    import os
    main()
