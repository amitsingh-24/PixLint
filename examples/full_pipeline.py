#!/usr/bin/env python3
"""
PixLint — Full Pipeline Example

Demonstrates: load → analyze → augment → split → export → pipeline
"""

import os

from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.quality import analyze_quality
from pixlint.augmentation.pipeline import augment_dataset
from pixlint.core.loader import load_dataset
from pixlint.core.pipeline import execute_pipeline, get_template
from pixlint.export.pytorch import export_pytorch
from pixlint.splitting.splitter import split_dataset
from pixlint.transformation.normalize import compute_channel_stats, normalize_dataset
from pixlint.transformation.resize import resize_dataset

DATA_DIR = os.environ.get("CV_DATA_DIR", "/tmp/demo_data")
WORK_DIR = os.environ.get("CV_WORKSPACE", "/tmp/demo_workspace")

def main():
    print("=" * 50)
    print("CV DATASET MCP — FULL PIPELINE")
    print("=" * 50)

    # 1. LOAD
    print("\n[1/7] Loading dataset...")
    ds = load_dataset(os.path.join(DATA_DIR, "your_dataset"))
    print(f"  Loaded: {ds.name} ({len(ds)} images, {ds.get_num_annotations()} annotations)")

    # 2. ANALYZE
    print("\n[2/7] Analyzing quality...")
    quality = analyze_quality(ds)
    print(f"  Avg quality: {quality.average_overall:.1f}/100")

    print("\n[3/7] Finding duplicates...")
    dups = find_duplicates(ds, methods=["exact", "perceptual"])
    print(f"  Duplicate groups: {dups.total_duplicate_groups}")

    print("\n[4/7] Analyzing distribution...")
    dist = analyze_distribution(ds)
    print(f"  Classes: {len(dist.class_distribution)}, Gini: {dist.gini_coefficient:.3f}")

    # 3. AUGMENT
    print("\n[5/7] Augmenting (YOLO detection pipeline, 3x)...")
    aug_result = augment_dataset(ds, pipeline="yolo_detection", multiplier=3)
    aug_ds = load_dataset(aug_result.output_dir)
    print(f"  Augmented: {len(aug_ds)} images (was {len(ds)})")

    # 4. TRANSFORM
    print("\n[6/7] Resizing (letterbox 640x640) & Normalizing...")
    resized = resize_dataset(aug_ds, size=(640, 640), strategy="letterbox")
    stats = compute_channel_stats(resized)
    normalized = normalize_dataset(resized, mean=stats["mean"], std=stats["std"])

    # 5. SPLIT
    print("\n[7/7] Stratified split (70/15/15)...")
    split_result = split_dataset(normalized, strategy="stratified", ratios={"train": 0.7, "val": 0.15, "test": 0.15})
    print(f"  Train: {split_result.splits['train']}, Val: {split_result.splits['val']}, Test: {split_result.splits['test']}")

    # 6. EXPORT
    print("\nExporting to PyTorch...")
    export_pytorch(normalized, output_dir=os.path.join(WORK_DIR, "pytorch_export"), image_size=(640, 640))
    print(f"  Exported to {WORK_DIR}/pytorch_export")

    # 7. PIPELINE TEMPLATE
    print("\nRunning pipeline template 'clean_analyze_split'...")
    tpl = get_template("clean_analyze_split")
    result = execute_pipeline(ds, tpl, work_dir=os.path.join(WORK_DIR, "pipeline_out"))
    print(f"  Status: {result.status}, Steps: {result.completed_steps}/{result.total_steps}")

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
