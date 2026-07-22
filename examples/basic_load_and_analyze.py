#!/usr/bin/env python3
"""
PixLint — Quick Example

Load a dataset, analyze it, and print a summary.
"""

import os
from pixlint.core.loader import load_dataset

# Use env var or fallback for demo
DATA_DIR = os.environ.get("CV_DATA_DIR", "/tmp/demo_data")

def main():
    # Load with auto-detect (COCO, VOC, YOLO, KITTI, folder)
    ds = load_dataset(os.path.join(DATA_DIR, "your_dataset"))
    
    # Print info
    info = ds.to_info()
    print(f"Dataset: {info.name}")
    print(f"Format:  {info.format.value}")
    print(f"Images:  {info.num_images}")
    print(f"Anns:    {info.num_annotations}")
    print(f"Classes: {', '.join(info.class_names)}")

    # Analyze quality
    from pixlint.analysis.quality import analyze_quality
    quality = analyze_quality(ds)
    print(f"\nAvg Quality: {quality.average_overall:.1f}/100")
    print(f"Flagged:     {quality.flagged_images} images")

    # Find duplicates
    from pixlint.analysis.duplicates import find_duplicates
    dups = find_duplicates(ds)
    print(f"Duplicate groups: {dups.total_duplicate_groups}")

    # Distribution
    from pixlint.analysis.distribution import analyze_distribution
    dist = analyze_distribution(ds)
    print(f"Imbalance ratio: {dist.imbalance_ratio}")
    print(f"Gini coeff:      {dist.gini_coefficient}")

if __name__ == "__main__":
    main()