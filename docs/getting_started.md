# PixLint — Getting Started

## Installation

```bash
# Clone and enter the repo
git clone https://github.com/amitsingh-24/PixLint.git
cd pixlint

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install all extras
pip install -e ".[all]"

# Or install specific extras:
#   [torch] [tensorflow] [clip] [umap] [hdf5] [huggingface] [dashboard] [dev] [all]
pip install -e ".[torch,umap]"       # For embeddings + visualization
pip install -e ".[huggingface]"      # For Hugging Face dataset export
pip install -e ".[dev]"              # For development (pytest, ruff, mypy)
```

## Quick Start

### 1. Configure Environment (Required for Security)

```bash
# Add to ~/.bashrc or ~/.zshrc
export CV_DATA_DIR="$HOME/datasets"           # Where your CV datasets live
export CV_WORKSPACE="$HOME/cv-workspace"      # Where outputs go
export CV_SECURITY_LOG_FILE="$HOME/.cv-mcp-security.log"
source ~/.bashrc
```

### 2. Load a Dataset

```python
from pixlint.core.loader import load_dataset

# Auto-detect format (COCO, VOC, YOLO, KITTI, or folder)
ds = load_dataset("/path/to/dataset")

# Or specify format explicitly
ds = load_dataset("/path/to/coco.json", format="coco")
ds = load_dataset("/path/to/voc", format="voc")
ds = load_dataset("/path/to/yolo", format="yolo")

print(f"Dataset: {ds.name}")
print(f"Format:  {ds.format.value}")
print(f"Images:  {len(ds)}")
print(f"Classes: {', '.join(ds.get_class_names())}")
```

### 3. Start the MCP Server

```bash
# From the project root
python -m pixlint.server

# Or if installed
pixlint
```

### 4. Add to MCP Client

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": {
        "CV_DATA_DIR": "/Users/you/datasets",
        "CV_WORKSPACE": "/Users/you/cv-workspace"
      }
    }
  }
}
```

**Cursor / VS Code** (`.cursor/mcp.json` or `.vscode/mcp.json`):
```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": { "CV_DATA_DIR": "/path/to/datasets", "CV_WORKSPACE": "/path/to/workspace" }
    }
  }
}
```

### 5. Use via Natural Language

```
You: "Load my YOLO dataset at /datasets/coco_person and check for duplicates"
Assistant: [calls load_dataset, then find_duplicates_tool]
```

---

## Core Operations

### Analysis

```python
from pixlint.analysis.quality import analyze_quality
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.integrity import check_integrity

# Quality report
quality = analyze_quality(ds)
print(f"Avg quality: {quality.average_overall:.1f}")
print(f"Flagged:     {quality.flagged_images} images")

# Find duplicates
dups = find_duplicates(ds, methods=["exact", "perceptual"])
print(f"Duplicate groups: {dups.total_duplicate_groups}")

# Distribution analysis
dist = analyze_distribution(ds)
print(f"Classes: {len(dist.class_distribution)}")
print(f"Gini coefficient: {dist.gini_coefficient}")

# Integrity check
integrity = check_integrity(ds)
print(f"Corrupt: {integrity.corrupt_images}, Missing labels: {integrity.missing_labels}")
```

### Embeddings & Search

```python
from pixlint.analysis.embeddings import compute_embeddings, semantic_search, cluster_dataset

# Compute embeddings (ResNet50 or CLIP)
emb = compute_embeddings(ds, model="resnet50")  # or "clip-vit-base"
print(f"Embeddings: {emb.embeddings.shape}")

# Semantic search by text
results = semantic_search(ds, query="person riding bicycle", top_k=10)
for r in results:
    print(f"{r.image_id}: {r.similarity:.4f}")

# Cluster dataset
clusters = cluster_dataset(ds, n_clusters=5, model="resnet50")
print(f"Cluster sizes: {clusters.cluster_sizes}")
```

### Augmentation & Transformation

```python
from pixlint.augmentation.pipeline import augment_dataset
from pixlint.transformation.resize import resize_dataset
from pixlint.transformation.normalize import normalize_dataset, compute_channel_stats
from pixlint.transformation.format_converter import convert_format

# Augment with YOLO detection pipeline
aug = augment_dataset(ds, pipeline="yolo_detection", multiplier=3)
aug_ds = load_dataset(aug.output_dir)

# Resize with letterbox
resized = resize_dataset(ds, size=(640, 640), strategy="letterbox")

# Normalize (ImageNet stats or custom)
stats = compute_channel_stats(ds)
normed = normalize_dataset(ds, mean=stats["mean"], std=stats["std"])

# Convert formats (COCO ↔ VOC ↔ YOLO ↔ KITTI ↔ CSV)
converted = convert_format(ds, target_format="yolo", output_dir="./yolo_out")
```

### Splitting & Export

```python
from pixlint.splitting.splitter import split_dataset
from pixlint.export.pytorch import export_pytorch
from pixlint.export.ultralytics import export_ultralytics
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.hdf5 import export_hdf5

# Stratified split
split = split_dataset(ds, strategy="stratified", ratios={"train": 0.7, "val": 0.15, "test": 0.15})

# Export to various frameworks
export_pytorch(ds, output_dir="./pytorch_export", image_size=(640, 640))
export_ultralytics(ds, output_dir="./ultralytics_export", image_size=640)
export_tensorflow(ds, output_dir="./tf_export", image_size=(640, 640))
export_hdf5(ds, output_path="./dataset.h5", image_size=(640, 640))
```

### Pipeline Engine

```python
from pixlint.core.pipeline import execute_pipeline, get_template

# Load pre-built template
tpl = get_template("clean_analyze_split")

# Execute on your dataset
result = execute_pipeline(ds, tpl, work_dir="/workspace/pipeline_out")
print(f"Status: {result.status}")
print(f"Steps: {result.completed_steps}/{result.total_steps}")
```

### Video Processing

```python
from pixlint.analysis.video import extract_frames, video_to_dataset

# Extract frames
frames = extract_frames("video.mp4", output_dir="./frames", frame_interval=30)

# Convert video to dataset
ds = video_to_dataset("video.mp4", output_dir="./video_ds", frame_interval=30, name="my_video")
```

---

## Next Steps

- See [examples/](examples/) for runnable scripts
- Explore [Pipeline Templates](docs/pipeline_templates.md)
- Read [Security Guide](docs/security.md) for production deployment
- Check [API Reference](docs/api_reference.md) for all 67 tools

### Beyond the basics

Newer releases add curation and diagnostics tools that go past raw analysis:

- **Curation** — `filter_dataset` (build a subset), `clean_dataset` (fix corrupt/out-of-bounds/degenerate/duplicate labels), `remap_classes`, and `sample_dataset`.
- **Dataset Doctor** — `dataset_readiness_report` produces a training-readiness report with executable remediation steps.
- **Query** — `query_dataset` turns a natural-language description into a predicate query over your images/annotations.
- **Label errors & slices** — `find_label_errors` surfaces likely mislabels and `discover_slices` finds weak/biased slices of the data.

See the [API Reference](docs/api_reference.md) for the full signatures.