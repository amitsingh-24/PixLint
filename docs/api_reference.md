# PixLint — API Reference

## Overview

**103 MCP components** exposed via the Model Context Protocol:
- **67 Tools** — callable operations with input validation, path checking, rate limiting, audit logging
- **23 Resources** — read-only data endpoints (dataset info, statistics, quality scores, class stats, etc.)
- **13 Prompts** — reusable prompt templates for common CV workflows

Runs locally over stdio, or self-hosted over authenticated HTTP (`MCP_TRANSPORT=streamable-http`, bearer token via `CV_MCP_AUTH_TOKEN`, unauthenticated `GET /health`). See [MCP Client Setup](mcp_client_setup.md) for the remote-server config.

All components enforce (wired into **every** tool call at one choke point):
- **Input validation** (dataset_id, format, model names, ratios, etc.)
- **Path validation** (paths confined to CV_DATA_DIR/CV_WORKSPACE on reads AND writes)
- **Rate limiting** (sliding window per tool+dataset)
- **Resource limits** (max-concurrency slot + execution timeout per call)
- **Decompression-bomb guard** (`CV_MAX_IMAGE_PIXELS`)
- **Output sanitization** (credentials redacted; errors returned as strings)
- **Audit logging** (every call logged to CV_SECURITY_LOG_FILE)

---

## Tools (67)

### Dataset Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `load_dataset` | Load a CV dataset from directory | `path`, `format` (auto/coco/voc/yolo/kitti/folder), `name` |
| `list_datasets` | List all loaded datasets | — |
| `dataset_info` | Get detailed info for a loaded dataset | `dataset_id` |
| `compute_statistics_tool` | Image sizes, class counts, annotation density | `dataset_id` |
| `sample_dataset_tool` | Sample image IDs (random/stratified) | `n`, `strategy`, `seed` |
| `unload_dataset_tool` | Unload a dataset from memory + registry | `dataset_id` |
| `detect_format_tool` | Auto-detect dataset format from directory | `path` |
| `load_cloud_dataset_tool` | Load from S3/GCS/Azure | `provider`, `bucket`, `prefix`, `cache_dir`, `name` |
| `list_cloud_objects_tool` | List objects in cloud bucket | `provider`, `bucket`, `prefix`, `endpoint_url` |

### Analysis

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `find_duplicates_tool` | Find duplicate images (exact, perceptual, SSIM, semantic) | `methods`, `thresholds` |
| `analyze_quality_tool` | Per-image quality (blur, exposure, noise, contrast, resolution, color) | `metrics` |
| `check_integrity_tool` | Validate dataset integrity (corrupt, missing labels, bounds) | `checks` |
| `analyze_distribution_tool` | Class dist, spatial heatmap, co-occurrence, complexity | `analyses` |
| `compute_embeddings_tool` | ResNet50/CLIP embeddings | `model` (resnet50/clip-vit-base/clip-vit-large/dinov2) |
| `semantic_search_tool` | Text-to-image search | `query`, `top_k` |
| `cluster_dataset_tool` | KMeans/DBSCAN clustering | `n_clusters`, `method` |
| `detect_outliers_tool` | Outlier detection (embedding/resolution/annotation) | `methods`, `contamination` |
| `dataset_health_score_tool` | 0-100 health score with 5-axis breakdown | — |
| `dataset_diff_tool` | Compare two datasets | `dataset_id_a`, `dataset_id_b` |
| `generate_captions_tool` | BLIP/ResNet/CLIP captions or tags | `model`, `batch_size` |
| `auto_tag_dataset_tool` | Auto-tag with ResNet/CLIP/EfficientNet | `method` |
| `enrich_metadata_tool` | Write captions/tags into image metadata | `caption_model`, `tag_method` |

### 🩺 Data Intelligence (differentiators)

Advanced dataset-debugging capabilities that incumbents gate behind paid/GUI/SaaS tiers.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `dataset_readiness_report_tool` | **Dataset Doctor** — full diagnostic battery → prioritized, executable remediation plan + runnable pipeline JSON | `dataset_id` |
| `find_label_errors_tool` | Detect likely-**mislabeled** annotations via embedding kNN neighbor-consistency (needs `[torch]`) | `k`, `threshold`, `model` |
| `query_dataset_tool` | **Natural-language → predicate query** over class/size/quality/position/object-count/CLIP-similarity; returns matching image IDs | `filters` (dict) |
| `discover_slices_tool` | Weak-slice / **bias discovery** (class × brightness × size × aspect) with collect/augment recommendations | `dataset_id` |

`query_dataset_tool` `filters` keys: `classes`, `exclude_classes`, `min_annotations`, `max_annotations`, `min_width`/`max_width`/`min_height`/`max_height`, `min_area`/`max_area`, `quality_issues`, `min_quality`/`max_quality`, `position` (left/right/top/bottom/center), `position_class`, `similar_to_text`, `limit`.

### ✂️ Curation (produce a NEW dataset)

Turn "detect" into "fix": these produce a new, registered dataset (in-memory, or COCO on disk when `output_dir` is given).

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `filter_dataset_tool` | Subset by class / box-area / annotation presence / explicit IDs | `classes`, `min_area`, `max_area`, `has_annotations`, `image_ids`, `output_dir` |
| `clean_dataset_tool` | Drop corrupt images, clip out-of-bounds boxes, drop degenerate boxes, remove duplicates | `drop_corrupt`, `clip_out_of_bounds`, `drop_degenerate`, `drop_duplicates`, `output_dir` |
| `remap_classes_tool` | Rename / merge / drop classes (map label→"" to drop) | `mapping`, `drop_unmapped`, `output_dir` |
| `auto_label_tool` | Auto-label with a pretrained COCO-80 detector → pre-annotated dataset (needs `[torch]`) | `model`, `confidence`, `max_detections`, `output_dir` |

### Augmentation & Transformation

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `augment_dataset_tool` | YOLO/cls/seg augmentation pipelines | `pipeline`, `multiplier`, `output_dir` |
| `preview_augmentation_tool` | Preview augmentation on samples | `pipeline`, `n_samples` |
| `convert_format_tool` | COCO↔VOC↔YOLO↔KITTI↔CSV | `target_format`, `output_dir`, `image_size` |
| `resize_dataset_tool` | Letterbox/stretch/crop | `size`, `strategy`, `output_dir` |
| `normalize_dataset_tool` | Channel-wise normalization | `mean`, `std`, `output_dir` |
| `compute_channel_stats_tool` | Dataset-wide channel mean/std | — |

### Splitting

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `split_dataset_tool` | Random/stratified/temporal/grouped | `strategy`, `ratios`, `seed` |
| `generate_kfold_tool` | K-fold cross-validation | `k`, `strategy` |
| `detect_leakage_tool` | MD5-based train/test leak detection | — |

### Export

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `export_dataset_tool` | PyTorch/TensorFlow/Ultralytics/HDF5 | `format`, `output_dir`, `image_size` |
| `export_webdataset_tool` | WebDataset tar shards | `output_dir`, `image_size`, `shard_size` |
| `export_fiftyone_tool` | FiftyOne format (COCO/VOC/YOLO/...) | `output_dir`, `export_format` |
| `export_cvat_xml_tool` | CVAT XML 1.1 | `output_path` |
| `export_labelme_json_tool` | LabelMe JSON (one file per image) | `output_dir` |
| `export_huggingface_tool` | Hugging Face `datasets` export + optional `push_to_hub` (needs `[huggingface]`; `HF_TOKEN` env to publish) | `output_dir`, `push_to_hub`, `repo_id`, `private` |

### Visualization

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `preview_images_tool` | Image grid with annotations | `n_samples`, `show_annotations` |
| `preview_single_image_tool` | Single image with annotation details | `image_id` |
| `plot_distribution_tool` | Class distribution chart | `chart_type` (bar/pie/treemap/radar) |
| `plot_quality_scores_tool` | Quality histograms + flagged pie | — |
| `plot_spatial_heatmap_tool` | Spatial object distribution | — |
| `plot_duplicate_groups_tool` | Duplicate group analysis | — |

### Pipeline

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `execute_pipeline_tool` | Execute a pipeline definition | `pipeline`, `work_dir` |
| `execute_template_tool` | Execute a pre-built template | `template_id`, `work_dir` |
| `register_pipeline_tool` | Register pipeline for reuse | `pipeline` |
| `list_pipelines_tool` | List registered pipelines | — |
| `save_pipeline_tool` | Save pipeline to JSON file | `pipeline_id`, `path` |
| `load_pipeline_tool` | Load pipeline from JSON file | `path` |
| `list_templates_tool` | List pre-built templates | — |
| `get_template_tool` | Get template definition | `template_id` |

### Video Processing

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `extract_frames_tool` | Extract frames from video | `video_path`, `output_dir`, `frame_interval`, `max_frames`, `resize` |
| `video_to_dataset_tool` | Convert video to CVDataset | `video_path`, `output_dir`, `frame_interval`, `max_frames`, `resize`, `name` |
| `batch_video_to_datasets_tool` | Batch convert all videos in dir | `video_dir`, `output_base`, `frame_interval`, `max_frames`, `resize` |
| `temporal_split_tool` | Temporal split for video frames | `ratios` |

### Active Learning

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `uncertainty_sampling_tool` | Entropy/margin/least-confidence | `method`, `n`, `model` |
| `diversity_sampling_tool` | Core-set / k-means++ diversity | `n`, `model` |
| `query_strategy_tool` | Combined uncertainty + diversity | `strategy`, `n`, `model`, `alpha` |

### Advanced Operations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `dataset_diff_tool` | Compare two datasets | `dataset_id_a`, `dataset_id_b` |
| `merge_datasets_tool` | Merge multiple datasets | `dataset_ids`, `strategy`, `deduplicate`, `unify_labels` |

---

## Resources (23)

Resources are read-only data endpoints accessed via URI templates.

| Resource URI | Description |
|--------------|-------------|
| `dataset://{dataset_id}/info` | Dataset metadata (name, format, counts, classes) |
| `dataset://{dataset_id}/statistics` | Image sizes, class counts, annotation density |
| `dataset://{dataset_id}/images` | All images: image_id, path, width, height |
| `dataset://{dataset_id}/annotations` | All annotations: image_id, label, bbox, mask flag |
| `dataset://{dataset_id}/quality` | Full quality analysis report |
| `dataset://{dataset_id}/duplicates` | Duplicate detection report |
| `dataset://{dataset_id}/distribution` | Full distribution analysis |
| `dataset://{dataset_id}/health` | Health score with 5-axis breakdown |
| `dataset://list` | All loaded datasets |
| `dataset://{dataset_id}/quality/scores` | Per-image quality scores (CSV) |
| `dataset://{dataset_id}/quality/flagged` | Images below quality threshold (50) |
| `dataset://{dataset_id}/class_stats` | Per-class counts, annotations, %, avg area |
| `dataset://{dataset_id}/class_balance` | Imbalance ratio, entropy, Gini coefficient |
| `dataset://{dataset_id}/spatial_heatmap` | Spatial heatmap grid (CSV) |
| `dataset://{dataset_id}/co_occurrence` | Class co-occurrence matrix |
| `dataset://{dataset_id}/complexity` | Per-image complexity scores |
| `dataset://{dataset_id}/duplicate_groups` | Detailed duplicate groups |
| `dataset://{dataset_id}/integrity` | Integrity check issues |
| `dataset://{dataset_id}/split/{split_name}` | Images in a specific split |
| `dataset://{dataset_id}/annotations/by_class` | Annotations grouped by class |
| `dataset://{dataset_id}/bbox_stats` | BBox area/aspect ratio statistics |
| `dataset://{dataset_id}/split_info` | Dataset split information |
| `dataset://{dataset_id}/export_config` | Export configs for PyTorch/TF/Ultralytics/ONNX |

---

## Prompts (13)

| Prompt | Description |
|--------|-------------|
| `prepare_yolo_training_prompt` | End-to-end YOLO training prep (integrity → convert → split → export) |
| `dataset_curation_prompt` | Audit → clean/remap/filter → re-audit → split/export a training-ready dataset |
| `publish_dataset_prompt` | Clean, validate, and publish a dataset to the Hugging Face Hub |
| `export_pipeline_prompt` | Export to PyTorch/TF/Ultralytics/HDF5 |
| `active_learning_prompt` | Uncertainty/diversity/combined sample selection |
| `prepare_coco_training_prompt` | COCO-format training (Faster R-CNN, DETR, Mask R-CNN) |
| `prepare_segmentation_training_prompt` | Segmentation training (Mask R-CNN, DeepLab, U-Net) |
| `dataset_versioning_prompt` | Compare old vs new dataset versions |
| `annotation_quality_improvement_prompt` | Fix annotation quality issues |
| `custom_augmentation_policy_prompt` | Design custom augmentation for detection/cls/seg |
| `multi_dataset_merge_prompt` | Merge multiple datasets with label unification |
| `dataset_migration_prompt` | Convert between COCO/VOC/YOLO/KITTI/CSV |
| `dataset_health_monitoring_prompt` | Set up periodic health monitoring |

---

## Validation Rules Summary

| Parameter | Rule |
|-----------|------|
| `dataset_id` | `^[a-zA-Z0-9_\-.]{1,200}$` |
| `format` | Whitelist: coco, voc, yolo, kitti, csv, folder, tfrecord, webdataset, labelme, cvat, fiftyone, pytorch, tensorflow, ultralytics, hdf5 |
| `model` | Whitelist: resnet50, resnet101, clip-vit-base, clip-vit-large, dinov2, sam, sam2, grounding-dino, blip, resnet, clip, efficientnet |
| `ratios` | Dict of positive floats summing to 1.0 (±0.001) |
| `image_size` | `[width, height]`, both positive ints |
| `bbox` | `[x1, y1, x2, y2]`, x2>x1, y2>y1, all ≥0 |
| `paths` | Must resolve within CV_DATA_DIR or CV_WORKSPACE |
| `size` (multiplier) | Positive int ≤ 100 (augmentation) or ≤ 100000 (shard) |

---

## Error Responses

All tools return string. Errors are sanitized:

```
"Error: Dataset 'invalid_id' not found."
"Error: Path not within allowed directories: /etc/passwd. Allowed: ['/home/user/datasets']"
"Error: Invalid format: 'invalid'. Supported: ['coco', 'voc', 'yolo', ...]"
"Error: Rate limit exceeded for load_dataset"
"Error: Resource limit exceeded: tool ran for 301.2s"
```