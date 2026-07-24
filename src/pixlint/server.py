from __future__ import annotations

import json
import os
import threading
from functools import wraps
from typing import Any

from mcp.server.fastmcp import FastMCP

from pixlint.analysis.active_learning import (
    diversity_sampling,
    query_strategy,
    uncertainty_sampling,
)
from pixlint.analysis.autolabel import auto_label
from pixlint.analysis.captioning import auto_tag_dataset, enrich_metadata, generate_captions
from pixlint.analysis.diff import dataset_diff
from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.embeddings import cluster_dataset, compute_embeddings, semantic_search
from pixlint.analysis.health import dataset_health_score
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.label_errors import find_label_errors
from pixlint.analysis.outliers import detect_outliers
from pixlint.analysis.quality import analyze_quality
from pixlint.analysis.query import query_dataset
from pixlint.analysis.readiness import dataset_readiness_report
from pixlint.analysis.slices import discover_slices
from pixlint.analysis.statistics import compute_statistics, sample_dataset
from pixlint.analysis.video import (
    extract_frames,
    temporal_split,
    video_batch_to_datasets,
    video_to_dataset,
)
from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.core.cloud import list_s3_objects, load_cloud_dataset
from pixlint.core.curation import (
    clean_dataset as _clean_dataset,
)
from pixlint.core.curation import (
    filter_dataset as _filter_dataset,
)
from pixlint.core.curation import (
    remap_classes as _remap_classes,
)
from pixlint.core.loader import CVDataset, detect_format
from pixlint.core.loader import load_dataset as load_dataset_impl
from pixlint.core.merge import merge_datasets
from pixlint.core.metadata import (
    get_dataset_info,
    register_dataset,
)
from pixlint.core.metadata import (
    list_datasets as _list_datasets,
)
from pixlint.core.metadata import (
    remove_dataset as _remove_dataset,
)
from pixlint.core.pipeline import (
    execute_pipeline,
    get_pipeline,
    get_template,
    list_pipelines,
    list_templates,
    load_pipeline_from_json,
    register_pipeline,
    save_pipeline_to_json,
)
from pixlint.export.extra_formats import (
    export_cvat_xml,
    export_fiftyone,
    export_labelme_json,
    export_webdataset,
)
from pixlint.export.hdf5 import export_hdf5
from pixlint.export.huggingface import export_huggingface
from pixlint.export.pytorch import export_pytorch
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.ultralytics import export_ultralytics
from pixlint.splitting.cross_validation import generate_kfold_splits
from pixlint.splitting.leakage import detect_leakage
from pixlint.splitting.splitter import split_dataset
from pixlint.transformation.format_converter import convert_format
from pixlint.transformation.normalize import compute_channel_stats, normalize_dataset
from pixlint.transformation.resize import resize_dataset
from pixlint.utils.schemas import DatasetFormat, PipelineDefinition
from pixlint.utils.security import (
    RESOURCE_LIMITER,
    SECURITY_AUDITOR,
    TOOL_RATE_LIMITER,
    SecurityError,
    sanitize_output,
    validate_dataset_id,
    validate_dataset_path,
    validate_file_path,
    validate_format,
    validate_output_path,
    validate_positive_int,
    validate_ratio_dict,
)
from pixlint.visualization.charts import (
    plot_distribution,
    plot_duplicate_groups,
    plot_quality_scores,
    plot_spatial_heatmap,
)
from pixlint.visualization.previews import preview_images, preview_single_image

mcp = FastMCP("pixlint")

_dataset_store: dict[str, CVDataset] = {}
_store_lock = threading.Lock()


def _get_or_load_dataset(dataset_id: str) -> CVDataset | None:
    with _store_lock:
        existing = _dataset_store.get(dataset_id)
        if existing is not None:
            return existing
    info = get_dataset_info(dataset_id)
    if info is None:
        return None
    dataset = CVDataset(info.path, format=DatasetFormat(info.format))
    with _store_lock:
        if dataset_id not in _dataset_store:
            _dataset_store[dataset_id] = dataset
        return _dataset_store[dataset_id]


def _get_dataset(dataset_id: str) -> CVDataset:
    """Get dataset, assuming it exists (already checked)."""
    return _get_or_load_dataset(dataset_id)  # type: ignore[return-value]


@mcp.tool(name="load_dataset")
def load_dataset(path: str, format: str = "auto", name: str | None = None) -> str:
    """Load a computer vision dataset from a directory"""
    # Validate input path
    validated_path = validate_dataset_path(path)
    # Validate format
    validated_format = validate_format(format)
    # Validate dataset ID/name if provided
    if name:
        name = validate_dataset_id(name)

    dataset = load_dataset_impl(str(validated_path), format=validated_format, name=name)
    dataset_id = register_dataset(dataset.to_info())
    with _store_lock:
        _dataset_store[dataset_id] = dataset

    # Sanitize output
    return sanitize_output(dataset.to_info().model_dump_json(indent=2))


@mcp.tool(name="list_datasets")
def list_datasets() -> str:
    """List all loaded datasets"""
    datasets = _list_datasets()
    if not datasets:
        return "No datasets loaded."
    result: Any = [d.model_dump(mode="json") for d in datasets]
    return str(result)


@mcp.tool()
def dataset_info(dataset_id: str) -> str:
    """Get information about a loaded dataset"""
    dataset_id = validate_dataset_id(dataset_id)
    info = get_dataset_info(dataset_id)
    if info is None:
        return f"Dataset '{dataset_id}' not found."
    return info.model_dump_json(indent=2)


@mcp.tool()
def compute_statistics_tool(dataset_id: str) -> str:
    """Compute dataset statistics (image sizes, class counts, annotation density)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = compute_statistics(dataset)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def sample_dataset_tool(dataset_id: str, n: int = 10, strategy: str = "random", seed: int | None = 42) -> str:
    """Sample image IDs from a dataset (random or stratified)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    n = validate_positive_int(n, max_value=100000)
    if strategy not in ("random", "stratified"):
        return sanitize_output(f"Invalid strategy: {strategy}. Use random or stratified.")
    if seed is not None:
        seed = validate_positive_int(seed, max_value=2**31 - 1)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = sample_dataset(dataset, n=n, strategy=strategy, seed=seed)
    return sanitize_output(str({"dataset_id": validated_dataset_id, "sampled_image_ids": result}))


@mcp.tool()
def unload_dataset_tool(dataset_id: str) -> str:
    """Unload a dataset from memory and remove it from the registry"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    with _store_lock:
        _dataset_store.pop(validated_dataset_id, None)
    removed = _remove_dataset(validated_dataset_id)
    if removed:
        return sanitize_output(f"Dataset '{validated_dataset_id}' unloaded.")
    return sanitize_output(f"Dataset '{validated_dataset_id}' was not loaded.")


@mcp.tool()
def find_duplicates_tool(dataset_id: str, methods: list[str] | None = None, thresholds: dict | None = None) -> str:
    """Find duplicate images (exact, perceptual hash, SSIM, semantic)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if methods:
        valid_methods = {"exact", "perceptual", "ssim", "semantic"}
        for m in methods:
            if m not in valid_methods:
                return sanitize_output(f"Invalid method: {m}. Must be one of: {valid_methods}")
    if thresholds:
        for k, v in thresholds.items():
            if not isinstance(v, (int, float)) or v < 0 or v > 1:
                return sanitize_output(f"Invalid threshold for {k}: {v}. Must be between 0 and 1")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = find_duplicates(dataset, methods=methods, thresholds=thresholds)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def analyze_quality_tool(dataset_id: str, metrics: list[str] | None = None) -> str:
    """Analyze image quality (blur, exposure, noise, contrast, resolution, color)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if metrics:
        valid_metrics = {"blur", "exposure", "noise", "contrast", "resolution", "color"}
        for m in metrics:
            if m not in valid_metrics:
                return sanitize_output(f"Invalid metric: {m}. Must be one of: {valid_metrics}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = analyze_quality(dataset, metrics=metrics)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def check_integrity_tool(dataset_id: str, checks: list[str] | None = None) -> str:
    """Check dataset integrity (corrupt images, missing labels, annotation bounds)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if checks:
        valid_checks = {"corrupt", "missing_labels", "annotation_bounds", "empty_images"}
        for c in checks:
            if c not in valid_checks:
                return sanitize_output(f"Invalid check: {c}. Must be one of: {valid_checks}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = check_integrity(dataset, checks=checks)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def analyze_distribution_tool(dataset_id: str, analyses: list[str] | None = None) -> str:
    """Analyze class distribution, spatial heatmap, label co-occurrence, complexity"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if analyses:
        valid_analyses = {"class_dist", "spatial", "cooccurrence", "complexity"}
        for a in analyses:
            if a not in valid_analyses:
                return sanitize_output(f"Invalid analysis: {a}. Must be one of: {valid_analyses}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = analyze_distribution(dataset, analyses=analyses)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def detect_format_tool(path: str) -> str:
    """Auto-detect the format of a dataset directory"""
    path = str(validate_dataset_path(path))
    fmt = detect_format(path)
    return sanitize_output(str({"path": path, "detected_format": fmt}))


@mcp.tool()
def compute_embeddings_tool(dataset_id: str, model: str = "resnet50") -> str:
    """Compute image embeddings using ResNet50 or CLIP"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_models = {"resnet50", "clip-vit-base", "clip-vit-large", "dinov2"}
    if model not in valid_models:
        return sanitize_output(f"Invalid model: {model}. Must be one of: {valid_models}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = compute_embeddings(dataset, model=model)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def semantic_search_tool(dataset_id: str, query: str, top_k: int = 10) -> str:
    """Search images by semantic similarity to a text query"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    top_k = validate_positive_int(top_k, max_value=1000)
    if not query or not query.strip():
        return sanitize_output("Query cannot be empty")
    query = sanitize_output(query.strip())

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    results: list[Any] = semantic_search(dataset, query=query, top_k=top_k)
    return sanitize_output(str([r.model_dump() for r in results]))


@mcp.tool()
def cluster_dataset_tool(dataset_id: str, n_clusters: int = 5, method: str = "kmeans") -> str:
    """Cluster dataset images using embeddings"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    n_clusters = validate_positive_int(n_clusters, max_value=100)
    if method not in ("kmeans", "dbscan"):
        return sanitize_output(f"Invalid method: {method}. Must be one of: kmeans, dbscan")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = cluster_dataset(dataset, n_clusters=n_clusters, method=method)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def detect_outliers_tool(dataset_id: str, methods: list[str] | None = None, contamination: float = 0.05) -> str:
    """Detect outlier images in dataset"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_methods = {"embedding", "resolution", "annotation"}
    if methods:
        for m in methods:
            if m not in valid_methods:
                return sanitize_output(f"Invalid method: {m}. Must be one of: {valid_methods}")
    if contamination < 0 or contamination > 1:
        return sanitize_output("Contamination must be between 0 and 1")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = detect_outliers(dataset, methods=methods, contamination=contamination)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def dataset_health_score_tool(dataset_id: str) -> str:
    """Compute overall dataset health score"""
    validated_dataset_id = validate_dataset_id(dataset_id)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = dataset_health_score(dataset)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def generate_kfold_tool(dataset_id: str, k: int = 5, strategy: str = "stratified") -> str:
    """Generate K-fold cross-validation splits"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    k = validate_positive_int(k, max_value=100)
    valid_strategies = {"stratified", "random", "group"}
    if strategy not in valid_strategies:
        return sanitize_output(f"Invalid strategy: {strategy}. Must be one of: {valid_strategies}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = generate_kfold_splits(dataset, k=k, strategy=strategy)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def detect_leakage_tool(dataset_id: str) -> str:
    """Detect data leakage between train/test splits"""
    validated_dataset_id = validate_dataset_id(dataset_id)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = detect_leakage(dataset)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def preview_images_tool(dataset_id: str, n_samples: int = 16, show_annotations: bool = True) -> str:
    """Preview dataset images in a grid with annotation overlays"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    n_samples = validate_positive_int(n_samples, max_value=100)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = preview_images(dataset, n_samples=n_samples, show_annotations=show_annotations)
    return sanitize_output(str(result))


@mcp.tool()
def preview_single_image_tool(dataset_id: str, image_id: str) -> str:
    """Preview a single image with annotation details"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_image_id = validate_dataset_id(image_id)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = preview_single_image(dataset, validated_image_id)
    return sanitize_output(str(result))


@mcp.tool()
def augment_dataset_tool(dataset_id: str, pipeline: str = "yolo_detection", multiplier: int = 3, output_dir: str | None = None) -> str:
    """Apply augmentation pipeline to a dataset"""
    dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{dataset_id}' not found.")
    if output_dir:
        output_dir = str(validate_output_path(output_dir))
    result = augment_dataset(dataset, pipeline=pipeline, multiplier=multiplier, output_dir=output_dir)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def preview_augmentation_tool(dataset_id: str, pipeline: str = "yolo_detection", n_samples: int = 8) -> str:
    """Preview augmentation results for sample images"""
    dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{dataset_id}' not found.")
    result = preview_augmentation(dataset, pipeline=pipeline, n_samples=n_samples)
    return sanitize_output(str(result))


@mcp.tool()
def convert_format_tool(dataset_id: str, target_format: str, output_dir: str, image_size: list[int] | None = None) -> str:
    """Convert dataset between formats (COCO, VOC, YOLO, KITTI, CSV)"""
    # Validate dataset ID
    validated_dataset_id = validate_dataset_id(dataset_id)
    # Validate target format
    validated_format = validate_format(target_format)
    # Validate output path
    validated_output = validate_output_path(output_dir, allow_create=True)
    # Validate image_size if provided
    if image_size:
        if len(image_size) != 2:
            return sanitize_output("image_size must be [width, height]")
        image_size = tuple(image_size)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = convert_format(dataset, target_format=validated_format, output_dir=str(validated_output), image_size=image_size)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def resize_dataset_tool(dataset_id: str, size: list[int], strategy: str = "letterbox", output_dir: str | None = None) -> str:
    """Resize dataset images (letterbox, stretch, or crop)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if len(size) != 2:
        return sanitize_output("size must be [width, height]")
    size_tuple = tuple(size)
    # Validate strategy
    if strategy not in ("letterbox", "stretch", "crop"):
        return sanitize_output(f"Invalid strategy: {strategy}. Use letterbox, stretch, or crop.")
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = resize_dataset(dataset, size=size_tuple, strategy=strategy, output_dir=output_dir)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def normalize_dataset_tool(dataset_id: str, mean: list[float] | None = None, std: list[float] | None = None, output_dir: str | None = None) -> str:
    """Normalize dataset images using channel mean/std"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = normalize_dataset(dataset, mean=mean, std=std, output_dir=output_dir)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def compute_channel_stats_tool(dataset_id: str) -> str:
    """Compute dataset channel-wise mean and std"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = compute_channel_stats(dataset)
    return sanitize_output(str(result))


@mcp.tool()
def split_dataset_tool(dataset_id: str, strategy: str = "stratified", ratios: dict[str, float] | None = None, seed: int | None = None) -> str:
    """Split dataset into train/val/test sets"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if ratios:
        ratios = validate_ratio_dict(ratios)
    if seed is not None:
        seed = validate_positive_int(seed, max_value=2**31 - 1)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = split_dataset(dataset, strategy=strategy, ratios=ratios, seed=seed)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def export_dataset_tool(dataset_id: str, format: str, output_dir: str, image_size: list[int] | None = None) -> str:
    """Export dataset in various formats (pytorch, tensorflow, ultralytics, hdf5)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_output = validate_output_path(output_dir, allow_create=True)
    if image_size:
        if len(image_size) != 2:
            return sanitize_output("image_size must be [width, height]")
        image_size = tuple(image_size)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")

    exporters = {
        "pytorch": lambda: export_pytorch(dataset, output_dir=str(validated_output), image_size=image_size or (640, 640)),
        "tensorflow": lambda: export_tensorflow(dataset, output_dir=str(validated_output), image_size=image_size or (640, 640)),
        "ultralytics": lambda: export_ultralytics(dataset, output_dir=str(validated_output), image_size=(image_size or (640, 640))[0]),
        "hdf5": lambda: export_hdf5(dataset, output_path=os.path.join(str(validated_output), "dataset.h5"), image_size=image_size or (640, 640)),
    }
    exporter = exporters.get(format)
    if exporter is None:
        return sanitize_output(f"Unknown export format: {format}. Use pytorch, tensorflow, ultralytics, or hdf5.")
    result = exporter()
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def plot_distribution_tool(dataset_id: str, chart_type: str = "bar") -> str:
    """Generate class distribution chart (bar, pie, treemap, radar)"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if chart_type not in ("bar", "pie", "treemap", "radar"):
        return sanitize_output(f"Invalid chart_type: {chart_type}. Use bar, pie, treemap, or radar.")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = plot_distribution(dataset, chart_type=chart_type)
    return sanitize_output(str(result))


@mcp.tool()
def plot_quality_scores_tool(dataset_id: str) -> str:
    """Generate quality score distribution chart"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = plot_quality_scores(dataset)
    return sanitize_output(str(result))


@mcp.tool()
def plot_spatial_heatmap_tool(dataset_id: str) -> str:
    """Generate spatial object distribution heatmap"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = plot_spatial_heatmap(dataset)
    return sanitize_output(str(result))


@mcp.tool()
def plot_duplicate_groups_tool(dataset_id: str) -> str:
    """Generate duplicate group analysis charts"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = plot_duplicate_groups(dataset)
    return sanitize_output(str(result))


@mcp.tool()
def dataset_diff_tool(dataset_id_a: str, dataset_id_b: str) -> str:
    """Compare two datasets and show differences"""
    validated_a = validate_dataset_id(dataset_id_a)
    validated_b = validate_dataset_id(dataset_id_b)

    dataset_a = _get_or_load_dataset(validated_a)
    dataset_b = _get_or_load_dataset(validated_b)
    if dataset_a is None:
        return sanitize_output(f"Dataset '{validated_a}' not found.")
    if dataset_b is None:
        return sanitize_output(f"Dataset '{validated_b}' not found.")
    result = dataset_diff(dataset_a, dataset_b)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def merge_datasets_tool(dataset_ids: list[str], merged_name: str = "merged_dataset", strategy: str = "union", deduplicate: bool = True, unify_labels: bool = True) -> str:
    """Merge multiple datasets into one with label unification and deduplication"""
    # Validate dataset IDs
    validated_ids = [validate_dataset_id(ds_id) for ds_id in dataset_ids]
    # Validate merged_name
    merged_name = validate_dataset_id(merged_name)

    datasets: list[CVDataset] = []
    for ds_id in validated_ids:
        ds = _get_or_load_dataset(ds_id)
        if ds is None:
            return sanitize_output(f"Dataset '{ds_id}' not found.")
        datasets.append(ds)
    result = merge_datasets(datasets, merged_name=merged_name, strategy=strategy, deduplicate=deduplicate, unify_labels=unify_labels)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def find_label_errors_tool(dataset_id: str, k: int = 10, threshold: float = 0.6, model: str = "resnet50") -> str:
    """Find likely-MISLABELED annotations via embedding neighbor-consistency (a 'data linter'). Requires the [torch] extra."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    k = validate_positive_int(k, max_value=100)
    if threshold < 0 or threshold > 1:
        return sanitize_output("threshold must be between 0 and 1")
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = find_label_errors(dataset, k=k, threshold=threshold, model=model)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def query_dataset_tool(dataset_id: str, filters: dict) -> str:
    """Query a dataset with structured predicates (class/size/quality/position/object-count/text-similarity) and return matching image IDs. The agent translates a natural-language request into `filters`. Keys: classes, exclude_classes, min_annotations, max_annotations, min_width, max_width, min_height, max_height, min_area, max_area, quality_issues, min_quality, max_quality, position (left|right|top|bottom|center), position_class, similar_to_text, limit."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if not isinstance(filters, dict) or not filters:
        return sanitize_output("filters must be a non-empty predicate dictionary.")
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = query_dataset(dataset, filters)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def dataset_readiness_report_tool(dataset_id: str) -> str:
    """The 'Dataset Doctor': run the full diagnostic battery and return a prioritized, EXECUTABLE remediation plan plus a runnable pipeline JSON."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = dataset_readiness_report(dataset)
    return sanitize_output(report.model_dump_json(indent=2))


@mcp.tool()
def discover_slices_tool(dataset_id: str) -> str:
    """Discover weak (under-represented or low-quality) data slices across class/brightness/size/aspect so you know what to collect or augment next."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    report = discover_slices(dataset)
    return sanitize_output(report.model_dump_json(indent=2))


def _register_curated(new_ds: CVDataset) -> None:
    """Register a curation/auto-label output dataset so other tools can use it."""
    register_dataset(new_ds.to_info())
    with _store_lock:
        _dataset_store[new_ds.dataset_id] = new_ds


@mcp.tool()
def filter_dataset_tool(dataset_id: str, classes: list[str] | None = None, min_area: float | None = None, max_area: float | None = None, has_annotations: bool | None = None, image_ids: list[str] | None = None, output_dir: str | None = None) -> str:
    """Create a NEW filtered subset dataset (by class, box area, presence of annotations, or explicit image IDs)."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    new_ds, result = _filter_dataset(dataset, classes=classes, min_area=min_area, max_area=max_area, has_annotations=has_annotations, image_ids=image_ids, output_dir=output_dir)
    _register_curated(new_ds)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def clean_dataset_tool(dataset_id: str, drop_corrupt: bool = True, clip_out_of_bounds: bool = True, drop_degenerate: bool = True, drop_duplicates: bool = False, output_dir: str | None = None) -> str:
    """Create a NEW cleaned dataset: drop corrupt images, clip out-of-bounds boxes, drop degenerate boxes, optionally remove duplicates."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    new_ds, result = _clean_dataset(dataset, drop_corrupt=drop_corrupt, clip_out_of_bounds=clip_out_of_bounds, drop_degenerate=drop_degenerate, drop_duplicates=drop_duplicates, output_dir=output_dir)
    _register_curated(new_ds)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def remap_classes_tool(dataset_id: str, mapping: dict[str, str], drop_unmapped: bool = False, output_dir: str | None = None) -> str:
    """Create a NEW dataset with classes renamed/merged/dropped per the given old->new label mapping."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if not isinstance(mapping, dict) or not mapping:
        return sanitize_output("mapping must be a non-empty {old_label: new_label} dictionary.")
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    new_ds, result = _remap_classes(dataset, mapping=mapping, drop_unmapped=drop_unmapped, output_dir=output_dir)
    _register_curated(new_ds)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def auto_label_tool(dataset_id: str, model: str = "fasterrcnn", confidence: float = 0.5, max_detections: int = 100, output_dir: str | None = None) -> str:
    """Auto-label images with a pretrained detector (COCO-80), producing a NEW pre-annotated dataset. Requires the [torch] extra."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if confidence < 0 or confidence > 1:
        return sanitize_output("confidence must be between 0 and 1")
    max_detections = validate_positive_int(max_detections, max_value=1000)
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    try:
        new_ds, result = auto_label(dataset, model=model, confidence=confidence, max_detections=max_detections, output_dir=output_dir)
    except ImportError as e:
        return sanitize_output(f"Missing dependency: {e}")
    _register_curated(new_ds)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def export_huggingface_tool(dataset_id: str, output_dir: str | None = None, push_to_hub: bool = False, repo_id: str | None = None, private: bool = True) -> str:
    """Export dataset to Hugging Face `datasets` format and optionally publish it to the Hub. Requires the [huggingface] extra; HF_TOKEN env for pushing."""
    validated_dataset_id = validate_dataset_id(dataset_id)
    if output_dir:
        output_dir = str(validate_output_path(output_dir, allow_create=True))
    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    try:
        result = export_huggingface(dataset, output_dir=output_dir, push_to_hub=push_to_hub, repo_id=repo_id, private=private)
    except ImportError as e:
        return sanitize_output(f"Missing dependency: {e}")
    except ValueError as e:
        return sanitize_output(f"Error: {e}")
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def load_cloud_dataset_tool(provider: str = "s3", bucket: str = "", prefix: str = "", anon: bool = False, endpoint_url: str | None = None, connection_string: str | None = None, cache_dir: str | None = None, name: str | None = None) -> str:
    """Load a dataset from cloud storage (S3, GCS, Azure)"""
    # Validate cache_dir if provided
    if cache_dir:
        cache_dir = str(validate_output_path(cache_dir, allow_create=True))
    # Validate name if provided
    if name:
        name = validate_dataset_id(name)

    try:
        dataset = load_cloud_dataset(provider=provider, bucket=bucket, prefix=prefix, anon=anon, endpoint_url=endpoint_url, connection_string=connection_string, cache_dir=cache_dir, name=name)
        dataset_id = register_dataset(dataset.to_info())
        with _store_lock:
            _dataset_store[dataset_id] = dataset
        return sanitize_output(dataset.to_info().model_dump_json(indent=2))
    except ImportError as e:
        return sanitize_output(f"Missing dependency: {e}")
    except Exception as e:
        return sanitize_output(f"Error loading cloud dataset: {e}")


@mcp.tool()
def list_cloud_objects_tool(provider: str = "s3", bucket: str = "", prefix: str = "", anon: bool = False, endpoint_url: str | None = None) -> str:
    """List objects in a cloud storage bucket"""
    try:
        if provider == "s3":
            objects = list_s3_objects(bucket, prefix=prefix, anon=anon, endpoint_url=endpoint_url)
        else:
            from pixlint.core.cloud import _list_objects
            objects = _list_objects(provider, bucket, prefix, anon, endpoint_url, None)
        return sanitize_output(str({"provider": provider, "bucket": bucket, "objects": objects}))
    except ImportError as e:
        return sanitize_output(f"Missing dependency: {e}")
    except Exception as e:
        return sanitize_output(f"Error listing objects: {e}")


@mcp.tool()
def generate_captions_tool(dataset_id: str, model: str = "blip", batch_size: int = 8) -> str:
    """Generate captions for dataset images using BLIP or tags using ResNet"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_models = {"blip", "resnet", "clip"}
    if model not in valid_models:
        return sanitize_output(f"Invalid model: {model}. Must be one of: {valid_models}")
    batch_size = validate_positive_int(batch_size, max_value=256)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = generate_captions(dataset, model=model, batch_size=batch_size)
    return sanitize_output(str(result))


@mcp.tool()
def auto_tag_dataset_tool(dataset_id: str, method: str = "resnet") -> str:
    """Auto-tag dataset images with ResNet ImageNet classes"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_methods = {"resnet", "clip", "efficientnet"}
    if method not in valid_methods:
        return sanitize_output(f"Invalid method: {method}. Must be one of: {valid_methods}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = auto_tag_dataset(dataset, method=method)
    return sanitize_output(str(result))


@mcp.tool()
def enrich_metadata_tool(dataset_id: str, caption_model: str = "blip", tag_method: str = "resnet") -> str:
    """Enrich dataset metadata with captions and tags"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_caption_models = {"blip", "resnet", "clip"}
    valid_tag_methods = {"resnet", "clip", "efficientnet"}
    if caption_model not in valid_caption_models:
        return sanitize_output(f"Invalid caption_model: {caption_model}. Must be one of: {valid_caption_models}")
    if tag_method not in valid_tag_methods:
        return sanitize_output(f"Invalid tag_method: {tag_method}. Must be one of: {valid_tag_methods}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = enrich_metadata(dataset, caption_model=caption_model, tag_method=tag_method)
    return sanitize_output(str(result))


@mcp.tool()
def uncertainty_sampling_tool(dataset_id: str, method: str = "entropy", n: int = 10, model: str = "resnet50") -> str:
    """Select images for labeling using uncertainty sampling"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_methods = {"entropy", "margin", "least_confidence"}
    if method not in valid_methods:
        return sanitize_output(f"Invalid method: {method}. Must be one of: {valid_methods}")
    n = validate_positive_int(n, max_value=10000)
    valid_models = {"resnet50", "resnet101", "clip-vit-base", "clip-vit-large", "dinov2"}
    if model not in valid_models:
        return sanitize_output(f"Invalid model: {model}. Must be one of: {valid_models}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = uncertainty_sampling(dataset, method=method, n=n, model=model)
    return sanitize_output(str(result))


@mcp.tool()
def diversity_sampling_tool(dataset_id: str, n: int = 10, model: str = "resnet50") -> str:
    """Select diverse images for labeling using embedding distances"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    n = validate_positive_int(n, max_value=10000)
    valid_models = {"resnet50", "resnet101", "clip-vit-base", "clip-vit-large", "dinov2"}
    if model not in valid_models:
        return sanitize_output(f"Invalid model: {model}. Must be one of: {valid_models}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = diversity_sampling(dataset, n=n, model=model)
    return sanitize_output(str(result))


@mcp.tool()
def query_strategy_tool(dataset_id: str, strategy: str = "combined", n: int = 10, model: str = "resnet50", alpha: float = 0.5) -> str:
    """Combined active learning query strategy"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    valid_strategies = {"uncertainty", "diversity", "combined"}
    if strategy not in valid_strategies:
        return sanitize_output(f"Invalid strategy: {strategy}. Must be one of: {valid_strategies}")
    n = validate_positive_int(n, max_value=10000)
    valid_models = {"resnet50", "resnet101", "clip-vit-base", "clip-vit-large", "dinov2"}
    if model not in valid_models:
        return sanitize_output(f"Invalid model: {model}. Must be one of: {valid_models}")
    if alpha < 0 or alpha > 1:
        return sanitize_output("alpha must be between 0 and 1")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = query_strategy(dataset, strategy=strategy, n=n, model=model, alpha=alpha)
    return sanitize_output(str(result))


@mcp.tool()
def export_webdataset_tool(dataset_id: str, output_dir: str, image_size: list[int] = [640, 640], shard_size: int = 1000) -> str:
    """Export dataset to WebDataset tar format"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_output = validate_output_path(output_dir, allow_create=True)
    if len(image_size) != 2:
        return sanitize_output("image_size must be [width, height]")
    image_size = tuple(image_size)
    shard_size = validate_positive_int(shard_size, max_value=100000)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = export_webdataset(dataset, output_dir=str(validated_output), image_size=image_size, shard_size=shard_size)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def export_fiftyone_tool(dataset_id: str, output_dir: str, export_format: str = "coco") -> str:
    """Export dataset to FiftyOne format"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_output = validate_output_path(output_dir, allow_create=True)
    # Validate export format
    valid_formats = {"coco", "voc", "yolo", "kitti", "tfrecord", "webdataset", "labelme", "cvat"}
    if export_format.lower() not in valid_formats:
        return sanitize_output(f"Invalid export_format: {export_format}. Must be one of: {sorted(valid_formats)}")

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = export_fiftyone(dataset, output_dir=str(validated_output), export_format=export_format)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def export_cvat_xml_tool(dataset_id: str, output_path: str) -> str:
    """Export dataset to CVAT XML format"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_output = validate_output_path(output_path, allow_create=True)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = export_cvat_xml(dataset, output_path=str(validated_output))
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def export_labelme_json_tool(dataset_id: str, output_dir: str) -> str:
    """Export dataset to LabelMe JSON format"""
    validated_dataset_id = validate_dataset_id(dataset_id)
    validated_output = validate_output_path(output_dir, allow_create=True)

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    result = export_labelme_json(dataset, output_dir=str(validated_output))
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def extract_frames_tool(video_path: str, output_dir: str | None = None, frame_interval: int = 30, max_frames: int | None = None, resize: list[int] | None = None) -> str:
    """Extract frames from a video file"""
    validated_video_path = str(validate_file_path(video_path))
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(validated_video_path), "frames")
    validated_output = str(validate_output_path(output_dir, allow_create=True))
    if resize:
        if len(resize) != 2:
            return sanitize_output("resize must be [width, height]")
        resize = tuple(resize)
    if frame_interval:
        frame_interval = validate_positive_int(frame_interval)
    if max_frames:
        max_frames = validate_positive_int(max_frames)
    result = extract_frames(validated_video_path, output_dir=validated_output, frame_interval=frame_interval, max_frames=max_frames, resize=resize)
    return sanitize_output(str(result))


@mcp.tool()
def video_to_dataset_tool(video_path: str, output_dir: str | None = None, frame_interval: int = 30, max_frames: int | None = None, resize: list[int] | None = None, name: str | None = None) -> str:
    """Convert video to CVDataset of frames"""
    validated_video_path = str(validate_file_path(video_path))
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(validated_video_path), "frames")
    validated_output = str(validate_output_path(output_dir, allow_create=True))
    if resize:
        if len(resize) != 2:
            return sanitize_output("resize must be [width, height]")
        resize = tuple(resize)
    if frame_interval:
        frame_interval = validate_positive_int(frame_interval)
    if max_frames:
        max_frames = validate_positive_int(max_frames)
    if name:
        name = validate_dataset_id(name)
    dataset = video_to_dataset(validated_video_path, output_dir=validated_output, frame_interval=frame_interval, max_frames=max_frames, resize=resize, name=name)
    dataset_id = register_dataset(dataset.to_info())
    with _store_lock:
        _dataset_store[dataset_id] = dataset
    return sanitize_output(dataset.to_info().model_dump_json(indent=2))


@mcp.tool()
def batch_video_to_datasets_tool(video_dir: str, output_base: str | None = None, frame_interval: int = 30, max_frames: int | None = None, resize: list[int] | None = None) -> str:
    """Convert all videos in a directory to datasets"""
    video_dir = str(validate_dataset_path(video_dir))
    if output_base:
        output_base = str(validate_output_path(output_base, allow_create=True))
    frame_interval = validate_positive_int(frame_interval)
    if max_frames:
        max_frames = validate_positive_int(max_frames)
    if resize:
        if len(resize) != 2:
            return sanitize_output("resize must be [width, height]")
        resize = tuple(resize)
    results = video_batch_to_datasets(video_dir, output_base=output_base, frame_interval=frame_interval, max_frames=max_frames, resize=resize)
    return sanitize_output(str(results))


@mcp.tool()
def temporal_split_tool(dataset_id: str, ratios: dict[str, float] | None = None) -> str:
    """Temporal split of video frames (train/val/test)"""
    dataset_id = validate_dataset_id(dataset_id)
    if ratios:
        ratios = validate_ratio_dict(ratios)
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{dataset_id}' not found.")
    result = temporal_split(dataset, ratios=ratios)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def list_templates_tool() -> str:
    """List pre-built pipeline templates"""
    templates = list_templates()
    return str(templates)


@mcp.tool()
def get_template_tool(template_id: str) -> str:
    """Get a pipeline template by ID"""
    tpl = get_template(template_id)
    if tpl is None:
        return f"Template '{template_id}' not found."
    return tpl.model_dump_json(indent=2)


@mcp.tool()
def execute_pipeline_tool(dataset_id: str, pipeline: dict, work_dir: str | None = None) -> str:
    """Execute a pipeline on a dataset"""
    # Validate dataset ID
    validated_dataset_id = validate_dataset_id(dataset_id)
    # Validate work_dir if provided
    if work_dir:
        work_dir = str(validate_output_path(work_dir, allow_create=True))

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    pipeline_obj = PipelineDefinition(**pipeline)
    result = execute_pipeline(dataset, pipeline_obj, work_dir=work_dir)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def execute_template_tool(dataset_id: str, template_id: str, work_dir: str | None = None) -> str:
    """Execute a pipeline template on a dataset"""
    # Validate dataset ID
    validated_dataset_id = validate_dataset_id(dataset_id)
    # Validate work_dir if provided
    if work_dir:
        work_dir = str(validate_output_path(work_dir, allow_create=True))

    dataset = _get_or_load_dataset(validated_dataset_id)
    if dataset is None:
        return sanitize_output(f"Dataset '{validated_dataset_id}' not found.")
    tpl = get_template(template_id)
    if tpl is None:
        return sanitize_output(f"Template '{template_id}' not found.")
    result = execute_pipeline(dataset, tpl, work_dir=work_dir)
    return sanitize_output(result.model_dump_json(indent=2))


@mcp.tool()
def register_pipeline_tool(pipeline: dict) -> str:
    """Register a pipeline definition for reuse"""
    pipeline_obj = PipelineDefinition(**pipeline)
    pid = register_pipeline(pipeline_obj)
    return f"Pipeline '{pid}' registered."


@mcp.tool()
def list_pipelines_tool() -> str:
    """List all registered pipelines"""
    pipelines = list_pipelines()
    if not pipelines:
        return "No pipelines registered."
    result = [p.model_dump() for p in pipelines]
    return str(result)


@mcp.tool()
def save_pipeline_tool(pipeline_id: str, path: str) -> str:
    """Save a pipeline definition to a JSON file"""
    # Validate output path
    validated_path = validate_output_path(path, allow_create=True)
    pipeline = get_pipeline(pipeline_id)
    if pipeline is None:
        return sanitize_output(f"Pipeline '{pipeline_id}' not found.")
    out = save_pipeline_to_json(pipeline, str(validated_path))
    return sanitize_output(f"Saved pipeline to {out}")


@mcp.tool()
def load_pipeline_tool(path: str) -> str:
    """Load a pipeline definition from a JSON file"""
    # Validate input path
    validated_path = validate_file_path(path, allowed_extensions={'.json'})
    pipeline = load_pipeline_from_json(str(validated_path))
    register_pipeline(pipeline)
    return sanitize_output(pipeline.model_dump_json(indent=2))


# =============================================================================
# RESOURCES
# =============================================================================


@mcp.resource("dataset://{dataset_id}/info")
def dataset_info_resource(dataset_id: str) -> str:
    """Get dataset info as a resource"""
    info = get_dataset_info(dataset_id)
    if info is None:
        return f"Dataset '{dataset_id}' not found."
    return info.model_dump_json(indent=2)


@mcp.resource("dataset://{dataset_id}/images")
def dataset_images_resource(dataset_id: str) -> str:
    """List all images in a dataset as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    images = [{"image_id": img.image_id, "path": img.path, "width": img.width, "height": img.height} for img in dataset.images]
    return json.dumps(images, indent=2)


@mcp.resource("dataset://{dataset_id}/annotations")
def dataset_annotations_resource(dataset_id: str) -> str:
    """List all annotations in a dataset as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    annotations = []
    for img in dataset.images:
        for ann in img.annotations:
            annotations.append({
                "image_id": img.image_id,
                "label": ann.label,
                "bbox": ann.bbox,
                "mask": ann.segmentation is not None
            })
    return json.dumps(annotations, indent=2)


@mcp.resource("dataset://{dataset_id}/quality")
def dataset_quality_resource(dataset_id: str) -> str:
    """Get quality analysis as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_quality(dataset)
    return report.model_dump_json(indent=2)


@mcp.resource("dataset://{dataset_id}/duplicates")
def dataset_duplicates_resource(dataset_id: str) -> str:
    """Get duplicate detection results as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = find_duplicates(dataset)
    return report.model_dump_json(indent=2)


@mcp.resource("dataset://{dataset_id}/distribution")
def dataset_distribution_resource(dataset_id: str) -> str:
    """Get distribution analysis as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    return report.model_dump_json(indent=2)


@mcp.resource("dataset://{dataset_id}/health")
def dataset_health_resource(dataset_id: str) -> str:
    """Get dataset health score as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    score = dataset_health_score(dataset)
    return score.model_dump_json(indent=2)


@mcp.resource("dataset://{dataset_id}/statistics")
def dataset_statistics_resource(dataset_id: str) -> str:
    """Dataset statistics (sizes, class counts, annotation density) as a resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    return compute_statistics(dataset).model_dump_json(indent=2)


@mcp.resource("dataset://list")
def dataset_list_resource() -> str:
    """List all loaded datasets as a resource"""
    datasets = _list_datasets()
    if not datasets:
        return "No datasets loaded."
    result = [d.model_dump(mode="json") for d in datasets]
    return json.dumps(result, indent=2)


# =============================================================================
# ADVANCED RESOURCES
# =============================================================================


@mcp.resource("dataset://{dataset_id}/quality/scores")
def dataset_quality_scores_resource(dataset_id: str) -> str:
    """Per-image quality scores as CSV-like resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_quality(dataset)
    lines = ["image_id,blur,exposure,noise,contrast,resolution,color,overall,issues"]
    for q in report.per_image:
        issues = "|".join(q.issues) if q.issues else "none"
        lines.append(f"{q.image_id},{q.blur_score:.3f},{q.exposure_score:.3f},{q.noise_score:.3f},{q.contrast_score:.3f},{q.resolution_score:.3f},{q.color_score:.3f},{q.overall_score:.3f},{issues}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/quality/flagged")
def dataset_flagged_images_resource(dataset_id: str) -> str:
    """Images below quality threshold"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_quality(dataset)
    flagged = [q for q in report.per_image if q.overall_score is not None and q.overall_score < 50.0]
    lines = ["image_id,overall_score,issues"]
    for q in flagged:
        issues = "|".join(q.issues) if q.issues else "none"
        lines.append(f"{q.image_id},{q.overall_score:.1f},{issues}")
    return "Flagged images (threshold < 50.0):\n" + "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/class_stats")
def dataset_class_stats_resource(dataset_id: str) -> str:
    """Per-class statistics as resource"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    lines = ["class_name,image_count,annotation_count,percentage,avg_area"]
    for c in report.class_distribution:
        lines.append(f"{c.class_name},{c.count},{c.annotation_count},{c.percentage:.2f},{c.avg_area or 0:.2f}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/class_balance")
def dataset_class_balance_resource(dataset_id: str) -> str:
    """Class balance metrics"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    lines = ["metric,value"]
    if report.imbalance_ratio:
        lines.append(f"imbalance_ratio,{report.imbalance_ratio:.2f}")
    if report.gini_coefficient:
        lines.append(f"gini_coefficient,{report.gini_coefficient:.4f}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/spatial_heatmap")
def dataset_spatial_heatmap_resource(dataset_id: str) -> str:
    """Spatial distribution heatmap as CSV"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    if report.spatial_heatmap:
        grid = report.spatial_heatmap.grid
        lines = [f"Spatial Heatmap ({report.spatial_heatmap.grid_size[0]}x{report.spatial_heatmap.grid_size[1]})"]
        for row in grid:
            lines.append(",".join(f"{v:.4f}" for v in row))
        return "\n".join(lines)
    return "No spatial heatmap data"


@mcp.resource("dataset://{dataset_id}/co_occurrence")
def dataset_co_occurrence_resource(dataset_id: str) -> str:
    """Class co-occurrence matrix"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    if report.label_co_occurrence:
        lines = ["class_a,class_b,count,jaccard_index"]
        for c in report.label_co_occurrence:
            lines.append(f"{c.class_a},{c.class_b},{c.count},{c.jaccard_index:.4f}")
        return "\n".join(lines)
    return "No co-occurrence data"


@mcp.resource("dataset://{dataset_id}/complexity")
def dataset_complexity_resource(dataset_id: str) -> str:
    """Per-image complexity scores"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = analyze_distribution(dataset)
    if report.complexity_scores:
        lines = ["image_id,object_count,annotation_density,size_variance,complexity_score"]
        for c in report.complexity_scores:
            lines.append(f"{c.image_id},{c.object_count},{c.annotation_density:.4f},{c.size_variance:.4f},{c.complexity_score:.4f}")
        return "\n".join(lines)
    return "No complexity data"


@mcp.resource("dataset://{dataset_id}/duplicate_groups")
def dataset_duplicate_groups_resource(dataset_id: str) -> str:
    """Detailed duplicate groups"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = find_duplicates(dataset)
    lines = ["group_id,method,image_ids,recommended_keeper"]
    for g in report.duplicate_groups:
        lines.append(f"{g.group_id},{g.method},{','.join(g.image_ids)},{g.recommended_keeper or 'none'}")
    lines.append(f"# estimated_space_savings: {report.estimated_space_savings}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/integrity")
def dataset_integrity_resource(dataset_id: str) -> str:
    """Integrity check results"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    report = check_integrity(dataset)
    lines = ["image_id,issue_type,description,severity"]
    for issue in report.issues:
        lines.append(f"{issue.image_id},{issue.issue_type},{issue.description},{issue.severity}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/split/{split_name}")
def dataset_split_resource(dataset_id: str, split_name: str) -> str:
    """Images in a specific split"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    return f"Split '{split_name}' info for dataset '{dataset_id}'"


@mcp.resource("dataset://{dataset_id}/annotations/by_class")
def dataset_annotations_by_class_resource(dataset_id: str) -> str:
    """Annotations grouped by class"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    counts = {}
    for img in dataset.images:
        for ann in img.annotations:
            counts[ann.label] = counts.get(ann.label, 0) + 1
    lines = ["class,count"]
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"{cls},{count}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/bbox_stats")
def dataset_bbox_stats_resource(dataset_id: str) -> str:
    """Bounding box statistics"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    areas = []
    aspect_ratios = []
    for img in dataset.images:
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                w, h = abs(x2 - x1), abs(y2 - y1)
                areas.append(w * h)
                if h > 0:
                    aspect_ratios.append(w / h)
    import numpy as np
    lines = ["metric,value"]
    if areas:
        lines.append(f"area_mean,{np.mean(areas):.2f}")
        lines.append(f"area_median,{np.median(areas):.2f}")
        lines.append(f"area_std,{np.std(areas):.2f}")
        lines.append(f"area_min,{np.min(areas):.2f}")
        lines.append(f"area_max,{np.max(areas):.2f}")
    if aspect_ratios:
        lines.append(f"ar_mean,{np.mean(aspect_ratios):.4f}")
        lines.append(f"ar_median,{np.median(aspect_ratios):.4f}")
        lines.append(f"ar_std,{np.std(aspect_ratios):.4f}")
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/split_info")
def dataset_split_info_resource(dataset_id: str) -> str:
    """Dataset split information if available"""
    dataset = _get_or_load_dataset(dataset_id)
    if dataset is None:
        return f"Dataset '{dataset_id}' not found."
    lines = ["split_name,image_count,annotation_count"]
    return "\n".join(lines)


@mcp.resource("dataset://{dataset_id}/export_config")
def dataset_export_config_resource(dataset_id: str) -> str:
    """Export configuration for various formats"""
    configs = {
        "pytorch": json.dumps({"batch_size": 16, "num_workers": 4, "shuffle": True, "pin_memory": True}, indent=2),
        "tensorflow": json.dumps({"batch_size": 32, "shuffle_buffer": 1000, "prefetch": "AUTOTUNE"}, indent=2),
        "ultralytics": json.dumps({"imgsz": 640, "batch": 16, "epochs": 100, "patience": 50}, indent=2),
        "tensorflow_lite": json.dumps({"input_size": 320, "quantization": "int8"}, indent=2),
        "onnx": json.dumps({"opset": 11, "dynamic_axes": True}, indent=2),
    }
    return json.dumps(configs, indent=2)


# =============================================================================
# PROMPTS
# =============================================================================


@mcp.prompt()
def prepare_yolo_training_prompt(dataset_id: str, val_split: float = 0.2, test_split: float = 0.1) -> str:
    """Prepare dataset for YOLO training"""
    return f"""Prepare dataset '{dataset_id}' for YOLO training.

Steps:
1. Check dataset integrity: use check_integrity_tool with dataset_id="{dataset_id}"
2. Convert to YOLO format: use convert_format_tool with dataset_id="{dataset_id}", target_format="yolo"
3. Split dataset: use split_dataset_tool with dataset_id="{dataset_id}", strategy="stratified", ratios={{"train": {1-val_split-test_split}, "val": {val_split}, "test": {test_split}}}
4. Export for Ultralytics: use export_dataset_tool with dataset_id="<yolo_dataset_id>", format="ultralytics"

The output will be a ready-to-train YOLO dataset with dataset.yaml configuration."""


@mcp.prompt()
def export_pipeline_prompt(dataset_id: str, target_format: str = "pytorch") -> str:
    """Export dataset to various training formats"""
    formats = {
        "pytorch": "PyTorch Dataset/DataLoader",
        "tensorflow": "TFRecord/tf.data.Dataset",
        "ultralytics": "Ultralytics YOLO format",
        "hdf5": "HDF5 for fast I/O"
    }
    format_desc = formats.get(target_format, "Custom format")

    return f"""Export dataset '{dataset_id}' to {format_desc}.

Steps:
1. Validate dataset: use check_integrity_tool with dataset_id="{dataset_id}"
2. Convert format if needed: use convert_format_tool with dataset_id="{dataset_id}", target_format="{target_format}"
3. Export: use export_dataset_tool with dataset_id="{dataset_id}", format="{target_format}"

Target format: {target_format} ({format_desc})"""


@mcp.prompt()
def active_learning_prompt(dataset_id: str, budget: int = 100, strategy: str = "combined") -> str:
    """Active learning sample selection"""
    strategies = {
        "entropy": "Entropy-based uncertainty sampling",
        "margin": "Margin-based uncertainty sampling",
        "least_confidence": "Least confidence uncertainty sampling",
        "diversity": "Diversity-based sampling (k-means++ / core-set)",
        "combined": "Combined uncertainty + diversity (alpha=0.5)"
    }
    strategy_desc = strategies.get(strategy, "Combined strategy")

    return f"""Active learning sample selection for dataset '{dataset_id}'.

Budget: {budget} samples to label
Strategy: {strategy_desc}

Steps:
1. Compute embeddings: use compute_embeddings_tool with dataset_id="{dataset_id}", model="resnet50"
2. Select samples: use query_strategy_tool with dataset_id="{dataset_id}", strategy="{strategy}", n={budget}, model="resnet50"
3. Review selected samples: use preview_images_tool with dataset_id="{dataset_id}" and the sampled IDs
4. Label the selected samples (manual or via annotation tool)
5. Retrain model and repeat

This implements {strategy_desc} for optimal sample selection."""


@mcp.prompt()
def dataset_versioning_prompt(dataset_id: str, new_dataset_id: str) -> str:
    """Dataset versioning and diff"""
    return f"""Dataset versioning: Compare '{dataset_id}' (old) vs '{new_dataset_id}' (new).

Steps:
1. Load both datasets: use dataset_info with dataset_id="{dataset_id}" and dataset_id="{new_dataset_id}"
2. Check integrity of both: use check_integrity_tool on both
3. Compare distributions: use analyze_distribution_tool on both
3. Check quality: use analyze_quality_tool on both
4. Find differences: use dataset_diff_tool with dataset_id_a="{dataset_id}", dataset_id_b="{new_dataset_id}"

Report on:
- Added/removed images
- Class distribution shifts
- Quality improvements/regressions
- Annotation changes
- Health score changes
- Recommendation: promote new version or investigate regressions"""


@mcp.prompt()
def annotation_quality_improvement_prompt(dataset_id: str) -> str:
    """Improve annotation quality"""
    return f"""Improve annotation quality for dataset '{dataset_id}'.

Steps:
1. Check integrity: use check_integrity_tool with dataset_id="{dataset_id}"
2. Find issues: Look for missing labels, out-of-bounds boxes, degenerate boxes
3. Analyze quality: use analyze_quality_tool with dataset_id="{dataset_id}"
4. Find duplicates: use find_duplicates_tool with dataset_id="{dataset_id}", methods=["exact", "perceptual", "ssim"]
5. Review class balance: use analyze_distribution_tool with dataset_id="{dataset_id}"
5. Fix issues:
   - Remove/fix corrupt images
   - Fix out-of-bounds boxes
   - Remove degenerate boxes (zero area)
   - Add missing labels
   - Remove/merge duplicate annotations
6. Re-run full audit: use dataset_health_score_tool with dataset_id="{dataset_id}"

Output: Fixed dataset with improved health score."""


@mcp.prompt()
def custom_augmentation_policy_prompt(dataset_id: str, target_task: str = "detection") -> str:
    """Design custom augmentation policy"""
    tasks = {
        "detection": "Object detection (YOLO, Faster R-CNN, DETR)",
        "classification": "Image classification (ResNet, EfficientNet, ViT)",
        "segmentation": "Instance/semantic segmentation (Mask R-CNN, DeepLab)",
        "pose": "Keypoint detection (HRNet, YOLO-Pose)"
    }
    task_desc = tasks.get(target_task, "General computer vision")

    return f"""Design custom augmentation policy for dataset '{dataset_id}' for {task_desc}.

Steps:
1. Analyze distribution: use analyze_distribution_tool with dataset_id="{dataset_id}"
2. Check quality: use analyze_quality_tool with dataset_id="{dataset_id}"
3. Identify gaps: Look for underrepresented classes, aspects, lighting, backgrounds
4. Design policy based on task:
   - Detection: Mosaic, MixUp, Copy-Paste, RandomCrop, HSV jitter
   - Classification: AutoAugment, RandAugment, Cutout, MixUp
   - Segmentation: Copy-Paste, ElasticTransform, GridDistortion
4. Configure pipeline: use augment_dataset_tool with dataset_id="{dataset_id}", pipeline="custom"
5. Preview: use preview_augmentation_tool with dataset_id="{dataset_id}", n_samples=8
5. Evaluate: Train quick baseline, compare mAP/accuracy

Task: {task_desc}
Output: Augmentation config for training pipeline."""


@mcp.prompt()
def prepare_coco_training_prompt(dataset_id: str, model_type: str = "faster_rcnn") -> str:
    """Prepare dataset for COCO-format training (Faster R-CNN, DETR, etc.)"""
    models = {
        "faster_rcnn": "Faster R-CNN (ResNet50-FPN)",
        "mask_rcnn": "Mask R-CNN (instance segmentation)",
        "detr": "DETR (Transformer-based detection)",
        "deformable_detr": "Deformable DETR",
        "retinanet": "RetinaNet"
    }
    model_desc = models.get(model_type, "COCO-format detector")

    return f"""Prepare dataset '{dataset_id}' for {model_desc} training.

Steps:
1. Validate: use check_integrity_tool with dataset_id="{dataset_id}"
2. Convert to COCO: use convert_format_tool with dataset_id="{dataset_id}", target_format="coco"
3. Split: use split_dataset_tool with dataset_id="{dataset_id}", strategy="stratified", ratios={{"train": 0.8, "val": 0.1, "test": 0.1}}
4. Export COCO: use export_dataset_tool with dataset_id="{dataset_id}", format="coco"
5. Configure training:
   - Model: {model_desc}
   - COCO train/val JSON ready
   - Image directory structure ready

Target model: {model_desc}"""


@mcp.prompt()
def prepare_segmentation_training_prompt(dataset_id: str, model_type: str = "mask_rcnn") -> str:
    """Prepare dataset for segmentation training"""
    models = {
        "mask_rcnn": "Mask R-CNN (instance segmentation)",
        "deepLab": "DeepLabV3+ (semantic segmentation)",
        "unet": "U-Net (semantic segmentation)",
        "panoptic_fpn": "Panoptic FPN (panoptic segmentation)"
    }
    model_desc = models.get(model_type, "Segmentation model")

    return f"""Prepare dataset '{dataset_id}' for {model_desc} training.

Steps:
1. Validate masks: use check_integrity_tool with dataset_id="{dataset_id}"
2. Check mask quality: Verify masks are valid polygons/masks
3. Convert format: use convert_format_tool with dataset_id="{dataset_id}", target_format="coco" (for Mask R-CNN)
4. Split: use split_dataset_tool with dataset_id="{dataset_id}", strategy="stratified"
5. Export: use export_dataset_tool with dataset_id="{dataset_id}", format="pytorch"

For {model_desc}:
- Instance seg: COCO format with segmentation polygons
- Semantic seg: PNG masks or COCO panoptic
- Ensure mask annotations are valid polygons/RLE"""


@mcp.prompt()
def multi_dataset_merge_prompt(dataset_ids: str, strategy: str = "union") -> str:
    """Merge multiple datasets"""
    ids = [id.strip() for id in dataset_ids.split(",")]
    strategies = {
        "union": "Union - all images from all datasets",
        "intersection": "Intersection - only images in all datasets",
        "balanced": "Balanced - equal samples per class per dataset"
    }
    strategy_desc = strategies.get(strategy, "Union")

    return f"""Merge datasets: {', '.join(ids)} using {strategy_desc} strategy.

Steps:
1. Load all datasets: use load_dataset_tool for each dataset_id in {ids}
2. Check compatibility: Compare classes, formats, annotation styles
3. Merge: use merge_datasets_tool with dataset_ids={ids}, strategy="{strategy}", deduplicate=True, unify_labels=True
4. Validate: use check_integrity_tool on merged dataset
5. Analyze: use analyze_distribution_tool on merged dataset
6. Export: use export_dataset_tool with desired format

Strategy: {strategy_desc}
Input datasets: {len(ids)} datasets
Output: Merged dataset with unified labels"""


@mcp.prompt()
def dataset_migration_prompt(source_format: str, target_format: str, dataset_path: str) -> str:
    """Migrate dataset between formats"""
    return f"""Migrate dataset from {source_format} to {target_format}.

Source: {source_format} at {dataset_path}
Target: {target_format}

Steps:
1. Load source: use load_dataset_tool with path="{dataset_path}", format="{source_format}"
2. Validate: use check_integrity_tool on loaded dataset
3. Convert: use convert_format_tool with dataset_id="<loaded_id>", target_format="{target_format}"
4. Validate: use check_integrity_tool on converted dataset
5. Export: use export_dataset_tool with format="{target_format}"

Supported conversions:
- COCO ↔ VOC ↔ YOLO ↔ KITTI ↔ CSV
- Includes: images, annotations, classes, splits
- Preserves: bboxes, polygons, masks, keypoints, metadata"""


@mcp.prompt()
def dataset_health_monitoring_prompt(dataset_id: str, threshold: float = 70.0) -> str:
    """Monitor dataset health over time"""
    return f"""Set up health monitoring for dataset '{dataset_id}'.

Threshold: {threshold}/100

Monitoring workflow:
1. Baseline: use dataset_health_score_tool with dataset_id="{dataset_id}"
2. Set up periodic checks (weekly/monthly):
   - Re-run health score
   - Compare to baseline
   - Alert if score drops below {threshold}
3. Track metrics over time:
   - Overall health score
   - Class balance (entropy, Gini)
   - Quality metrics (blur, exposure, noise)
   - Integrity issues count
   - Duplicate percentage
3. Alert conditions:
   - Health score < {threshold}
   - New integrity issues > 5
   - Duplicate rate > 5%
   - Class entropy drop > 10%
4. Automated remediation suggestions:
   - If quality drops: run analyze_quality_tool
   - If duplicates spike: run find_duplicates_tool
   - If class imbalance: run analyze_distribution_tool

Output: Health monitoring dashboard configuration"""


# =============================================================================
# SECURITY WIRING
# =============================================================================
# The security primitives (rate limiter, resource limiter, audit log) are wired
# into every registered tool at one choke point instead of decorating each of
# the ~55 tools by hand. Each tool call is rate-limited per (tool, dataset),
# bounded by a max-concurrency slot, and audit-logged.


def _secure_tool_wrapper(tool_name: str, fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        dataset_id = kwargs.get("dataset_id") or kwargs.get("dataset_id_a") or "global"
        rl_key = f"{tool_name}:{dataset_id}"

        if not TOOL_RATE_LIMITER.is_allowed(rl_key):
            SECURITY_AUDITOR.log_security_violation("rate_limit", tool_name, dataset_id)
            return sanitize_output(
                f"Rate limit exceeded for '{tool_name}'. Please slow down and retry."
            )

        if not RESOURCE_LIMITER.acquire(tool_name):
            SECURITY_AUDITOR.log_security_violation("max_concurrency", tool_name, dataset_id)
            return sanitize_output(
                "Server busy: maximum concurrent operations reached. Retry shortly."
            )

        try:
            result = fn(*args, **kwargs)
            SECURITY_AUDITOR.log_tool_call(tool_name, dataset_id, success=True)
            return result
        except SecurityError as e:
            SECURITY_AUDITOR.log_security_violation("security_error", tool_name, dataset_id,
                                                    {"error": str(e)})
            return sanitize_output(f"Security error: {e}")
        except Exception as e:  # noqa: BLE001 - tools must return a string, never crash the server
            SECURITY_AUDITOR.log_tool_call(tool_name, dataset_id, success=False, error=str(e))
            return sanitize_output(f"Error in {tool_name}: {e}")
        finally:
            RESOURCE_LIMITER.release()

    return wrapper


def _apply_security_wrappers() -> None:
    """Wrap every registered tool's handler with the security choke point."""
    try:
        tools = mcp._tool_manager._tools  # type: ignore[attr-defined]
    except AttributeError:
        return
    for name, tool in tools.items():
        if getattr(tool, "_security_wrapped", False):
            continue
        tool.fn = _secure_tool_wrapper(name, tool.fn)
        try:
            tool._security_wrapped = True  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


_apply_security_wrappers()


@mcp.prompt()
def dataset_curation_prompt(dataset_id: str) -> str:
    """Audit then curate a dataset into a clean, training-ready version"""
    return f"""Curate dataset '{dataset_id}' into a clean, training-ready dataset.

Steps:
1. Audit: use check_integrity_tool, analyze_quality_tool, and find_duplicates_tool with dataset_id="{dataset_id}"
2. Review class balance: use analyze_distribution_tool with dataset_id="{dataset_id}"
3. Clean: use clean_dataset_tool with dataset_id="{dataset_id}", drop_corrupt=true, clip_out_of_bounds=true, drop_degenerate=true, drop_duplicates=true, output_dir="<path>"
   -> returns a NEW cleaned dataset (new_dataset_id)
4. (Optional) Merge/rename classes: use remap_classes_tool on the cleaned dataset with a mapping like {{"car":"vehicle","truck":"vehicle"}}
5. (Optional) Keep only useful classes/sizes: use filter_dataset_tool with classes=[...] or min_area=...
6. Re-audit the curated dataset: use dataset_health_score_tool on the new_dataset_id
7. Split and export: use split_dataset_tool then export_dataset_tool (or export_huggingface_tool to publish)

Output: a curated dataset with a higher health score, ready for training or sharing."""


@mcp.prompt()
def publish_dataset_prompt(dataset_id: str, repo_id: str = "your-username/your-dataset") -> str:
    """Prepare and publish a dataset to the Hugging Face Hub"""
    return f"""Publish dataset '{dataset_id}' to the Hugging Face Hub as '{repo_id}'.

Steps:
1. Clean it first: use clean_dataset_tool with dataset_id="{dataset_id}", drop_duplicates=true, output_dir="<path>"
2. Validate: use check_integrity_tool and dataset_health_score_tool on the cleaned dataset
3. Ensure HF_TOKEN is set in the server environment (never pass tokens as arguments)
4. Publish: use export_huggingface_tool with dataset_id="<cleaned_id>", push_to_hub=true, repo_id="{repo_id}", private=true

Note: requires the [huggingface] extra installed on the server."""


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Liveness/readiness probe for load balancers and container orchestrators."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "service": "pixlint",
        "version": "1.0.2",
        "tools": len(mcp._tool_manager._tools),
        "datasets_loaded": len(_dataset_store),
    })


def main() -> None:
    """Run the MCP server.

    Transport is selected via the ``MCP_TRANSPORT`` environment variable:
      - ``stdio`` (default): local use, e.g. Claude Desktop / Cursor.
      - ``streamable-http`` / ``sse``: network transport for internet hosting.

    For network transports, set ``MCP_HOST`` / ``MCP_PORT`` and, strongly
    recommended when public, ``CV_MCP_AUTH_TOKEN`` to require a bearer token.
    """
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()

    if transport in ("http", "streamable-http", "sse"):
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.settings.host = host
        mcp.settings.port = port
        _install_http_auth()
        run_transport = "sse" if transport == "sse" else "streamable-http"
        mcp.run(transport=run_transport)
    else:
        mcp.run()


def _install_http_auth() -> None:
    """Require a bearer token on the HTTP app when CV_MCP_AUTH_TOKEN is set."""
    token = os.environ.get("CV_MCP_AUTH_TOKEN")
    if not token:
        import logging
        logging.getLogger(__name__).warning(
            "CV_MCP_AUTH_TOKEN is not set — the HTTP server is UNAUTHENTICATED. "
            "Set CV_MCP_AUTH_TOKEN before exposing this server to a network."
        )
        return

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path in ("/health", "/healthz"):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            expected = f"Bearer {token}"
            # Constant-time comparison to avoid timing side channels.
            import hmac
            if not hmac.compare_digest(auth, expected):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    # FastMCP exposes the underlying Starlette app builders; attach middleware.
    for builder in ("streamable_http_app", "sse_app"):
        app_factory = getattr(mcp, builder, None)
        if app_factory is None:
            continue
        original = app_factory

        def make_wrapped(orig):
            def wrapped(*a, **kw):
                app = orig(*a, **kw)
                app.add_middleware(BearerAuthMiddleware)
                return app
            return wrapped

        setattr(mcp, builder, make_wrapped(original))


if __name__ == "__main__":
    main()
