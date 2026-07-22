"""PixLint - Computer Vision Dataset Management and Analysis."""

from pixlint.core.loader import CVDataset, load_dataset, detect_format
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.quality import analyze_quality
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.statistics import compute_statistics, sample_dataset
from pixlint.analysis.embeddings import compute_embeddings, semantic_search, cluster_dataset
from pixlint.analysis.outliers import detect_outliers
from pixlint.analysis.health import dataset_health_score
from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.transformation.format_converter import convert_format
from pixlint.transformation.resize import resize_dataset
from pixlint.transformation.normalize import normalize_dataset, compute_channel_stats
from pixlint.splitting.splitter import split_dataset
from pixlint.splitting.cross_validation import generate_kfold_splits
from pixlint.splitting.leakage import detect_leakage
from pixlint.export.pytorch import export_pytorch
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.ultralytics import export_ultralytics
from pixlint.export.hdf5 import export_hdf5
from pixlint.export.extra_formats import (
    export_webdataset,
    export_fiftyone,
    export_cvat_xml,
    export_labelme_json,
)
from pixlint.visualization.previews import preview_images, preview_single_image
from pixlint.visualization.charts import (
    plot_distribution,
    plot_quality_scores,
    plot_spatial_heatmap,
    plot_duplicate_groups,
)
from pixlint.visualization.embeddings_viz import plot_embeddings
from pixlint.analysis.active_learning import uncertainty_sampling, diversity_sampling, query_strategy
from pixlint.analysis.video import extract_frames, video_to_dataset, video_batch_to_datasets, temporal_split
from pixlint.analysis.captioning import generate_captions, auto_tag_dataset, enrich_metadata
from pixlint.analysis.diff import dataset_diff
from pixlint.core.metadata import list_datasets, get_dataset_info, register_dataset
from pixlint.core.merge import merge_datasets
from pixlint.core.pipeline import (
    execute_pipeline,
    register_pipeline,
    list_pipelines,
    get_pipeline,
    save_pipeline_to_json,
    load_pipeline_from_json,
    get_template,
    list_templates,
)
from pixlint.utils.schemas import PipelineDefinition

__version__ = "1.0.2"

__all__ = [
    # Core
    "CVDataset",
    "load_dataset",
    "detect_format",
    "list_datasets",
    "get_dataset_info",
    "register_dataset",
    "merge_datasets",
    # Analysis
    "find_duplicates",
    "analyze_quality",
    "check_integrity",
    "analyze_distribution",
    "compute_statistics",
    "sample_dataset",
    "compute_embeddings",
    "semantic_search",
    "cluster_dataset",
    "detect_outliers",
    "dataset_health_score",
    # Augmentation
    "augment_dataset",
    "preview_augmentation",
    # Transformation
    "convert_format",
    "resize_dataset",
    "normalize_dataset",
    "compute_channel_stats",
    # Splitting
    "split_dataset",
    "generate_kfold_splits",
    "detect_leakage",
    # Export
    "export_pytorch",
    "export_tensorflow",
    "export_ultralytics",
    "export_hdf5",
    "export_webdataset",
    "export_fiftyone",
    "export_cvat_xml",
    "export_labelme_json",
    # Visualization
    "preview_images",
    "preview_single_image",
    "plot_distribution",
    "plot_quality_scores",
    "plot_spatial_heatmap",
    "plot_duplicate_groups",
    "plot_embeddings",
    # Active Learning
    "uncertainty_sampling",
    "diversity_sampling",
    "query_strategy",
    # Video
    "extract_frames",
    "video_to_dataset",
    "video_batch_to_datasets",
    "temporal_split",
    # Captioning
    "generate_captions",
    "auto_tag_dataset",
    "enrich_metadata",
    # Diff
    "dataset_diff",
    # Pipeline
    "execute_pipeline",
    "register_pipeline",
    "list_pipelines",
    "get_pipeline",
    "save_pipeline_to_json",
    "load_pipeline_from_json",
    "get_template",
    "list_templates",
    "PipelineDefinition",
]