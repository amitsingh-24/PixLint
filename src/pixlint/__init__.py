"""PixLint - Computer Vision Dataset Management and Analysis."""

from pixlint.analysis.active_learning import (
    diversity_sampling,
    query_strategy,
    uncertainty_sampling,
)
from pixlint.analysis.captioning import auto_tag_dataset, enrich_metadata, generate_captions
from pixlint.analysis.diff import dataset_diff
from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.embeddings import cluster_dataset, compute_embeddings, semantic_search
from pixlint.analysis.health import dataset_health_score
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.outliers import detect_outliers
from pixlint.analysis.quality import analyze_quality
from pixlint.analysis.statistics import compute_statistics, sample_dataset
from pixlint.analysis.video import (
    extract_frames,
    temporal_split,
    video_batch_to_datasets,
    video_to_dataset,
)
from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.core.loader import CVDataset, detect_format, load_dataset
from pixlint.core.merge import merge_datasets
from pixlint.core.metadata import get_dataset_info, list_datasets, register_dataset
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
from pixlint.export.pytorch import export_pytorch
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.ultralytics import export_ultralytics
from pixlint.splitting.cross_validation import generate_kfold_splits
from pixlint.splitting.leakage import detect_leakage
from pixlint.splitting.splitter import split_dataset
from pixlint.transformation.format_converter import convert_format
from pixlint.transformation.normalize import compute_channel_stats, normalize_dataset
from pixlint.transformation.resize import resize_dataset
from pixlint.utils.schemas import PipelineDefinition
from pixlint.visualization.charts import (
    plot_distribution,
    plot_duplicate_groups,
    plot_quality_scores,
    plot_spatial_heatmap,
)
from pixlint.visualization.embeddings_viz import plot_embeddings
from pixlint.visualization.previews import preview_images, preview_single_image

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
