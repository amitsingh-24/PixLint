from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DatasetFormat(str, Enum):
    AUTO = "auto"
    FOLDER = "folder"
    COCO = "coco"
    VOC = "voc"
    YOLO = "yolo"
    KITTI = "kitti"
    IMAGENET = "imagenet"
    CSV = "csv"


class Annotation(BaseModel):
    label: str
    bbox: tuple[float, float, float, float] | None = None
    segmentation: list[list[float]] | None = None
    keypoints: list[float] | None = None
    area: float | None = None
    iscrowd: bool = False
    confidence: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ImageRecord(BaseModel):
    image_id: str
    path: str
    width: int | None = None
    height: int | None = None
    annotations: list[Annotation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CVDatasetInfo(BaseModel):
    dataset_id: str
    name: str
    path: str
    format: DatasetFormat
    num_images: int
    num_annotations: int
    class_names: list[str]
    image_size: tuple[int, int] | None = None


class DuplicateGroup(BaseModel):
    group_id: int
    method: str
    image_ids: list[str]
    scores: list[float]
    recommended_keeper: str | None = None


class DuplicateReport(BaseModel):
    dataset_id: str
    total_duplicate_groups: int
    total_duplicate_images: int
    duplicate_groups: list[DuplicateGroup]
    estimated_space_savings: str
    method_summary: dict[str, int]


class QualityMetrics(BaseModel):
    image_id: str
    blur_score: float | None = None
    blur_label: str | None = None
    exposure_score: float | None = None
    exposure_label: str | None = None
    noise_score: float | None = None
    contrast_score: float | None = None
    resolution_score: float | None = None
    color_score: float | None = None
    overall_score: float | None = None
    issues: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    dataset_id: str
    per_image: list[QualityMetrics]
    average_overall: float
    flagged_images: int
    summary: dict[str, Any]


class IntegrityIssue(BaseModel):
    image_id: str
    issue_type: str
    description: str
    severity: str = "warning"


class IntegrityReport(BaseModel):
    dataset_id: str
    total_issues: int
    issues: list[IntegrityIssue]
    summary: dict[str, int]


class DistributionMetric(BaseModel):
    class_name: str
    count: int
    percentage: float
    annotation_count: int
    avg_area: float | None = None


class SpatialHeatmap(BaseModel):
    grid: list[list[float]]
    grid_size: tuple[int, int]


class LabelCoOccurrence(BaseModel):
    class_a: str
    class_b: str
    count: int
    jaccard_index: float


class ComplexityMetric(BaseModel):
    image_id: str
    object_count: int
    annotation_density: float
    size_variance: float
    complexity_score: float


class DistributionReport(BaseModel):
    dataset_id: str
    class_distribution: list[DistributionMetric]
    total_images: int
    total_annotations: int
    imbalance_ratio: float | None = None
    gini_coefficient: float | None = None
    spatial_heatmap: SpatialHeatmap | None = None
    label_co_occurrence: list[LabelCoOccurrence] | None = None
    complexity_scores: list[ComplexityMetric] | None = None


class SamplingResult(BaseModel):
    dataset_id: str
    sampled_ids: list[str]
    strategy: str
    n: int
    class_distribution: dict[str, int]


class SearchResult(BaseModel):
    image_id: str
    path: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    dataset_id: str
    num_images: int
    num_annotations: int
    num_classes: int
    class_names: list[str]
    total_pixels: int
    avg_image_size: tuple[float, float]
    min_image_size: tuple[int, int]
    max_image_size: tuple[int, int]
    avg_annotations_per_image: float
    class_counts: dict[str, int]


class EmbeddingResult(BaseModel):
    dataset_id: str
    model: str
    image_ids: list[str]
    embeddings: list[list[float]] | None = None
    embedding_dim: int | None = None
    num_images: int


class ClusterResult(BaseModel):
    dataset_id: str
    method: str
    n_clusters: int
    labels: list[int]
    image_ids: list[str]
    cluster_sizes: dict[str, int]
    silhouette_score: float | None = None


class OutlierReport(BaseModel):
    dataset_id: str
    methods_used: list[str]
    outlier_ids: list[str]
    outlier_scores: dict[str, float]
    total_outliers: int
    summary: dict[str, Any]


class HealthScoreBreakdown(BaseModel):
    class_balance: float
    image_quality: float
    integrity: float
    annotation_consistency: float
    diversity: float


class HealthScore(BaseModel):
    dataset_id: str
    overall: float
    breakdown: HealthScoreBreakdown
    recommendations: list[str]


class DatasetDiff(BaseModel):
    dataset_id_a: str
    dataset_id_b: str
    added_images: list[str]
    removed_images: list[str]
    common_images: int
    distribution_shift: dict[str, Any] | None = None
    quality_changes: dict[str, Any] | None = None


class SplitResult(BaseModel):
    dataset_id: str
    strategy: str
    ratios: dict[str, float]
    splits: dict[str, list[str]]
    class_distributions: dict[str, dict[str, int]] | None = None
    seed: int | None = None


class KFoldResult(BaseModel):
    dataset_id: str
    k: int
    strategy: str
    folds: list[SplitResult]
    seed: int | None = None


class LeakageReport(BaseModel):
    dataset_id: str
    total_leaks: int
    leakage_pairs: list[dict[str, str]]
    methods_used: list[str]


class ExportResult(BaseModel):
    dataset_id: str
    export_format: str
    output_dir: str
    num_images_exported: int
    num_annotations_exported: int


class CurationResult(BaseModel):
    """Result of a curation operation that produces a NEW dataset."""
    source_dataset_id: str
    new_dataset_id: str
    new_dataset_name: str
    operation: str
    num_images_in: int
    num_images_out: int
    num_annotations_in: int
    num_annotations_out: int
    removed_image_ids: list[str] = Field(default_factory=list)
    output_dir: str | None = None
    notes: list[str] = Field(default_factory=list)


class AutoLabelResult(BaseModel):
    """Result of running auto-labeling (model-predicted annotations)."""
    source_dataset_id: str
    new_dataset_id: str
    new_dataset_name: str
    model: str
    confidence_threshold: float
    num_images: int
    num_predicted_annotations: int
    predicted_classes: dict[str, int] = Field(default_factory=dict)
    output_dir: str | None = None
    notes: list[str] = Field(default_factory=list)


# ---- Differentiator: label-error detection ("data linter") ----
class LabelErrorItem(BaseModel):
    image_id: str
    given_label: str
    suggested_label: str
    confidence: float  # 0..1, how likely the given label is wrong
    neighbor_agreement: float  # fraction of neighbors sharing suggested_label
    reason: str


class LabelErrorReport(BaseModel):
    dataset_id: str
    method: str
    num_images_analyzed: int
    num_suspected_errors: int
    suspected_error_rate: float
    suspected_errors: list[LabelErrorItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---- Differentiator: natural-language / predicate query engine ----
class QueryResult(BaseModel):
    dataset_id: str
    num_matched: int
    total_images: int
    matched_image_ids: list[str] = Field(default_factory=list)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ---- Differentiator: dataset "Doctor" readiness report ----
class ReadinessRecommendation(BaseModel):
    severity: str  # critical | high | medium | low
    category: str
    finding: str
    action: str  # human/agent-readable next step (often names a tool)
    auto_fixable: bool = False


class ReadinessReport(BaseModel):
    dataset_id: str
    readiness_score: float  # 0..100
    readiness_level: str  # not_ready | needs_work | training_ready
    summary: str
    recommendations: list[ReadinessRecommendation] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    remediation_pipeline: dict[str, Any] | None = None  # runnable PipelineDefinition


# ---- Differentiator: weak-slice / bias discovery ----
class SliceInfo(BaseModel):
    slice_key: str  # e.g. "class=dog & brightness=dark"
    dimension: str
    count: int
    percentage: float
    avg_quality: float | None = None
    problems: list[str] = Field(default_factory=list)


class SliceReport(BaseModel):
    dataset_id: str
    num_images: int
    weak_slices: list[SliceInfo] = Field(default_factory=list)
    all_slices: list[SliceInfo] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class AugmentationConfig(BaseModel):
    pipeline_name: str
    multiplier: int = 1
    transforms: list[dict[str, Any]] = Field(default_factory=list)


class AugmentationResult(BaseModel):
    dataset_id: str
    pipeline: str
    multiplier: int
    output_dir: str
    total_generated: int
    class_distribution: dict[str, int] | None = None


class FormatConversionResult(BaseModel):
    dataset_id: str
    source_format: str
    target_format: str
    output_dir: str
    num_images_converted: int
    class_mapping: dict[str, str] | None = None


class ResizeResult(BaseModel):
    dataset_id: str
    target_size: tuple[int, int]
    strategy: str
    num_images_resized: int


class NormalizeResult(BaseModel):
    dataset_id: str
    mean: list[float]
    std: list[float]
    num_images_normalized: int


class LabelMapEntry(BaseModel):
    source_label: str
    source_dataset: str
    unified_label: str


class MergeResult(BaseModel):
    merged_dataset_id: str
    source_datasets: list[str]
    total_images: int
    total_annotations: int
    num_classes: int
    class_names: list[str]
    label_mapping: list[LabelMapEntry]
    duplicate_count: int
    duplicate_removed: int


class PipelineStep(BaseModel):
    step_id: str
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""


class PipelineDefinition(BaseModel):
    pipeline_id: str
    name: str
    version: str = "1.0"
    description: str = ""
    steps: list[PipelineStep]
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""
    input_dataset_id: str | None = None


class PipelineStepResult(BaseModel):
    step_id: str
    operation: str
    status: str
    duration_ms: float
    output: str = ""
    error: str | None = None


class PipelineResult(BaseModel):
    pipeline_id: str
    dataset_id: str
    status: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    results: list[PipelineStepResult]
    total_duration_ms: float
    output_dataset_id: str | None = None
