from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.quality import analyze_quality
from pixlint.augmentation.pipeline import augment_dataset
from pixlint.core.loader import CVDataset, load_dataset
from pixlint.core.metadata import register_dataset
from pixlint.export.hdf5 import export_hdf5
from pixlint.export.pytorch import export_pytorch
from pixlint.export.tensorflow import export_tensorflow
from pixlint.export.ultralytics import export_ultralytics
from pixlint.export.extra_formats import (
    export_webdataset, export_fiftyone, export_cvat_xml, export_labelme_json,
)
from pixlint.splitting.splitter import split_dataset
from pixlint.splitting.cross_validation import generate_kfold_splits
from pixlint.splitting.leakage import detect_leakage
from pixlint.transformation.format_converter import convert_format
from pixlint.transformation.normalize import normalize_dataset, compute_channel_stats
from pixlint.transformation.resize import resize_dataset
from pixlint.utils.schemas import (
    PipelineDefinition,
    PipelineResult,
    PipelineStep,
    PipelineStepResult,
)


_PIPELINE_TEMPLATES: dict[str, PipelineDefinition] = {}
_PIPELINE_REGISTRY: dict[str, PipelineDefinition] = {}


def _get_operation_fn(name: str) -> Callable | None:
    ops: dict[str, Callable] = {
        "find_duplicates": find_duplicates,
        "analyze_quality": analyze_quality,
        "check_integrity": check_integrity,
        "augment_dataset": augment_dataset,
        "resize_dataset": resize_dataset,
        "normalize_dataset": normalize_dataset,
        "compute_channel_stats": compute_channel_stats,
        "convert_format": convert_format,
        "split_dataset": split_dataset,
        "generate_kfold": generate_kfold_splits,
        "detect_leakage": detect_leakage,
        "export_pytorch": export_pytorch,
        "export_tensorflow": export_tensorflow,
        "export_ultralytics": export_ultralytics,
        "export_hdf5": export_hdf5,
        "export_webdataset": export_webdataset,
        "export_fiftyone": export_fiftyone,
        "export_cvat_xml": export_cvat_xml,
        "export_labelme_json": export_labelme_json,
    }
    return ops.get(name)


def execute_pipeline(
    dataset: CVDataset,
    pipeline: PipelineDefinition,
    work_dir: str | None = None,
) -> PipelineResult:
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(dataset.path), f".pipeline_{pipeline.pipeline_id}")

    os.makedirs(work_dir, exist_ok=True)
    total_start = time.time()

    step_results: list[PipelineStepResult] = []
    completed = 0
    failed = 0
    status = "running"
    current_ds = dataset

    dag = _build_dag(pipeline.steps)
    execution_order = _topological_sort(dag, pipeline.steps)
    if execution_order is None:
        return PipelineResult(
            pipeline_id=pipeline.pipeline_id,
            dataset_id=dataset.dataset_id,
            status="failed",
            total_steps=len(pipeline.steps),
            completed_steps=0,
            failed_steps=1,
            results=[PipelineStepResult(
                step_id="__dag__",
                operation="validation",
                status="failed",
                duration_ms=0,
                error="Circular dependency detected in pipeline steps",
            )],
            total_duration_ms=0,
        )

    completed_ids: set[str] = set()

    for step in execution_order:
        if status == "failed":
            break

        deps_satisfied = all(d in completed_ids for d in step.depends_on)
        if not deps_satisfied:
            step_results.append(PipelineStepResult(
                step_id=step.step_id,
                operation=step.operation,
                status="skipped",
                duration_ms=0,
                error="Dependencies not satisfied",
            ))
            continue

        step_start = time.time()
        try:
            result = _run_step(current_ds, step, work_dir)
            if isinstance(result, CVDataset):
                current_ds = result
            step_duration = (time.time() - step_start) * 1000
            step_results.append(PipelineStepResult(
                step_id=step.step_id,
                operation=step.operation,
                status="completed",
                duration_ms=round(step_duration, 2),
                output=str(result)[:500] if result else "",
            ))
            completed += 1
            completed_ids.add(step.step_id)
        except Exception as e:
            step_duration = (time.time() - step_start) * 1000
            step_results.append(PipelineStepResult(
                step_id=step.step_id,
                operation=step.operation,
                status="failed",
                duration_ms=round(step_duration, 2),
                error=str(e),
            ))
            failed += 1
            if step.params.get("fail_fast", True):
                status = "failed"
                break

    total_duration = (time.time() - total_start) * 1000

    if len(pipeline.steps) == 0:
        status = "completed"
    elif failed > 0 and status != "failed":
        status = "completed_with_errors"
    elif failed == 0 and completed > 0:
        status = "completed"

    output_ds_id = current_ds.dataset_id if current_ds is not dataset else None
    if output_ds_id and output_ds_id != dataset.dataset_id:
        try:
            register_dataset(current_ds.to_info())
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to register output dataset: {e}")

    return PipelineResult(
        pipeline_id=pipeline.pipeline_id,
        dataset_id=dataset.dataset_id,
        status=status,
        total_steps=len(pipeline.steps),
        completed_steps=completed,
        failed_steps=failed,
        results=step_results,
        total_duration_ms=round(total_duration, 2),
        output_dataset_id=output_ds_id,
    )


def _run_step(dataset: CVDataset, step: PipelineStep, work_dir: str) -> Any:
    op = step.operation
    params = dict(step.params)
    fn = _get_operation_fn(op)

    if fn is None:
        raise ValueError(f"Unknown pipeline operation: {op}")

    if op in ("export_pytorch", "export_tensorflow", "export_ultralytics", "export_hdf5"):
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        params.setdefault("image_size", params.pop("image_size", (640, 640)))
        return fn(dataset, **params)

    elif op in ("export_webdataset",):
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        return fn(dataset, **params)

    elif op == "export_cvat_xml":
        params.setdefault("output_path", os.path.join(work_dir, f"{step.step_id}.xml"))
        return fn(dataset, **params)

    elif op == "export_labelme_json":
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        return fn(dataset, **params)

    elif op == "export_fiftyone":
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        return fn(dataset, **params)

    elif op in ("resize_dataset", "augment_dataset", "normalize_dataset"):
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        result = fn(dataset, **params)
        if hasattr(result, "output_dir") and result.output_dir:
            loaded = load_dataset(result.output_dir, format="folder")
            return loaded
        return result

    elif op == "convert_format":
        params.setdefault("output_dir", os.path.join(work_dir, step.step_id))
        result = fn(dataset, **params)
        if hasattr(result, "output_dir") and result.output_dir:
            loaded = load_dataset(result.output_dir, format=params.get("target_format", "voc"))
            return loaded
        return result

    elif op == "split_dataset":
        return fn(dataset, **params)

    elif op == "generate_kfold":
        return fn(dataset, **params)

    else:
        return fn(dataset, **params)


def _build_dag(steps: list[PipelineStep]) -> dict[str, list[str]]:
    dag: dict[str, list[str]] = {}
    for step in steps:
        dag[step.step_id] = step.depends_on
    return dag


def _topological_sort(
    dag: dict[str, list[str]], steps: list[PipelineStep],
) -> list[PipelineStep] | None:
    step_ids = {s.step_id for s in steps}
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    for step_id, deps in dag.items():
        if step_id not in in_degree:
            continue
        for dep in deps:
            if dep in step_ids:
                in_degree[step_id] = in_degree.get(step_id, 0) + 1

    step_map = {s.step_id: s for s in steps}
    queue = [sid for sid, deg in in_degree.items() if deg == 0 and sid in step_map]
    sorted_order: list[PipelineStep] = []

    while queue:
        sid = queue.pop(0)
        if sid in step_map:
            sorted_order.append(step_map[sid])
        for other_sid, other_deps in dag.items():
            if other_sid not in in_degree:
                continue
            if sid in other_deps and sid in step_ids:
                in_degree[other_sid] -= 1
                if in_degree[other_sid] == 0:
                    queue.append(other_sid)

    if len(sorted_order) != len(steps):
        return None

    return sorted_order


def register_pipeline(pipeline: PipelineDefinition) -> str:
    _PIPELINE_REGISTRY[pipeline.pipeline_id] = pipeline
    return pipeline.pipeline_id


def list_pipelines() -> list[PipelineDefinition]:
    return list(_PIPELINE_REGISTRY.values())


def get_pipeline(pipeline_id: str) -> PipelineDefinition | None:
    return _PIPELINE_REGISTRY.get(pipeline_id)


def remove_pipeline(pipeline_id: str) -> bool:
    return _PIPELINE_REGISTRY.pop(pipeline_id, None) is not None


def save_pipeline_to_json(pipeline: PipelineDefinition, path: str) -> str:
    with open(path, "w") as f:
        f.write(pipeline.model_dump_json(indent=2))
    return path


def load_pipeline_from_json(path: str) -> PipelineDefinition:
    with open(path) as f:
        data = json.load(f)
    return PipelineDefinition(**data)


def get_template(name: str) -> PipelineDefinition | None:
    return _PIPELINE_TEMPLATES.get(name)


def list_templates() -> list[dict[str, str]]:
    return [
        {"id": tid, "name": t.name, "description": t.description}
        for tid, t in _PIPELINE_TEMPLATES.items()
    ]


def _register_template(p: PipelineDefinition) -> None:
    _PIPELINE_TEMPLATES[p.pipeline_id] = p


_register_template(PipelineDefinition(
    pipeline_id="clean_analyze_split",
    name="Clean → Analyze → Split",
    version="1.0",
    description="Validate integrity, analyze quality, remove duplicates, then split into train/val/test",
    steps=[
        PipelineStep(step_id="integrity", operation="check_integrity", params={}, description="Check for corrupt images and missing labels"),
        PipelineStep(step_id="quality", operation="analyze_quality", params={}, depends_on=["integrity"], description="Analyze image quality metrics"),
        PipelineStep(step_id="duplicates", operation="find_duplicates", params={"methods": ["exact", "perceptual"]}, depends_on=["quality"], description="Find and report duplicate images"),
        PipelineStep(step_id="split", operation="split_dataset", params={"strategy": "stratified", "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}, depends_on=["duplicates"], description="Stratified train/val/test split"),
    ],
    tags=["tutorial", "clean", "split"],
))

_register_template(PipelineDefinition(
    pipeline_id="augment_export_pytorch",
    name="Augment → Export PyTorch",
    version="1.0",
    description="Augment dataset with YOLO pipeline, then export to PyTorch format",
    steps=[
        PipelineStep(step_id="resize", operation="resize_dataset", params={"strategy": "letterbox", "size": [640, 640]}, description="Letterbox resize to 640x640"),
        PipelineStep(step_id="augment", operation="augment_dataset", params={"pipeline": "yolo_detection", "multiplier": 3}, depends_on=["resize"], description="Apply YOLO detection augmentation pipeline"),
        PipelineStep(step_id="export", operation="export_pytorch", params={}, depends_on=["augment"], description="Export to PyTorch format"),
    ],
    tags=["augmentation", "pytorch", "export"],
))

_register_template(PipelineDefinition(
    pipeline_id="full_pipeline",
    name="Full CV Pipeline",
    version="1.0",
    description="End-to-end: integrity → quality → duplicates → split → augment → normalize → export all formats",
    steps=[
        PipelineStep(step_id="integrity", operation="check_integrity", params={}, description="Integrity check"),
        PipelineStep(step_id="quality", operation="analyze_quality", params={}, depends_on=["integrity"], description="Quality analysis"),
        PipelineStep(step_id="leakage", operation="detect_leakage", params={}, depends_on=["quality"], description="Check for data leakage"),
        PipelineStep(step_id="split", operation="split_dataset", params={"strategy": "stratified"}, depends_on=["leakage"], description="Stratified split"),
        PipelineStep(step_id="resize", operation="resize_dataset", params={"strategy": "letterbox", "size": [640, 640]}, depends_on=["split"], description="Resize images"),
        PipelineStep(step_id="augment", operation="augment_dataset", params={"pipeline": "yolo_detection", "multiplier": 2}, depends_on=["resize"], description="Data augmentation"),
        PipelineStep(step_id="normalize", operation="normalize_dataset", params={}, depends_on=["augment"], description="Normalize pixel values"),
        PipelineStep(step_id="export_pt", operation="export_pytorch", params={}, depends_on=["normalize"], description="PyTorch export"),
        PipelineStep(step_id="export_ult", operation="export_ultralytics", params={}, depends_on=["normalize"], description="Ultralytics YOLO export"),
    ],
    tags=["end-to-end", "full", "production"],
))
