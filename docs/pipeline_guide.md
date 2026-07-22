# Pipeline Guide

The pipeline engine lets you chain dataset operations into reproducible, parameterized workflows.

## Pipeline Structure

A pipeline is a DAG (Directed Acyclic Graph) of steps:

```python
from pixlint.utils.schemas import PipelineDefinition, PipelineStep

pipeline = PipelineDefinition(
    pipeline_id="my_pipeline",
    name="My Pipeline",
    description="Description of what this pipeline does",
    steps=[
        PipelineStep(
            step_id="step_1",
            operation="check_integrity",
            params={},
            description="First step",
        ),
        PipelineStep(
            step_id="step_2",
            operation="analyze_quality",
            params={"metrics": ["blur", "noise"]},
            depends_on=["step_1"],  # Runs after step_1
            description="Second step depends on first",
        ),
    ],
)
```

## Supported Operations

| Operation | Function | Parameters |
|-----------|----------|------------|
| `check_integrity` | `check_integrity` | `checks`, etc. |
| `analyze_quality` | `analyze_quality` | `metrics`, etc. |
| `find_duplicates` | `find_duplicates` | `methods`, `thresholds` |
| `detect_leakage` | `detect_leakage` | — |
| `split_dataset` | `split_dataset` | `strategy`, `ratios`, `seed` |
| `generate_kfold` | `generate_kfold_splits` | `k`, `strategy` |
| `resize_dataset` | `resize_dataset` | `size`, `strategy` |
| `augment_dataset` | `augment_dataset` | `pipeline`, `multiplier` |
| `normalize_dataset` | `normalize_dataset` | `mean`, `std` |
| `compute_channel_stats` | `compute_channel_stats` | — |
| `convert_format` | `convert_format` | `target_format`, `output_dir` |
| `export_pytorch` | `export_pytorch` | `image_size` |
| `export_tensorflow` | `export_tensorflow` | `image_size` |
| `export_ultralytics` | `export_ultralytics` | `image_size` |
| `export_hdf5` | `export_hdf5` | `image_size` |
| `export_webdataset` | `export_webdataset` | `image_size`, `shard_size` |
| `export_fiftyone` | `export_fiftyone` | `export_format` |
| `export_cvat_xml` | `export_cvat_xml` | `output_path` |
| `export_labelme_json` | `export_labelme_json` | `output_dir` |

## Execution

```python
from pixlint.core.pipeline import execute_pipeline

result = execute_pipeline(dataset, pipeline)
print(f"Status: {result.status}")
print(f"Completed: {result.completed_steps}/{result.total_steps}")
for step_result in result.results:
    print(f"  {step_result.step_id}: {step_result.status} ({step_result.duration_ms:.0f}ms)")
```

Steps that produce output datasets (resize, augment, normalize, convert) automatically chain into the next step's input.

## Pre-built Templates

```python
from pixlint.core.pipeline import get_template, list_templates

# List available templates
for t in list_templates():
    print(f"{t['id']}: {t['name']}")

# Get a template
tpl = get_template("clean_analyze_split")

# Execute it
result = execute_pipeline(dataset, tpl)
```

### Template: `clean_analyze_split`
Integrity → Quality → Duplicates → Stratified Split

### Template: `augment_export_pytorch`
Resize → Augment (3x YOLO) → PyTorch Export

### Template: `full_pipeline`
Integrity → Quality → Leakage → Split → Resize → Augment → Normalize → PyTorch + Ultralytics Export

## Save & Load Pipelines

```python
from pixlint.core.pipeline import (save_pipeline_to_json, load_pipeline_from_json,
                                          register_pipeline, list_pipelines)

# Save
save_pipeline_to_json(pipeline, "my_pipeline.json")

# Load
loaded = load_pipeline_from_json("my_pipeline.json")

# Register for reuse
register_pipeline(loaded)

# List all registered
list_pipelines()
```

## Fail-Fast Behavior

Each step can set `fail_fast` (default: True). When True, the pipeline stops on the first failure. When False, remaining steps still execute:

```python
PipelineStep(
    step_id="optional_step",
    operation="analyze_quality",
    params={"fail_fast": False},
)
```

## Error Handling

- Circular dependencies are detected at validation time
- Missing step dependencies are gracefully skipped
- Each step reports individual duration and error status
- The pipeline result shows per-step status: `completed`, `failed`, or `skipped`
- Overall status is one of: `completed`, `completed_with_errors`, `failed`, `running`
