# Pipeline Templates

PixLint includes pre-built pipeline templates for common CV workflows.

## Built-in Templates

### `clean_analyze_split`
**Description:** Validate integrity → Analyze quality → Find duplicates → Stratified split

**Steps:**
1. `check_integrity` — Corrupt images, missing labels, annotation bounds
2. `analyze_quality` — Blur, exposure, noise, contrast, resolution, color
3. `find_duplicates` — Exact + perceptual hash
4. `split_dataset` — Stratified 70/15/15

**Use case:** Initial dataset audit before training

```python
from pixlint.core.pipeline import execute_pipeline, get_template

tpl = get_template("clean_analyze_split")
result = execute_pipeline(dataset, tpl, work_dir="/workspace/pipeline_out")
```

---

### `augment_export_pytorch`
**Description:** Resize → Augment (YOLO) → Export PyTorch

**Steps:**
1. `resize_dataset` — Letterbox 640×640
2. `augment_dataset` — YOLO detection pipeline, 3× multiplier
3. `export_pytorch` — PyTorch format with `dataset.py`

**Use case:** Quick YOLO training prep

---

### `full_pipeline`
**Description:** End-to-end: Integrity → Quality → Leakage → Split → Resize → Augment → Normalize → Export all formats

**Steps:**
1. `check_integrity`
2. `analyze_quality`
3. `detect_leakage`
4. `split_dataset` — Stratified
5. `resize_dataset` — Letterbox 640×640
6. `augment_dataset` — YOLO detection, 2×
7. `normalize_dataset` — ImageNet stats
8. `export_pytorch`
9. `export_ultralytics`

**Use case:** Production dataset preparation

---

## Custom Pipelines

Define as `PipelineDefinition`:

```python
from pixlint.utils.schemas import PipelineDefinition, PipelineStep

pipeline = PipelineDefinition(
    pipeline_id="my_custom_pipeline",
    name="Custom Segmentation Prep",
    version="1.0",
    description="Resize → Augment (segmentation) → Normalize → Export",
    steps=[
        PipelineStep(
            step_id="resize",
            operation="resize_dataset",
            params={"size": [512, 512], "strategy": "letterbox"},
            description="Resize to 512x512"
        ),
        PipelineStep(
            step_id="augment",
            operation="augment_dataset",
            params={"pipeline": "segmentation", "multiplier": 4},
            depends_on=["resize"],
            description="Segmentation augmentation 4x"
        ),
        PipelineStep(
            step_id="normalize",
            operation="normalize_dataset",
            params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            depends_on=["augment"],
            description="ImageNet normalization"
        ),
        PipelineStep(
            step_id="export",
            operation="export_pytorch",
            params={"image_size": [512, 512]},
            depends_on=["normalize"],
            description="Export PyTorch"
        ),
    ],
    tags=["segmentation", "custom"]
)

# Register and execute
from pixlint.core.pipeline import register_pipeline, execute_pipeline
register_pipeline(pipeline)
result = execute_pipeline(dataset, pipeline, work_dir="/workspace/my_pipeline")
```

---

## Pipeline Execution

```python
from pixlint.core.pipeline import execute_pipeline, execute_template

# Execute template
result = execute_template(dataset, "clean_analyze_split", work_dir="/workspace/out")

# Execute custom pipeline
result = execute_pipeline(dataset, my_pipeline, work_dir="/workspace/out")

# Check result
print(f"Status: {result.status}")  # running, completed, completed_with_errors, failed
print(f"Steps: {result.completed_steps}/{result.total_steps}")
print(f"Duration: {result.total_duration_ms}ms")

# Access output dataset
if result.output_dataset_id:
    output_ds = load_dataset_by_id(result.output_dataset_id)
```

---

## Pipeline Safety

Pipelines are validated before execution:

1. **AST Validation** — No `exec`, `eval`, `compile`, `__import__`, `subprocess`, `os.system`, etc.
2. **DAG Check** — No circular dependencies
3. **Operation Whitelist** — Only registered operations allowed
4. **Resource Limits** — Enforced per-step via `ResourceLimiter`
5. **Audit Log** — Every step logged to `CV_SECURITY_LOG_FILE`

---

## CLI Usage

```bash
# List templates
pixlint --list-templates

# Execute template (via MCP tool)
# Assistant calls execute_template_tool(dataset_id="ds1", template_id="clean_analyze_split")
```