# PixLint — Pipeline Example

"""Create and execute a custom pipeline."""
from pixlint.core.loader import load_dataset
from pixlint.core.pipeline import (
    execute_pipeline, get_template, save_pipeline_to_json,
)
from pixlint.utils.schemas import PipelineDefinition, PipelineStep

# 1. Load a dataset
ds = load_dataset("/path/to/your/dataset")
print(f"Loaded: {ds.name} ({len(ds)} images)")

# 2. Create a custom pipeline
pipeline = PipelineDefinition(
    pipeline_id="custom_pipeline",
    name="My Custom Pipeline",
    version="1.0",
    description="Analyze → Split → Export",
    steps=[
        PipelineStep(
            step_id="quality",
            operation="analyze_quality",
            params={"metrics": ["blur", "exposure", "noise"]},
            description="Check image quality metrics",
        ),
        PipelineStep(
            step_id="integrity",
            operation="check_integrity",
            params={},
            description="Validate dataset integrity",
        ),
        PipelineStep(
            step_id="split",
            operation="split_dataset",
            params={"strategy": "stratified", "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}},
            depends_on=["quality", "integrity"],
            description="Split after analysis completes",
        ),
    ],
    tags=["example", "custom"],
)

# 3. Save the pipeline for later reuse
save_pipeline_to_json(pipeline, "custom_pipeline.json")
print("Pipeline saved to custom_pipeline.json")

# 4. Execute the pipeline
result = execute_pipeline(ds, pipeline)
print(f"\nPipeline result: {result.status}")
print(f"Steps: {result.completed_steps}/{result.total_steps} completed")
for step_result in result.results:
    status_icon = "✓" if step_result.status == "completed" else "✗"
    print(f"  {status_icon} {step_result.step_id}: {step_result.status} ({step_result.duration_ms:.0f}ms)")

# 5. Or use a pre-built template
print("\n--- Using template: clean_analyze_split ---")
tpl = get_template("clean_analyze_split")
result2 = execute_pipeline(ds, tpl)
print(f"Template result: {result2.status}")
print(f"Steps: {result2.completed_steps}/{result2.total_steps} completed")
