from __future__ import annotations

import os
import tempfile


from pixlint.core.pipeline import (
    _build_dag,
    _topological_sort,
    execute_pipeline,
    get_template,
    list_pipelines,
    list_templates,
    load_pipeline_from_json,
    register_pipeline,
    save_pipeline_to_json,
    _get_operation_fn,
)
from pixlint.utils.schemas import PipelineDefinition, PipelineStep, PipelineResult


class TestPipelineCore:
    def test_get_operation_known(self):
        fn = _get_operation_fn("find_duplicates")
        assert fn is not None

    def test_get_operation_unknown(self):
        fn = _get_operation_fn("nonexistent")
        assert fn is None

    def test_build_dag(self):
        steps = [
            PipelineStep(step_id="a", operation="check_integrity", depends_on=[]),
            PipelineStep(step_id="b", operation="analyze_quality", depends_on=["a"]),
            PipelineStep(step_id="c", operation="split_dataset", depends_on=["b"]),
        ]
        dag = _build_dag(steps)
        assert dag == {"a": [], "b": ["a"], "c": ["b"]}

    def test_topological_sort_valid(self):
        steps = [
            PipelineStep(step_id="a", operation="integrity", depends_on=[]),
            PipelineStep(step_id="b", operation="quality", depends_on=["a"]),
            PipelineStep(step_id="c", operation="split", depends_on=["b"]),
        ]
        dag = _build_dag(steps)
        result = _topological_sort(dag, steps)
        assert result is not None
        ids = [s.step_id for s in result]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_topological_sort_cycle(self):
        steps = [
            PipelineStep(step_id="a", operation="op1", depends_on=["c"]),
            PipelineStep(step_id="b", operation="op2", depends_on=["a"]),
            PipelineStep(step_id="c", operation="op3", depends_on=["b"]),
        ]
        dag = _build_dag(steps)
        result = _topological_sort(dag, steps)
        assert result is None

    def test_topological_sort_disconnected(self):
        steps = [
            PipelineStep(step_id="a", operation="op1", depends_on=[]),
            PipelineStep(step_id="b", operation="op2", depends_on=[]),
        ]
        dag = _build_dag(steps)
        result = _topological_sort(dag, steps)
        assert result is not None
        assert len(result) == 2

    def test_register_and_list_pipeline(self):
        p = PipelineDefinition(
            pipeline_id="test", name="Test", steps=[
                PipelineStep(step_id="s1", operation="check_integrity"),
            ],
        )
        pid = register_pipeline(p)
        assert pid == "test"
        pipelines = list_pipelines()
        assert any(pl.pipeline_id == "test" for pl in pipelines)


class TestTemplates:
    def test_list_templates(self):
        templates = list_templates()
        assert len(templates) >= 3

    def test_get_template_known(self):
        tpl = get_template("clean_analyze_split")
        assert tpl is not None
        assert tpl.pipeline_id == "clean_analyze_split"
        assert len(tpl.steps) >= 4

    def test_get_template_unknown(self):
        tpl = get_template("nonexistent")
        assert tpl is None

    def test_template_structure(self):
        tpl = get_template("full_pipeline")
        assert tpl is not None
        assert len(tpl.steps) >= 8
        has_export = any("export" in s.operation for s in tpl.steps)
        assert has_export

    def test_template_dependencies_valid(self):
        for tpl_id in ["clean_analyze_split", "augment_export_pytorch", "full_pipeline"]:
            tpl = get_template(tpl_id)
            assert tpl is not None
            dag = _build_dag(tpl.steps)
            result = _topological_sort(dag, tpl.steps)
            assert result is not None, f"Template {tpl_id} has circular dependencies"


class TestPipelineSaveLoad:
    def test_save_and_load_json(self):
        p = PipelineDefinition(
            pipeline_id="save_test", name="Save Test", steps=[
                PipelineStep(step_id="s1", operation="check_integrity"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pipeline.json")
            save_pipeline_to_json(p, path)
            assert os.path.exists(path)
            loaded = load_pipeline_from_json(path)
            assert loaded.pipeline_id == "save_test"
            assert len(loaded.steps) == 1


class TestPipelineExecution:
    def test_execute_empty_pipeline(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="empty", name="Empty", steps=[],
        )
        result = execute_pipeline(folder_dataset, p)
        assert isinstance(result, PipelineResult)
        assert result.status == "completed"

    def test_execute_integrity_step(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="test_integrity", name="Test Integrity", steps=[
                PipelineStep(step_id="s1", operation="check_integrity"),
            ],
        )
        result = execute_pipeline(folder_dataset, p)
        assert result.total_steps == 1
        assert result.completed_steps == 1

    def test_execute_unknown_op(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="test_unknown", name="Test Unknown", steps=[
                PipelineStep(step_id="s1", operation="nonexistent_op"),
            ],
        )
        result = execute_pipeline(folder_dataset, p)
        assert result.failed_steps >= 1

    def test_execute_with_dependency(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="test_dep", name="Test Dep", steps=[
                PipelineStep(step_id="s1", operation="check_integrity"),
                PipelineStep(step_id="s2", operation="analyze_quality", depends_on=["s1"]),
            ],
        )
        result = execute_pipeline(folder_dataset, p)
        assert result.completed_steps == 2

    def test_execute_unsatisfied_dependency(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="test_unsat", name="Test Unsat", steps=[
                PipelineStep(step_id="s1", operation="check_integrity"),
                PipelineStep(step_id="s2", operation="analyze_quality", depends_on=["nonexistent"]),
            ],
        )
        result = execute_pipeline(folder_dataset, p)
        assert result.failed_steps == 0
        assert result.status in ("completed", "completed_with_errors")

    def test_execute_fail_fast(self, folder_dataset):
        p = PipelineDefinition(
            pipeline_id="test_failfast", name="Test FailFast", steps=[
                PipelineStep(step_id="s1", operation="nonexistent_op", params={"fail_fast": True}),
                PipelineStep(step_id="s2", operation="check_integrity"),
            ],
        )
        result = execute_pipeline(folder_dataset, p)
        assert result.failed_steps >= 1
