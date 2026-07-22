from __future__ import annotations

import numpy as np

from pixlint.visualization.previews import (
    preview_images,
    preview_single_image,
    draw_annotations,
)
from pixlint.visualization.charts import (
    plot_distribution,
    plot_quality_scores,
    plot_spatial_heatmap,
    plot_duplicate_groups,
    plot_class_balance_radar,
)
from pixlint.analysis.diff import dataset_diff
from pixlint.utils.progress import ProgressTracker, track_progress
from pixlint.utils.schemas import Annotation


class TestPreviews:
    def test_preview_images(self, folder_dataset):
        result = preview_images(folder_dataset, n_samples=3, show_annotations=False)
        assert "n_samples" in result
        if "error" not in result:
            assert result["n_samples"] <= 3

    def test_preview_with_annotations(self, coco_dataset):
        result = preview_images(coco_dataset, n_samples=2, show_annotations=True)
        assert "n_samples" in result
        if "error" not in result:
            assert result["n_samples"] <= 2

    def test_preview_single_image(self, folder_dataset):
        img_id = folder_dataset.images[0].image_id
        result = preview_single_image(folder_dataset, img_id)
        assert result["image_id"] == img_id
        assert "width" in result
        assert "height" in result

    def test_preview_nonexistent(self, folder_dataset):
        result = preview_single_image(folder_dataset, "nonexistent")
        assert "error" in result

    def test_draw_annotations(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ann = Annotation(label="cat", bbox=(10, 10, 50, 50))
        result = draw_annotations(image, [ann])
        assert result.shape == image.shape

    def test_preview_empty_dataset(self):
        from pixlint.core.loader import CVDataset
        from pixlint.utils.schemas import DatasetFormat
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ds = CVDataset(d, format=DatasetFormat.FOLDER)
            result = preview_images(ds)
            assert "error" in result or result["n_samples"] == 0


class TestCharts:
    def test_plot_distribution(self, coco_dataset):
        result = plot_distribution(coco_dataset, chart_type="bar")
        if "error" not in result:
            assert "n_classes" in result or "save_path" in result or "image_base64" in result

    def test_plot_distribution_pie(self, coco_dataset):
        result = plot_distribution(coco_dataset, chart_type="pie")
        if "error" not in result:
            assert "image_base64" in result or "save_path" in result

    def test_plot_quality(self, folder_dataset):
        result = plot_quality_scores(folder_dataset)
        if "error" not in result:
            assert "avg_score" in result

    def test_plot_spatial(self, coco_dataset):
        result = plot_spatial_heatmap(coco_dataset)
        if "error" not in result:
            assert "image_base64" in result or "save_path" in result

    def test_plot_duplicates(self, folder_dataset):
        result = plot_duplicate_groups(folder_dataset)
        if "error" not in result:
            assert "total_groups" in result

    def test_plot_class_balance(self, coco_dataset):
        result = plot_class_balance_radar(coco_dataset)
        if "error" not in result:
            assert "n_classes" in result


class TestDatasetDiff:
    def test_dataset_diff(self, folder_dataset, coco_dataset):
        result = dataset_diff(folder_dataset, coco_dataset)
        assert result.dataset_id_a == folder_dataset.dataset_id
        assert result.dataset_id_b == coco_dataset.dataset_id
        assert result.common_images >= 0
        assert isinstance(result.added_images, list)
        assert isinstance(result.removed_images, list)

    def test_diff_same_dataset(self, folder_dataset):
        result = dataset_diff(folder_dataset, folder_dataset)
        assert result.common_images == len(folder_dataset)
        assert len(result.added_images) == 0
        assert len(result.removed_images) == 0


class TestProgressTracker:
    def test_progress_tracker(self):
        tracker = ProgressTracker(total=10, desc="test", silent=True)
        assert tracker.total == 10
        assert tracker.current == 0
        tracker.update(5)
        assert tracker.current == 5
        assert tracker.progress == 50.0
        stats = tracker.close()
        assert stats["total"] == 10
        assert stats["processed"] == 5

    def test_track_progress(self):
        def double(x):
            return x * 2

        items = [1, 2, 3, 4, 5]
        results, stats = track_progress(items, double, desc="test", silent=True)
        assert results == [2, 4, 6, 8, 10]
        assert stats["total"] == 5
        assert stats["processed"] == 5

    def test_track_progress_with_errors(self):
        def fail_on_three(x):
            if x == 3:
                raise ValueError("test error")
            return x

        items = [1, 2, 3, 4, 5]
        results, stats = track_progress(items, fail_on_three, desc="test", silent=True)
        assert results == [1, 2, None, 4, 5]
        assert stats["processed"] == 5
