from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from pixlint.analysis.video import (
    extract_frames,
    temporal_split,
    video_batch_to_datasets,
    video_to_dataset,
)


def _create_test_video(path: str, num_frames: int = 60, fps: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (100, 100))
    for i in range(num_frames):
        frame = np.full((100, 100, 3), i * 4, dtype=np.uint8)
        out.write(frame)
    out.release()


class TestVideo:
    def test_extract_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _create_test_video(video_path, num_frames=60, fps=30)
            out_dir = os.path.join(tmpdir, "frames")
            result = extract_frames(video_path, out_dir, frame_interval=15, max_frames=10)
            assert result["extracted_frames"] > 0
            assert result["total_frames"] == 60
            assert result["fps"] == 30

    def test_extract_frames_invalid(self):
        result = extract_frames("/nonexistent/video.mp4", "/tmp/out")
        assert "error" in result

    def test_video_to_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _create_test_video(video_path, num_frames=30, fps=30)
            out_dir = os.path.join(tmpdir, "dataset")
            ds = video_to_dataset(video_path, out_dir, frame_interval=10, max_frames=5)
            assert ds is not None
            assert ds.dataset_id is not None

    def test_video_to_dataset_default_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _create_test_video(video_path, num_frames=20, fps=30)
            ds = video_to_dataset(video_path, frame_interval=10)
            assert ds is not None
            assert len(ds) >= 0

    def test_temporal_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os.path
            ds_path = os.path.join(tmpdir, "images")
            os.makedirs(ds_path, exist_ok=True)
            for i in range(10):
                img = np.full((50, 50, 3), i * 25, dtype=np.uint8)
                cv2.imwrite(os.path.join(ds_path, f"frame_{i:04d}.jpg"), img)

            from pixlint.core.loader import CVDataset
            from pixlint.utils.schemas import DatasetFormat
            ds = CVDataset(ds_path, format=DatasetFormat.FOLDER)
            result = temporal_split(ds)
            assert result.strategy == "temporal"
            assert "train" in result.splits
            assert "val" in result.splits
            assert "test" in result.splits

    def test_batch_video_to_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                vp = os.path.join(tmpdir, f"video_{i}.mp4")
                _create_test_video(vp, num_frames=20, fps=30)
            results = video_batch_to_datasets(tmpdir, frame_interval=10, max_frames=3)
            assert len(results) == 2
            for r in results:
                assert "video" in r
                assert "dataset_id" in r
