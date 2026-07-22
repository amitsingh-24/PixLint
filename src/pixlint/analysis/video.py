from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DatasetFormat, SplitResult

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Many containers/streams report FPS as 0 or NaN; fall back to a sane
    # default so timestamp computation never divides by zero.
    if not fps or fps <= 0 or fps != fps:  # fps != fps catches NaN
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_stem = Path(video_path).stem

    extracted = 0
    frame_idx = 0
    timestamps: list[float] = []
    file_paths: list[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if max_frames and extracted >= max_frames:
                break

            if resize:
                frame = cv2.resize(frame, resize)

            timestamp = frame_idx / fps
            out_name = f"{video_stem}_frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)

            file_paths.append(out_path)
            timestamps.append(timestamp)
            extracted += 1

        frame_idx += 1

    cap.release()

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "original_size": (width, height),
        "extracted_frames": extracted,
        "frame_interval": frame_interval,
        "output_dir": output_dir,
        "timestamps": timestamps[:10],
        "file_paths": file_paths[:5],
    }


def video_to_dataset(
    video_path: str,
    output_dir: str | None = None,
    frame_interval: int = 30,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
    name: str | None = None,
) -> CVDataset:
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(video_path),
            f"{Path(video_path).stem}_frames",
        )

    extract_frames(
        video_path, output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        resize=resize,
    )

    return CVDataset(output_dir, format=DatasetFormat.FOLDER, name=name or Path(video_path).stem)


def video_batch_to_datasets(
    video_dir: str,
    output_base: str | None = None,
    frame_interval: int = 30,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    results: list[dict[str, Any]] = []

    for f in sorted(Path(video_dir).iterdir()):
        if f.suffix.lower() in video_extensions:
            out_dir = os.path.join(
                output_base or os.path.join(video_dir, "frames"),
                f.stem,
            ) if output_base else None

            ds = video_to_dataset(
                str(f), output_dir=out_dir,
                frame_interval=frame_interval,
                max_frames=max_frames,
                resize=resize,
                name=f.stem,
            )
            results.append({
                "video": str(f),
                "dataset_id": ds.dataset_id,
                "num_frames": len(ds),
            })

    return results


def temporal_split(
    dataset: CVDataset,
    ratios: dict[str, float] | None = None,
    order_by: str = "filename",
) -> SplitResult:
    if ratios is None:
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    sorted_images = sorted(dataset.images, key=lambda img: img.metadata.get(order_by, img.image_id))
    n = len(sorted_images)

    splits: dict[str, list[str]] = {}
    start = 0
    for split_name, ratio in ratios.items():
        if split_name == list(ratios.keys())[-1]:
            count = n - start
        else:
            count = int(n * ratio)
        end = start + count
        splits[split_name] = [img.image_id for img in sorted_images[start:end]]
        start = end

    return SplitResult(
        dataset_id=dataset.dataset_id,
        strategy="temporal",
        ratios=ratios,
        splits=splits,
    )
