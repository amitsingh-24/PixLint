from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches  # noqa: F401
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]


def _get_color(label: str) -> tuple[int, int, int]:
    idx = abs(hash(label)) % len(_COLORS)
    return _COLORS[idx]


def draw_annotations(
    image: np.ndarray,
    annotations: list[Any],
    class_names: list[str] | None = None,
) -> np.ndarray:
    result = image.copy()
    for ann in annotations:
        label = ann.label
        color = _get_color(label)
        if ann.bbox:
            x1, y1, x2, y2 = [int(v) for v in ann.bbox]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, label_text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Only polygon segmentation is drawable here; skip COCO RLE crowd masks (dict form).
        if ann.segmentation and isinstance(ann.segmentation, list):
            for seg in ann.segmentation:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(result, [pts], True, color, 2)
    return result


def preview_images(
    dataset: CVDataset,
    n_samples: int = 16,
    show_annotations: bool = True,
    grid_size: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> dict[str, Any]:
    n = min(n_samples, len(dataset.images))
    if n == 0:
        return {"error": "No images in dataset"}

    if grid_size:
        cols, rows = grid_size
    else:
        cols = min(4, n)
        rows = (n + cols - 1) // cols

    if not _MATPLOTLIB_AVAILABLE:
        return _preview_images_opencv(dataset, n, show_annotations)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(min(n, rows * cols)):
        img_record = dataset.images[i]
        image = read_image(img_record.path)
        if image is None:
            axes[i].text(0.5, 0.5, "Failed to load", ha="center", va="center")
            axes[i].axis("off")
            continue

        if show_annotations and img_record.annotations:
            image = draw_annotations(image, img_record.annotations)

        axes[i].imshow(image)
        axes[i].set_title(f"{img_record.image_id[:20]}", fontsize=8)
        axes[i].axis("off")

    for j in range(n, rows * cols):
        axes[j].axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"n_samples": n, "save_path": save_path, "grid": f"{cols}x{rows}"}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"n_samples": n, "image_base64": img_b64, "grid": f"{cols}x{rows}"}


def _preview_images_opencv(
    dataset: CVDataset, n: int, show_annotations: bool
) -> dict[str, Any]:
    previews: list[dict[str, Any]] = []
    for i in range(min(n, len(dataset.images))):
        img_record = dataset.images[i]
        image = read_image(img_record.path)
        if image is None:
            continue
        if show_annotations and img_record.annotations:
            image = draw_annotations(image, img_record.annotations)
        h, w = image.shape[:2]
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        import base64
        img_b64 = base64.b64encode(buffer).decode()
        previews.append({
            "image_id": img_record.image_id,
            "width": w,
            "height": h,
            "image_base64": img_b64,
        })
    return {"n_samples": len(previews), "previews": previews}


def preview_single_image(
    dataset: CVDataset,
    image_id: str,
    show_annotations: bool = True,
) -> dict[str, Any]:
    for img_record in dataset.images:
        if img_record.image_id == image_id:
            image = read_image(img_record.path)
            if image is None:
                return {"error": "Failed to load image"}
            if show_annotations and img_record.annotations:
                image = draw_annotations(image, img_record.annotations)
            h, w = image.shape[:2]
            num_anns = len(img_record.annotations)
            labels = list(set(ann.label for ann in img_record.annotations))
            return {
                "image_id": image_id,
                "width": w,
                "height": h,
                "num_annotations": num_anns,
                "labels": labels,
                "shape": list(image.shape),
            }
    return {"error": f"Image '{image_id}' not found"}
