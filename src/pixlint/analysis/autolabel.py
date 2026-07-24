"""Auto-labeling: run a pretrained detector over (unlabeled) images to produce
predicted bounding-box annotations, yielding a new pre-annotated dataset.

Uses torchvision's COCO-pretrained Faster R-CNN by default. torch/torchvision
are optional (the ``[torch]`` extra); the import is guarded.
"""
from __future__ import annotations

import uuid

from pixlint.core.curation import build_in_memory_dataset, materialize_coco
from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image
from pixlint.utils.schemas import (
    Annotation,
    AutoLabelResult,
    DatasetFormat,
    ImageRecord,
)

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# COCO 91-class label map (index -> name) used by torchvision detection models.
_COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def auto_label(
    dataset: CVDataset,
    model: str = "fasterrcnn",
    confidence: float = 0.5,
    max_detections: int = 100,
    output_dir: str | None = None,
) -> tuple[CVDataset, AutoLabelResult]:
    """Predict bounding boxes for every image and return a new labeled dataset."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch/torchvision are required for auto-labeling. "
            "Install with: pip install pixlint[torch]"
        )

    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.transforms.functional import to_tensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    net = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

    class_counts: dict[str, int] = {}
    total_preds = 0
    labeled: list[ImageRecord] = []

    for img in dataset.images:
        image = read_image(img.path)  # RGB ndarray
        if image is None:
            labeled.append(img.model_copy(update={"annotations": []}))
            continue
        h, w = image.shape[:2]
        tensor = to_tensor(image).to(device)
        with torch.no_grad():
            out = net([tensor])[0]

        anns: list[Annotation] = []
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        for box, score, lab in zip(boxes, scores, labels):
            if score < confidence:
                continue
            if len(anns) >= max_detections:
                break
            name = _COCO_INSTANCE_CATEGORY_NAMES[lab] if lab < len(_COCO_INSTANCE_CATEGORY_NAMES) else f"class_{lab}"
            if name in ("__background__", "N/A"):
                continue
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            anns.append(Annotation(
                label=name,
                bbox=(x1, y1, x2, y2),
                area=(x2 - x1) * (y2 - y1),
                confidence=float(score),
            ))
            class_counts[name] = class_counts.get(name, 0) + 1
            total_preds += 1

        labeled.append(img.model_copy(update={
            "annotations": anns,
            "width": img.width or w,
            "height": img.height or h,
        }))

    new_id = f"{dataset.dataset_id}_autolabel_{uuid.uuid4().hex[:6]}"
    new_name = f"{dataset.name}_autolabeled"
    out_dir = None
    if output_dir:
        materialize_coco(labeled, output_dir)
        out_dir = output_dir
        new_ds = build_in_memory_dataset(new_name, labeled, output_dir, DatasetFormat.COCO, new_id)
    else:
        new_ds = build_in_memory_dataset(new_name, labeled, str(dataset.path), dataset.format, new_id)

    result = AutoLabelResult(
        source_dataset_id=dataset.dataset_id,
        new_dataset_id=new_id,
        new_dataset_name=new_name,
        model=model,
        confidence_threshold=confidence,
        num_images=len(labeled),
        num_predicted_annotations=total_preds,
        predicted_classes=dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        output_dir=out_dir,
        notes=[f"Auto-labeled with torchvision {model} (COCO-80) on {device}"],
    )
    return new_ds, result
