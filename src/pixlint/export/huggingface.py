"""Export a dataset to the Hugging Face `datasets` format, and optionally
publish it to the Hugging Face Hub.

This is a differentiator: it gives users a one-command path to *share* a curated
dataset publicly. The `datasets` / `huggingface_hub` packages are optional; the
import is guarded so the base install stays lightweight.
"""
from __future__ import annotations

import os

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

_HF_AVAILABLE = False
try:
    import datasets as hf_datasets  # noqa: F401
    _HF_AVAILABLE = True
except ImportError:
    pass


def export_huggingface(
    dataset: CVDataset,
    output_dir: str | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    private: bool = True,
) -> ExportResult:
    """Build a HF dataset (image + objects) and save/push it.

    Credentials for pushing are read only from the ``HF_TOKEN`` environment
    variable — never from tool parameters.
    """
    if not _HF_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face export. "
            "Install with: pip install pixlint[huggingface]"
        )

    import datasets

    class_names = dataset.get_class_names()
    class_to_id = {name: i for i, name in enumerate(class_names)}

    records: list[dict] = []
    ann_count = 0
    for img in dataset.images:
        if not os.path.exists(img.path):
            continue
        objects = {"bbox": [], "category": [], "area": []}
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                objects["bbox"].append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                objects["area"].append(float(ann.area or (x2 - x1) * (y2 - y1)))
            else:
                objects["bbox"].append([0.0, 0.0, 0.0, 0.0])
                objects["area"].append(0.0)
            objects["category"].append(class_to_id.get(ann.label, 0))
            ann_count += 1
        records.append({
            "image": img.path,
            "image_id": img.image_id,
            "width": img.width or 0,
            "height": img.height or 0,
            "objects": objects,
        })

    features = datasets.Features({
        "image": datasets.Image(),
        "image_id": datasets.Value("string"),
        "width": datasets.Value("int32"),
        "height": datasets.Value("int32"),
        "objects": datasets.Sequence({
            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
            "category": datasets.ClassLabel(names=class_names) if class_names else datasets.Value("int32"),
            "area": datasets.Value("float32"),
        }),
    })

    # datasets.Sequence of a dict expects column-oriented data; convert.
    hf_ds = datasets.Dataset.from_list(records, features=features)

    resolved_out = output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        hf_ds.save_to_disk(output_dir)

    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id is required to push to the Hugging Face Hub")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable must be set to push to the Hub")
        hf_ds.push_to_hub(repo_id, private=private, token=token)
        resolved_out = f"https://huggingface.co/datasets/{repo_id}"

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="huggingface",
        output_dir=resolved_out or "(in-memory)",
        num_images_exported=len(records),
        num_annotations_exported=ann_count,
    )
