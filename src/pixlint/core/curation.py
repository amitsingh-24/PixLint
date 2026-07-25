"""Dataset curation: produce NEW datasets by filtering, cleaning, subsetting,
and remapping classes.

Unlike the analysis modules (which only *report* problems), these operations
*apply* changes and return a new, registerable dataset — the key capability that
turns pixlint from a linter into a curation tool.

Every operation works in-memory by default (fast, no file copies) and optionally
materializes the result to disk as a COCO dataset when ``output_dir`` is given.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import get_image_size
from pixlint.utils.schemas import (
    Annotation,
    CurationResult,
    DatasetFormat,
    ImageRecord,
)


def _new_dataset_id(source_id: str, suffix: str) -> str:
    return f"{source_id}_{suffix}_{uuid.uuid4().hex[:6]}"


def build_in_memory_dataset(
    name: str,
    images: list[ImageRecord],
    path: str,
    fmt: DatasetFormat,
    dataset_id: str,
) -> CVDataset:
    """Construct a CVDataset from in-memory records without touching disk."""
    ds = CVDataset.__new__(CVDataset)
    from pathlib import Path

    ds.path = Path(path)
    ds.recursive = True
    ds._name = name
    ds._format = fmt
    ds._dataset_id = dataset_id
    ds._images = list(images)
    return ds


def _count_anns(images: list[ImageRecord]) -> int:
    return sum(len(img.annotations) for img in images)


def materialize_coco(images: list[ImageRecord], output_dir: str) -> tuple[int, int]:
    """Write ``images`` (with annotations) to ``output_dir`` as a COCO dataset.

    Returns (num_images_written, num_annotations_written).
    """
    images_dir = os.path.join(output_dir, "images")
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    class_names = sorted({ann.label for img in images for ann in img.annotations})
    class_to_id = {name: i for i, name in enumerate(class_names)}

    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for n, i in class_to_id.items()],
    }

    ann_id = 1
    n_imgs = 0
    for img in images:
        if not os.path.exists(img.path):
            continue
        filename = f"{img.image_id}{os.path.splitext(img.path)[1] or '.jpg'}"
        dst = os.path.join(images_dir, filename)
        try:
            shutil.copy2(img.path, dst)
        except OSError:
            continue
        w, h = img.width, img.height
        if not (w and h):
            size = get_image_size(dst)
            if size:
                w, h = size
        img_id_int = n_imgs + 1
        coco["images"].append({
            "id": img_id_int,
            "file_name": filename,
            "width": w or 0,
            "height": h or 0,
        })
        for ann in img.annotations:
            entry: dict[str, Any] = {
                "id": ann_id,
                "image_id": img_id_int,
                "category_id": class_to_id.get(ann.label, 0),
                "iscrowd": 1 if ann.iscrowd else 0,
            }
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                bw, bh = x2 - x1, y2 - y1
                entry["bbox"] = [float(x1), float(y1), float(bw), float(bh)]
                entry["area"] = float(ann.area if ann.area else bw * bh)
            else:
                entry["bbox"] = []
                entry["area"] = float(ann.area or 0)
            if ann.segmentation:
                entry["segmentation"] = ann.segmentation
            coco["annotations"].append(entry)
            ann_id += 1
        n_imgs += 1

    with open(os.path.join(ann_dir, "instances.json"), "w") as f:
        json.dump(coco, f, indent=2)

    return n_imgs, ann_id - 1


def _finalize(
    source: CVDataset,
    kept: list[ImageRecord],
    operation: str,
    suffix: str,
    removed_ids: list[str],
    notes: list[str],
    output_dir: str | None,
) -> tuple[CVDataset, CurationResult]:
    new_id = _new_dataset_id(source.dataset_id, suffix)
    new_name = f"{source.name}_{suffix}"
    out_dir = None
    if output_dir:
        materialize_coco(kept, output_dir)
        out_dir = output_dir
        new_ds = build_in_memory_dataset(new_name, kept, output_dir, DatasetFormat.COCO, new_id)
    else:
        new_ds = build_in_memory_dataset(new_name, kept, str(source.path), source.format, new_id)

    result = CurationResult(
        source_dataset_id=source.dataset_id,
        new_dataset_id=new_id,
        new_dataset_name=new_name,
        operation=operation,
        num_images_in=len(source.images),
        num_images_out=len(kept),
        num_annotations_in=_count_anns(source.images),
        num_annotations_out=_count_anns(kept),
        removed_image_ids=removed_ids[:1000],
        output_dir=out_dir,
        notes=notes,
    )
    return new_ds, result


def filter_dataset(
    dataset: CVDataset,
    classes: list[str] | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    has_annotations: bool | None = None,
    image_ids: list[str] | None = None,
    output_dir: str | None = None,
) -> tuple[CVDataset, CurationResult]:
    """Build a subset dataset by predicates.

    - ``classes``: keep only annotations of these labels (and images that then
      still have at least one annotation, unless ``has_annotations`` is False).
    - ``min_area``/``max_area``: keep only boxes within the area range.
    - ``has_annotations``: True keeps only images with >=1 annotation; False
      keeps only images with none.
    - ``image_ids``: keep only these image IDs (intersected with other filters).
    """
    class_set = set(classes) if classes else None
    id_set = set(image_ids) if image_ids else None
    notes: list[str] = []

    kept: list[ImageRecord] = []
    removed: list[str] = []
    for img in dataset.images:
        if id_set is not None and img.image_id not in id_set:
            removed.append(img.image_id)
            continue

        new_anns: list[Annotation] = []
        for ann in img.annotations:
            if class_set is not None and ann.label not in class_set:
                continue
            area = ann.area
            if area is None and ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                area = abs(x2 - x1) * abs(y2 - y1)
            if min_area is not None and (area is None or area < min_area):
                continue
            if max_area is not None and (area is None or area > max_area):
                continue
            new_anns.append(ann)

        if has_annotations is True and not new_anns:
            removed.append(img.image_id)
            continue
        if has_annotations is False and new_anns:
            removed.append(img.image_id)
            continue
        # If a class/area filter removed all annotations but the user didn't
        # explicitly ask to keep empty images, drop the now-empty image.
        if (class_set is not None or min_area is not None or max_area is not None) \
                and not new_anns and has_annotations is not False:
            removed.append(img.image_id)
            continue

        kept.append(img.model_copy(update={"annotations": new_anns}))

    if class_set:
        notes.append(f"Kept classes: {sorted(class_set)}")
    return _finalize(dataset, kept, "filter", "filtered", removed, notes, output_dir)


def clean_dataset(
    dataset: CVDataset,
    drop_corrupt: bool = True,
    clip_out_of_bounds: bool = True,
    drop_degenerate: bool = True,
    drop_duplicates: bool = False,
    output_dir: str | None = None,
) -> tuple[CVDataset, CurationResult]:
    """Apply integrity/quality fixes and produce a cleaned dataset."""
    from pixlint.utils.image_io import read_image

    notes: list[str] = []
    removed: list[str] = []
    dup_ids: set[str] = set()

    if drop_duplicates:
        from pixlint.analysis.duplicates import find_duplicates
        report = find_duplicates(dataset, methods=["exact", "perceptual"])
        for group in report.duplicate_groups:
            keeper = group.recommended_keeper or (group.image_ids[0] if group.image_ids else None)
            for iid in group.image_ids:
                if iid != keeper:
                    dup_ids.add(iid)
        if dup_ids:
            notes.append(f"Removed {len(dup_ids)} duplicate images")

    n_clipped = 0
    n_degenerate = 0
    kept: list[ImageRecord] = []
    for img in dataset.images:
        if img.image_id in dup_ids:
            removed.append(img.image_id)
            continue

        w, h = img.width, img.height
        if drop_corrupt:
            if read_image(img.path) is None:
                removed.append(img.image_id)
                continue

        new_anns: list[Annotation] = []
        for ann in img.annotations:
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
                if clip_out_of_bounds and w and h:
                    nx1, ny1 = max(0.0, min(x1, w)), max(0.0, min(y1, h))
                    nx2, ny2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
                    if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                        n_clipped += 1
                    x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                if drop_degenerate and (x2 - x1 <= 0 or y2 - y1 <= 0):
                    n_degenerate += 1
                    continue
                new_anns.append(ann.model_copy(update={"bbox": (x1, y1, x2, y2)}))
            else:
                new_anns.append(ann)
        kept.append(img.model_copy(update={"annotations": new_anns}))

    if clip_out_of_bounds:
        notes.append(f"Clipped {n_clipped} out-of-bounds boxes")
    if drop_degenerate:
        notes.append(f"Dropped {n_degenerate} degenerate (zero-area) boxes")
    if drop_corrupt:
        notes.append(f"Removed {len([r for r in removed if r not in dup_ids])} corrupt/unreadable images")

    return _finalize(dataset, kept, "clean", "clean", removed, notes, output_dir)


def remap_classes(
    dataset: CVDataset,
    mapping: dict[str, str],
    drop_unmapped: bool = False,
    output_dir: str | None = None,
) -> tuple[CVDataset, CurationResult]:
    """Rename / merge / drop classes.

    ``mapping`` maps old label -> new label. Map a label to "" (empty) or set
    ``drop_unmapped`` to remove annotations. Merging happens naturally when two
    old labels map to the same new label.
    """
    notes: list[str] = []
    removed: list[str] = []
    kept: list[ImageRecord] = []
    for img in dataset.images:
        new_anns: list[Annotation] = []
        for ann in img.annotations:
            if ann.label in mapping:
                new_label = mapping[ann.label]
                if not new_label:  # mapped to empty -> drop
                    continue
                new_anns.append(ann.model_copy(update={"label": new_label}))
            elif drop_unmapped:
                continue
            else:
                new_anns.append(ann)
        kept.append(img.model_copy(update={"annotations": new_anns}))

    notes.append(f"Applied {len(mapping)} class mappings" + (" (dropped unmapped)" if drop_unmapped else ""))
    return _finalize(dataset, kept, "remap", "remapped", removed, notes, output_dir)
