from __future__ import annotations

from collections import OrderedDict

from pixlint.analysis.duplicates import find_duplicates
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import (
    Annotation,
    DatasetFormat,
    ImageRecord,
    LabelMapEntry,
    MergeResult,
)


def merge_datasets(
    datasets: list[CVDataset],
    merged_name: str = "merged_dataset",
    strategy: str = "union",
    deduplicate: bool = True,
    unify_labels: bool = True,
) -> MergeResult:
    if len(datasets) < 2:
        raise ValueError("Need at least 2 datasets to merge")

    source_ids: list[str] = []
    seen_images: OrderedDict[str, ImageRecord] = OrderedDict()
    duplicate_count = 0
    duplicate_removed = 0

    all_labels: dict[str, set[str]] = {}
    label_mapping_list: list[LabelMapEntry] = []

    for ds in datasets:
        source_ids.append(ds.dataset_id)
        for label in ds.get_class_names():
            all_labels.setdefault(label, set()).add(ds.dataset_id)

    if unify_labels and len(all_labels) > 1:
        unified_label_map = _unify_labels(all_labels)
        for source_label, source_dataset_ids in all_labels.items():
            unified = unified_label_map[source_label]
            for source_ds_id in source_dataset_ids:
                label_mapping_list.append(
                    LabelMapEntry(
                        source_label=source_label,
                        source_dataset=source_ds_id,
                        unified_label=unified,
                    )
                )
    else:
        for ds in datasets:
            for label in ds.get_class_names():
                label_mapping_list.append(
                    LabelMapEntry(
                        source_label=label,
                        source_dataset=ds.dataset_id,
                        unified_label=label,
                    )
                )
        unified_label_map = {lbl: lbl for lbl in all_labels}

    for ds_idx, ds in enumerate(datasets):
        for img in ds.images:
            dedup_key = img.image_id
            if strategy == "intersection" and dedup_key not in {
                i.image_id for i in datasets[0].images
            }:
                continue

            if dedup_key in seen_images:
                duplicate_count += 1
                if strategy == "union":
                    existing = seen_images[dedup_key]
                    existing.annotations.extend(
                        _remap_annotations(img.annotations, unified_label_map)
                    )
                    duplicate_removed += 1
                elif strategy == "intersection":
                    pass
                continue

            new_img = ImageRecord(
                image_id=dedup_key,
                path=img.path,
                width=img.width,
                height=img.height,
                annotations=_remap_annotations(img.annotations, unified_label_map),
                metadata={**img.metadata, "source_dataset": ds.dataset_id},
            )
            seen_images[dedup_key] = new_img

    merged = CVDataset.__new__(CVDataset)
    merged._dataset_id = merged_name + "_" + str(abs(hash(merged_name)) % 100000)
    merged._name = merged_name
    merged._format = DatasetFormat.FOLDER
    merged._images = list(seen_images.values())
    merged.path = datasets[0].path

    if deduplicate and len(merged) > 1:
        dups = find_duplicates(merged, methods=["exact"])
        dup_ids = set()
        for group in dups.duplicate_groups:
            for img_id in group.image_ids[1:]:
                dup_ids.add(img_id)
        if dup_ids:
            merged._images = [img for img in merged._images if img.image_id not in dup_ids]
            duplicate_removed += len(dup_ids)

    return MergeResult(
        merged_dataset_id=merged._dataset_id,
        source_datasets=source_ids,
        total_images=len(merged),
        total_annotations=merged.get_num_annotations(),
        num_classes=len(merged.get_class_names()),
        class_names=merged.get_class_names(),
        label_mapping=label_mapping_list,
        duplicate_count=duplicate_count,
        duplicate_removed=duplicate_removed,
    )


def _unify_labels(all_labels: dict[str, set[str]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    label_to_base: dict[str, str] = {}

    for label in all_labels:
        base = label.lower().strip().replace("-", "_").replace(" ", "_")
        label_to_base[label] = base

    base_to_unified: dict[str, str] = {}
    for label, base in label_to_base.items():
        if base not in base_to_unified:
            base_to_unified[base] = label
        else:
            existing = base_to_unified[base]
            if len(label) < len(existing):
                base_to_unified[base] = label

    for label in all_labels:
        base = label_to_base[label]
        normalized[label] = base_to_unified[base]

    return normalized


def _remap_annotations(
    annotations: list[Annotation],
    label_map: dict[str, str],
) -> list[Annotation]:
    return [
        Annotation(
            label=label_map.get(ann.label, ann.label),
            bbox=ann.bbox,
            segmentation=ann.segmentation,
            keypoints=ann.keypoints,
            area=ann.area,
            iscrowd=ann.iscrowd,
            confidence=ann.confidence,
            attributes=ann.attributes,
        )
        for ann in annotations
    ]
