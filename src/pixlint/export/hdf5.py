from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

# h5py is an optional dependency (the [hdf5] extra). Guard the import so the
# package still imports fine on a core install; export_hdf5 raises a clear error.
_H5PY_AVAILABLE = False
try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    pass


def export_hdf5(
    dataset: CVDataset,
    output_path: str,
    compression: str = "gzip",
    chunk_size: int = 100,
    image_size: tuple[int, int] = (640, 640),
) -> ExportResult:
    if not _H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 export. Install with: pip install pixlint[hdf5]"
        )
    n = len(dataset.images)
    class_names = dataset.get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    with h5py.File(output_path, "w") as f:
        # Create image dataset
        img_dset = f.create_dataset(
            "images", (n, image_size[1], image_size[0], 3),
            dtype=np.uint8, compression=compression,
            # Chunk dims must match data layout (n, H, W, 3), not (W, H).
            chunks=(min(chunk_size, n), image_size[1], image_size[0], 3),
        )
        
        # Create image_ids dataset
        ids_dset = f.create_dataset(
            "image_ids", (n,), dtype=h5py.string_dtype(),
        )

        # Create annotations as JSON strings (one per image)
        ann_dset = f.create_dataset(
            "annotations", (n,), dtype=h5py.string_dtype(), compression=compression,
        )

        export_count = 0
        ann_count = 0

        for i, img in enumerate(dataset.images):
            image = cv2.imread(img.path)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_dset[i] = image_rgb
            ids_dset[i] = img.image_id

            # Build annotation JSON for this image
            annotations_json = []
            for ann in img.annotations:
                label_idx = class_to_idx.get(ann.label, -1)
                if label_idx < 0:
                    continue
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    # Clean label to remove any null bytes
                    clean_label = ann.label.replace('\x00', '').replace('\0', '')
                    annotations_json.append({
                        "label": clean_label,
                        "category_id": label_idx,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    ann_count += 1

            # Store annotations as JSON string
            ann_dset[i] = json.dumps(annotations_json)

            export_count += 1

        f.attrs["num_images"] = export_count
        f.attrs["num_annotations"] = ann_count
        f.attrs["num_classes"] = len(class_names)
        f.attrs["class_names"] = json.dumps(class_names)
        f.attrs["image_size"] = json.dumps(list(image_size))

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="hdf5",
        output_dir=os.path.dirname(output_path),
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def load_hdf5(path: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        images = f["images"][:]
        ids = [id.decode() if isinstance(id, bytes) else id for id in f["image_ids"][:]]
        class_names = json.loads(f.attrs["class_names"])
        image_size = json.loads(f.attrs["image_size"])

        annotations = []
        for i in range(f["annotations"].shape[0]):
            raw = f["annotations"][i]
            if isinstance(raw, bytes):
                raw = raw.decode()
            if raw and raw != "[]":
                try:
                    annotations.append(json.loads(raw))
                except json.JSONDecodeError:
                    annotations.append([])
            else:
                annotations.append([])

    return {
        "images": images,
        "image_ids": ids,
        "class_names": class_names,
        "image_size": image_size,
        "num_images": len(ids),
        "annotations": annotations,
    }
