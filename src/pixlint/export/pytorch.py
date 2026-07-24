from __future__ import annotations

import json
import os
from typing import Any, Callable

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader, Dataset  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def export_pytorch(
    dataset: CVDataset,
    output_dir: str,
    image_size: tuple[int, int] = (640, 640),
    split: dict[str, list[str]] | None = None,
    num_workers: int = 4,
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    class_names = dataset.get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    metadata: list[dict[str, Any]] = []
    export_count = 0
    ann_count = 0

    for img in dataset.images:
        image = cv2.imread(img.path)
        if image is None:
            continue
        image = cv2.resize(image, image_size)
        filename = f"{img.image_id}.jpg"
        cv2.imwrite(os.path.join(images_dir, filename), image)

        record = {
            "image_id": img.image_id,
            "filename": filename,
            "width": image_size[0],
            "height": image_size[1],
            "annotations": [],
        }
        for ann in img.annotations:
            label_idx = class_to_idx.get(ann.label, -1)
            if label_idx < 0:
                continue
            bbox = ann.bbox
            if bbox:
                x1, y1, x2, y2 = bbox
                record["annotations"].append({
                    "label": ann.label,
                    "category_id": label_idx,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                })
                ann_count += 1

        metadata.append(record)
        export_count += 1

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "num_images": export_count,
            "num_annotations": ann_count,
            "class_names": class_names,
            "num_classes": len(class_names),
            "image_size": list(image_size),
            "samples": metadata,
        }, f, indent=2)

    _generate_pytorch_dataset_script(output_dir, class_names, image_size)

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="pytorch",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def _generate_pytorch_dataset_script(
    output_dir: str, class_names: list[str], image_size: tuple[int, int]
) -> None:
    class_names_str = str(class_names)
    w, h = image_size
    lines = [
        "from __future__ import annotations",
        "",
        "import json",
        "import os",
        "from pathlib import Path",
        "",
        "import torch",
        "from torch.utils.data import Dataset, DataLoader",
        "from PIL import Image",
        "import torchvision.transforms as T",
        "",
        "",
        "class CVDatasetTorch(Dataset):",
        '    def __init__(self, data_dir: str, split: str = "train", transform=None):',
        "        self.data_dir = Path(data_dir)",
        '        self.images_dir = self.data_dir / "images"',
        '        with open(self.data_dir / "metadata.json") as f:',
        "            self.metadata = json.load(f)",
        "        self.samples = self.metadata['samples']",
        f"        self.class_names = {class_names_str}",
        "        self.transform = transform or T.Compose([",
        f"            T.Resize({image_size}),",
        "            T.ToTensor(),",
        "            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
        "        ])",
        "",
        "    def __len__(self):",
        "        return len(self.samples)",
        "",
        "    def __getitem__(self, idx):",
        "        sample = self.samples[idx]",
        '        img_path = self.images_dir / sample["filename"]',
        '        image = Image.open(img_path).convert("RGB")',
        "        if self.transform:",
        "            image = self.transform(image)",
        "        return image, sample",
        "",
        "",
        "def detection_collate(batch):",
        "    # Samples carry variable-length annotation lists, which the default",
        "    # collate cannot batch. Stack the images and keep targets as a list.",
        "    images = torch.stack([item[0] for item in batch], dim=0)",
        "    targets = [item[1] for item in batch]",
        "    return images, targets",
        "",
        "",
        "def create_dataloader(",
        "    data_dir: str,",
        "    batch_size: int = 16,",
        "    shuffle: bool = True,",
        "    num_workers: int = 4,",
        ") -> DataLoader:",
        "    dataset = CVDatasetTorch(data_dir)",
        "    return DataLoader(",
        "        dataset,",
        "        batch_size=batch_size,",
        "        shuffle=shuffle,",
        "        num_workers=num_workers,",
        "        collate_fn=detection_collate,",
        "    )",
    ]
    with open(os.path.join(output_dir, "dataset.py"), "w") as f:
        f.write("\n".join(lines))


def export_pytorch_dataloader(
    dataset: CVDataset,
    batch_size: int = 16,
    image_size: tuple[int, int] = (640, 640),
    shuffle: bool = True,
    transform: Callable | None = None,
):
    if not _TORCH_AVAILABLE:
        return None

    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    class _CVDatasetWrapper(Dataset):
        def __init__(self, ds, transform):
            self.ds = ds
            self.transform = transform or T.Compose([
                T.ToPILImage(),
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.ds.images)

        def __getitem__(self, idx):
            img = self.ds.images[idx]
            image = cv2.imread(img.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image)
            return tensor, img.image_id

    wrapped = _CVDatasetWrapper(dataset, transform)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=0)
