# PixLint — Augment & Export Example

"""Augment a dataset and export to multiple formats."""
from pixlint.core.loader import load_dataset
from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.transformation.resize import resize_dataset
from pixlint.export.pytorch import export_pytorch
from pixlint.export.ultralytics import export_ultralytics
from pixlint.export.extra_formats import export_webdataset, export_cvat_xml
from pixlint.splitting.splitter import split_dataset

# Load
ds = load_dataset("/path/to/your/dataset")
print(f"Loaded: {ds.name} ({len(ds)} images, {ds.get_num_annotations()} annotations)")

# 1. Preview augmentation
preview = preview_augmentation(ds, pipeline="yolo_detection", n_samples=4)
print(f"Augmentation preview: {preview['n_samples']} samples")

# 2. Augment
aug_result = augment_dataset(ds, pipeline="yolo_detection", multiplier=3, output_dir="./augmented")
print(f"Generated: {aug_result.total_generated} images")

# 3. Resize
resize_result = resize_dataset(ds, size=(640, 640), strategy="letterbox", output_dir="./resized")
print(f"Resized: {resize_result.num_images_resized} images")

# 4. Split
split = split_dataset(ds, strategy="stratified", ratios={"train": 0.7, "val": 0.15, "test": 0.15})
print(f"Split: train={len(split.splits['train'])}, val={len(split.splits['val'])}, test={len(split.splits['test'])}")

# 5. Export to multiple formats
pt_result = export_pytorch(ds, output_dir="./export_pt", image_size=(640, 640))
print(f"PyTorch: {pt_result.num_images_exported} images")

yolo_result = export_ultralytics(ds, output_dir="./export_yolo", image_size=640, split=split.splits)
print(f"Ultralytics: {yolo_result.num_images_exported} images")

wds_result = export_webdataset(ds, output_dir="./export_wds", image_size=(640, 640))
print(f"WebDataset: {wds_result.num_images_exported} images")

xml_result = export_cvat_xml(ds, output_path="./export_cvat/annotations.xml")
print(f"CVAT XML: {xml_result.num_images_exported} images")
