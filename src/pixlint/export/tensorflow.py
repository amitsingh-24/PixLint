from __future__ import annotations

import json
import os
from typing import Any

import cv2

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ExportResult

_TF_AVAILABLE = False
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    pass


def export_tensorflow(
    dataset: CVDataset,
    output_dir: str,
    image_size: tuple[int, int] = (640, 640),
    tfrecord_shards: int = 10,
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

        record: dict[str, Any] = {
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
            if ann.bbox:
                x1, y1, x2, y2 = ann.bbox
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

    _generate_tf_dataset_script(output_dir, class_names, image_size)

    if _TF_AVAILABLE and tfrecord_shards > 0:
        _write_tfrecords(metadata, images_dir, output_dir, tfrecord_shards, image_size)

    return ExportResult(
        dataset_id=dataset.dataset_id,
        export_format="tensorflow",
        output_dir=output_dir,
        num_images_exported=export_count,
        num_annotations_exported=ann_count,
    )


def _generate_tf_dataset_script(
    output_dir: str, class_names: list[str], image_size: tuple[int, int]
) -> None:
    script = f'''import json
import os
from pathlib import Path

import tensorflow as tf


def load_tf_dataset(data_dir: str, batch_size: int = 32, shuffle: bool = True):
    data_dir = Path(data_dir)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    file_pattern = str(data_dir / "*.tfrecord")
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))

    feature_description = {{
        "image_id": tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }}

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_jpeg(example["image_raw"], channels=3)
        image = tf.image.resize(image, {image_size})
        image = tf.cast(image, tf.float32) / 255.0
        return image, example["image_id"]

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_tf_dataset(data_dir: str, batch_size: int = 32):
    return load_tf_dataset(data_dir, batch_size=batch_size)
'''
    with open(os.path.join(output_dir, "dataset.py"), "w") as f:
        f.write(script)


def _write_tfrecords(
    metadata: list[dict[str, Any]],
    images_dir: str,
    output_dir: str,
    num_shards: int,
    image_size: tuple[int, int],
) -> None:
    if not _TF_AVAILABLE:
        return

    shard_size = max(1, len(metadata) // num_shards)

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(metadata))
        if start >= end:
            break

        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.tfrecord")
        with tf.io.TFRecordWriter(shard_path) as writer:
            for i in range(start, end):
                sample = metadata[i]
                img_path = os.path.join(images_dir, sample["filename"])
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.resize(image, image_size)
                _, encoded = cv2.imencode(".jpg", image)

                feature = {
                    "image_id": _bytes_feature(sample["image_id"].encode()),
                    "image_raw": _bytes_feature(encoded.tobytes()),
                    "height": _int64_feature(image.shape[0]),
                    "width": _int64_feature(image.shape[1]),
                }

                if sample["annotations"]:
                    ann = sample["annotations"][0]
                    feature["label"] = _int64_feature(ann["category_id"])
                    bbox = ann["bbox"]
                    feature["bbox_x1"] = _int64_feature(int(bbox[0]))
                    feature["bbox_y1"] = _int64_feature(int(bbox[1]))
                    feature["bbox_x2"] = _int64_feature(int(bbox[2]))
                    feature["bbox_y2"] = _int64_feature(int(bbox[3]))

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
