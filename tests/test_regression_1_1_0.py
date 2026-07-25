"""Regression tests added in 1.1.0.

These pin real-world-data behaviour that synthetic fixtures previously missed:
the COCO bbox xywh->xyxy convention, crowd/RLE segmentation, the KITTI loader,
COCO round-trip integrity, and loader robustness to malformed inputs.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from pixlint.core.curation import clean_dataset
from pixlint.core.loader import CVDataset, load_dataset
from pixlint.utils.schemas import DatasetFormat
from pixlint.visualization.previews import draw_annotations


def _save_img(path, size=(100, 100)):
    Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


# --------------------------------------------------------------------------- #
# COCO bbox convention (the confirmed shipping bug)
# --------------------------------------------------------------------------- #
class TestCocoBboxConvention:
    def test_coco_bbox_is_xyxy(self, coco_dataset):
        # fixture img_0 has COCO bbox [x=10, y=10, w=50, h=50]
        ann = coco_dataset.images[0].annotations[0]
        x1, y1, x2, y2 = ann.bbox
        assert (x1, y1) == (10, 10)
        assert (x2 - x1, y2 - y1) == (50, 50)  # width/height, not (x2,y2)==(50,50)

    def test_coco_bbox_area_matches_field(self, coco_dataset):
        ann = coco_dataset.images[0].annotations[0]
        x1, y1, x2, y2 = ann.bbox
        assert abs((x2 - x1) * (y2 - y1) - ann.area) < 1e-6  # 2500

    def test_coco_roundtrip_preserves_boxes(self, coco_dataset, tmp_path):
        out = str(tmp_path / "roundtrip")
        cleaned, _ = clean_dataset(
            coco_dataset, drop_corrupt=False, clip_out_of_bounds=False,
            drop_degenerate=False, output_dir=out,
        )
        reloaded = load_dataset(out, format="auto")
        orig = {i.image_id: i.annotations[0].bbox for i in coco_dataset.images if i.annotations}
        # match by first annotation bbox on each reloaded image
        for img in reloaded.images:
            if not img.annotations:
                continue
            b = tuple(round(v) for v in img.annotations[0].bbox)
            assert b in {tuple(round(v) for v in ob) for ob in orig.values()}


# --------------------------------------------------------------------------- #
# Crowd / RLE segmentation
# --------------------------------------------------------------------------- #
class TestCocoCrowdRLE:
    def test_rle_segmentation_loads_as_dict(self, coco_crowd_dataset):
        crowd = next(
            a for img in coco_crowd_dataset.images for a in img.annotations if a.iscrowd
        )
        assert isinstance(crowd.segmentation, dict)
        assert "counts" in crowd.segmentation and "size" in crowd.segmentation

    def test_polygon_segmentation_loads_as_list(self, coco_crowd_dataset):
        poly = next(
            a for img in coco_crowd_dataset.images
            for a in img.annotations if not a.iscrowd and a.segmentation
        )
        assert isinstance(poly.segmentation, list)

    def test_preview_does_not_crash_on_rle(self, coco_crowd_dataset):
        img = coco_crowd_dataset.images[0]
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        # must not raise even though one annotation carries an RLE dict
        out = draw_annotations(canvas, img.annotations)
        assert out.shape == canvas.shape

    def test_crowd_roundtrip_preserves_rle(self, coco_crowd_dataset, tmp_path):
        out = str(tmp_path / "crowd_rt")
        clean_dataset(coco_crowd_dataset, drop_corrupt=False, clip_out_of_bounds=False,
                      drop_degenerate=False, output_dir=out)
        data = json.loads((tmp_path / "crowd_rt" / "annotations" / "instances.json").read_text())
        seg_types = [type(a.get("segmentation")).__name__ for a in data["annotations"]]
        assert "dict" in seg_types  # RLE dict survived the export


# --------------------------------------------------------------------------- #
# KITTI loader (previously had zero coverage)
# --------------------------------------------------------------------------- #
class TestKittiLoader:
    def test_detect_kitti(self, temp_kitti_dir):
        assert CVDataset(str(temp_kitti_dir), format=DatasetFormat.AUTO).format == DatasetFormat.KITTI

    def test_kitti_image_count(self, kitti_dataset):
        assert len(kitti_dataset.images) == 3

    def test_kitti_excludes_dontcare(self, kitti_dataset):
        classes = kitti_dataset.get_class_names()
        assert "DontCare" not in classes
        assert set(classes) == {"Car", "Pedestrian"}

    def test_kitti_bbox_is_xyxy(self, kitti_dataset):
        car = next(a for a in kitti_dataset.images[0].annotations if a.label == "Car")
        assert car.bbox == (20.0, 20.0, 80.0, 80.0)


# --------------------------------------------------------------------------- #
# YOLO edge cases
# --------------------------------------------------------------------------- #
class TestYoloEdgeCases:
    def test_yolo_bbox_numeric_conversion(self, yolo_dataset):
        # label "0.5 0.5 0.8 0.8" on a 100x100 image -> (10,10,90,90)
        ann = yolo_dataset.images[0].annotations[0]
        assert tuple(round(v) for v in ann.bbox) == (10, 10, 90, 90)

    def test_yolo_without_yaml_falls_back_to_class_id(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        _save_img(tmp_path / "images" / "a.jpg")
        (tmp_path / "labels" / "a.txt").write_text("3 0.5 0.5 0.4 0.4\n")
        ds = load_dataset(str(tmp_path), format="yolo")
        assert ds.images[0].annotations[0].label == "class_3"

    def test_yolo_segmentation_row(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        _save_img(tmp_path / "images" / "a.jpg")
        # polygon row: cls + 4 points (8 coords)
        (tmp_path / "labels" / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
        ds = load_dataset(str(tmp_path), format="yolo")
        ann = ds.images[0].annotations[0]
        assert ann.segmentation is not None
        # bbox derived from polygon extent -> (10,10,50,50) on 100x100
        assert tuple(round(v) for v in ann.bbox) == (10, 10, 50, 50)

    def test_yolo_malformed_line_skipped(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        _save_img(tmp_path / "images" / "a.jpg")
        (tmp_path / "labels" / "a.txt").write_text("garbage not numbers here now\n0 0.5 0.5 0.2 0.2\n")
        ds = load_dataset(str(tmp_path), format="yolo")
        assert len(ds.images[0].annotations) == 1  # bad line skipped, good line kept


# --------------------------------------------------------------------------- #
# VOC robustness
# --------------------------------------------------------------------------- #
class TestVocRobustness:
    def _write_voc(self, tmp_path, size_xml: str):
        (tmp_path / "Annotations").mkdir()
        (tmp_path / "JPEGImages").mkdir()
        _save_img(tmp_path / "JPEGImages" / "a.jpg")
        xml = (
            f"<annotation><filename>a.jpg</filename>{size_xml}"
            "<object><name>cat</name><bndbox>"
            "<xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax>"
            "</bndbox></object></annotation>"
        )
        (tmp_path / "Annotations" / "a.xml").write_text(xml)

    def test_voc_missing_size_element(self, tmp_path):
        self._write_voc(tmp_path, "")  # no <size>
        ds = load_dataset(str(tmp_path), format="voc")
        assert len(ds.images) == 1
        assert ds.images[0].width is None

    def test_voc_empty_size_text_does_not_crash(self, tmp_path):
        self._write_voc(tmp_path, "<size><width></width><height></height></size>")
        ds = load_dataset(str(tmp_path), format="voc")  # must not raise ValueError
        assert len(ds.images) == 1
        assert ds.images[0].annotations[0].bbox == (10.0, 10.0, 50.0, 50.0)


# --------------------------------------------------------------------------- #
# COCO loader robustness
# --------------------------------------------------------------------------- #
class TestCocoRobustness:
    def test_malformed_records_skipped(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "images").mkdir()
        _save_img(tmp_path / "images" / "img_0.jpg")
        data = {
            "images": [
                {"id": 1, "file_name": "img_0.jpg", "width": 100, "height": 100},
                {"file_name": "no_id.jpg"},  # missing id -> skipped
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]},
                {"id": 2, "category_id": 1},           # missing image_id -> skipped
                {"id": 3, "image_id": 1},              # missing category_id -> skipped
            ],
            "categories": [{"id": 1, "name": "thing"}, {"name": "no_id_cat"}],
        }
        (tmp_path / "annotations" / "instances.json").write_text(json.dumps(data))
        ds = load_dataset(str(tmp_path), format="coco")  # must not raise KeyError
        assert len(ds.images) == 1
        assert ds.get_num_annotations() == 1

    def test_empty_json_graceful(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances.json").write_text("{ this is not json")
        ds = load_dataset(str(tmp_path), format="coco")
        assert len(ds.images) == 0
