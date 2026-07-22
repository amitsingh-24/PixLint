from __future__ import annotations

import random
import cv2
import numpy as np




def apply_cutmix(
    image1: np.ndarray,
    image2: np.ndarray,
    bboxes1: list[list[float]],
    bboxes2: list[list[float]],
    labels1: list[str],
    labels2: list[str],
    alpha: float = 1.0,
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))

    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    mixed_bboxes: list[list[float]] = []
    mixed_labels: list[str] = []

    for bbox, label in zip(bboxes1, labels1):
        bx1, by1, bx2, by2 = bbox
        if not _is_bbox_in_region(bx1, by1, bx2, by2, x1, y1, x2, y2):
            mixed_bboxes.append(list(bbox))
            mixed_labels.append(label)

    for bbox, label in zip(bboxes2, labels2):
        bx1, by1, bx2, by2 = bbox
        if _is_bbox_in_region(bx1, by1, bx2, by2, x1, y1, x2, y2):
            ox1 = max(bx1, x1)
            oy1 = max(by1, y1)
            ox2 = min(bx2, x2)
            oy2 = min(by2, y2)
            if ox2 > ox1 and oy2 > oy1:
                mixed_bboxes.append([ox1, oy1, ox2, oy2])
                mixed_labels.append(label)

    return mixed_image, mixed_bboxes, mixed_labels


def apply_mixup(
    image1: np.ndarray,
    image2: np.ndarray,
    bboxes1: list[list[float]],
    bboxes2: list[list[float]],
    labels1: list[str],
    labels2: list[str],
    alpha: float = 0.2,
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))

    lam = np.random.beta(alpha, alpha)
    mixed_image = (image1 * lam + image2 * (1.0 - lam)).astype(np.uint8)

    mixed_bboxes = list(bboxes1) + list(bboxes2)
    mixed_labels = list(labels1) + list(labels2)

    return mixed_image, mixed_bboxes, mixed_labels


def apply_mosaic(
    images: list[np.ndarray],
    bboxes_list: list[list[list[float]]],
    labels_list: list[list[str]],
    output_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    if len(images) != 4:
        images = images[:4]
        while len(images) < 4:
            images.append(images[-1].copy())
            bboxes_list.append(bboxes_list[-1])
            labels_list.append(labels_list[-1])

    mosaic_w, mosaic_h = output_size
    half_w, half_h = mosaic_w // 2, mosaic_h // 2

    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    all_bboxes: list[list[float]] = []
    all_labels: list[str] = []

    cx = np.random.randint(half_w // 2, half_w + half_w // 2)
    cy = np.random.randint(half_h // 2, half_h + half_h // 2)

    for idx, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
        h_img, w_img = img.shape[:2]
        if idx == 0:
            x1a, y1a, x2a, y2a = 0, 0, cx, cy
            x1b, y1b, x2b, y2b = w_img - cx, h_img - cy, w_img, h_img
        elif idx == 1:
            x1a, y1a, x2a, y2a = cx, 0, mosaic_w, cy
            x1b, y1b, x2b, y2b = 0, h_img - cy, cx, h_img
        elif idx == 2:
            x1a, y1a, x2a, y2a = 0, cy, cx, mosaic_h
            x1b, y1b, x2b, y2b = w_img - cx, 0, w_img, cy
        else:
            x1a, y1a, x2a, y2a = cx, cy, mosaic_w, mosaic_h
            x1b, y1b, x2b, y2b = 0, 0, cx, cy

        if x2a > x1a and y2a > y1a and x2b > x1b and y2b > y1b:
            img_crop = img[y1b:y2b, x1b:x2b]
            img_crop = cv2.resize(img_crop, (x2a - x1a, y2a - y1a))
            mosaic[y1a:y2a, x1a:x2a] = img_crop

            offset_x, offset_y = x1a - x1b, y1a - y1b
            for bbox in bboxes:
                bx1, by1, bx2, by2 = bbox
                nx1 = bx1 + offset_x
                ny1 = by1 + offset_y
                nx2 = bx2 + offset_x
                ny2 = by2 + offset_y
                if nx2 > 0 and ny2 > 0 and nx1 < mosaic_w and ny1 < mosaic_h:
                    nx1 = max(0, nx1)
                    ny1 = max(0, ny1)
                    nx2 = min(mosaic_w, nx2)
                    ny2 = min(mosaic_h, ny2)
                    if nx2 > nx1 and ny2 > ny1:
                        all_bboxes.append([nx1, ny1, nx2, ny2])
                        all_labels.append(labels[0] if labels else "")

    return mosaic, all_bboxes, all_labels


def apply_copypaste(
    image: np.ndarray,
    source_image: np.ndarray,
    source_bboxes: list[list[float]],
    source_labels: list[str],
    target_bboxes: list[list[float]],
    target_labels: list[str],
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    h, w = image.shape[:2]
    src_h, src_w = source_image.shape[:2]
    result = image.copy()
    all_bboxes = list(target_bboxes)
    all_labels = list(target_labels)

    for bbox, label in zip(source_bboxes, source_labels):
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(src_w, bx2), min(src_h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue

        obj = source_image[by1:by2, bx1:bx2]
        obj_h, obj_w = obj.shape[:2]

        scale = random.uniform(0.5, 1.5)  # nosec B311
        new_w = min(w - 10, int(obj_w * scale))
        new_h = min(h - 10, int(obj_h * scale))
        if new_w < 10 or new_h < 10:
            continue

        obj = cv2.resize(obj, (new_w, new_h))
        tx = random.randint(0, max(1, w - new_w - 1))  # nosec B311
        ty = random.randint(0, max(1, h - new_h - 1))  # nosec B311

        try:
            result[ty : ty + new_h, tx : tx + new_w] = cv2.addWeighted(
                result[ty : ty + new_h, tx : tx + new_w], 0.3, obj, 0.7, 0
            )
        except Exception:
            continue  # nosec B112

        all_bboxes.append([float(tx), float(ty), float(tx + new_w), float(ty + new_h)])
        all_labels.append(label)

    return result, all_bboxes, all_labels


def _is_bbox_in_region(
    x1: float, y1: float, x2: float, y2: float,
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> bool:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
