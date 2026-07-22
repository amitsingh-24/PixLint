from __future__ import annotations

import random

import numpy as np

_AUGMENTATIONS_POOL = [
    "autocontrast", "equalize", "posterize", "rotate", "solarize",
    "shear_x", "shear_y", "translate_x", "translate_y",
    "color", "brightness", "contrast", "sharpness",
]


def _apply_op(
    image: np.ndarray, op_name: str, magnitude: float
) -> np.ndarray:
    try:
        import albumentations as A
    except ImportError:
        return image

    mag = max(0.0, min(1.0, magnitude))

    ops = {
        "autocontrast": lambda: A.Equalize(p=1.0)(image=image)["image"],
        "equalize": lambda: A.Equalize(p=1.0)(image=image)["image"],
        "posterize": lambda: _posterize(image, int(8 - mag * 7)),
        "rotate": lambda: A.Rotate(limit=int(mag * 30), p=1.0)(image=image)["image"],
        "solarize": lambda: _solarize(image, int(256 * (1 - mag))),
        "shear_x": lambda: A.Affine(shear={"x": mag * 30}, p=1.0)(image=image)["image"],
        "shear_y": lambda: A.Affine(shear={"y": mag * 30}, p=1.0)(image=image)["image"],
        "translate_x": lambda: A.Affine(translate_percent={"x": mag * 0.3}, p=1.0)(image=image)["image"],
        "translate_y": lambda: A.Affine(translate_percent={"y": mag * 0.3}, p=1.0)(image=image)["image"],
        "color": lambda: A.ColorJitter(
            brightness=0, contrast=0, saturation=mag * 0.9, hue=0, p=1.0
        )(image=image)["image"],
        "brightness": lambda: A.RandomBrightnessContrast(
            brightness_limit=mag * 0.5, contrast_limit=0, p=1.0
        )(image=image)["image"],
        "contrast": lambda: A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=mag * 0.5, p=1.0
        )(image=image)["image"],
        "sharpness": lambda: A.Sharpen(alpha=mag, lightness=mag, p=1.0)(image=image)["image"],
    }

    fn = ops.get(op_name)
    if fn is None:
        return image
    try:
        return fn()
    except Exception:
        return image


def _posterize(image: np.ndarray, bits: int) -> np.ndarray:
    bits = max(1, min(8, bits))
    shift = 8 - bits
    return (image.astype(np.int32) >> shift << shift).astype(np.uint8)


def _solarize(image: np.ndarray, threshold: int) -> np.ndarray:
    threshold = max(0, min(255, threshold))
    result = image.copy()
    mask = result >= threshold
    result[mask] = 255 - result[mask]
    return result


def apply_randaugment(
    image: np.ndarray,
    n_transforms: int = 2,
    magnitude: float = 9.0,
) -> np.ndarray:
    mag = magnitude / 30.0
    ops = random.choices(_AUGMENTATIONS_POOL, k=n_transforms)  # nosec B311
    result = image.copy()
    for op_name in ops:
        result = _apply_op(result, op_name, mag)
    return result


def apply_trivialaugment(image: np.ndarray) -> np.ndarray:
    op_name = random.choice(_AUGMENTATIONS_POOL)  # nosec B311
    magnitude = random.uniform(0.0, 1.0)  # nosec B311
    return _apply_op(image, op_name, magnitude)


def apply_autoaugment(
    image: np.ndarray,
    policy: str = "imagenet",
) -> np.ndarray:
    if policy == "imagenet":
        sub_policies = [
            [("autocontrast", 0.5, 0.5), ("equalize", 0.5, 0.5)],
            [("solarize", 0.5, 0.5), ("color", 0.5, 0.5)],
            [("brightness", 0.5, 0.5), ("contrast", 0.5, 0.5)],
            [("rotate", 0.5, 0.5), ("shear_x", 0.5, 0.5)],
            [("translate_x", 0.5, 0.5), ("translate_y", 0.5, 0.5)],
        ]
    else:
        sub_policies = [[("brightness", 0.5, 0.3), ("contrast", 0.5, 0.3)]]

    policy_ops = random.choice(sub_policies)  # nosec B311
    result = image.copy()
    for op_name, prob, mag in policy_ops:
        if random.random() < prob:  # nosec B311
            result = _apply_op(result, op_name, mag)
    return result
