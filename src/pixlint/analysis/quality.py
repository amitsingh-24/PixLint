from __future__ import annotations


import cv2
import numpy as np

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image
from pixlint.utils.schemas import QualityMetrics, QualityReport


def analyze_quality(
    dataset: CVDataset,
    metrics: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    per_class: bool = True,
) -> QualityReport:
    if metrics is None:
        metrics = ["blur", "exposure", "noise", "contrast", "resolution", "color"]

    results: list[QualityMetrics] = []
    flagged_count = 0

    for img in dataset.images:
        image = read_image(img.path)
        if image is None:
            results.append(
                QualityMetrics(
                    image_id=img.image_id,
                    issues=["unable to read image"],
                )
            )
            flagged_count += 1
            continue

        scores: dict[str, float] = {}
        issues: list[str] = []

        if "blur" in metrics:
            blur_score, blur_label = _analyze_blur(image)
            scores["blur"] = blur_score
            if blur_label == "blurry":
                issues.append("blurry")

        if "exposure" in metrics:
            exp_score, exp_label = _analyze_exposure(image)
            scores["exposure"] = exp_score
            if exp_label != "good":
                issues.append(exp_label)

        if "noise" in metrics:
            noise_score, noise_label = _analyze_noise(image)
            scores["noise"] = noise_score
            if noise_label == "noisy":
                issues.append("noisy")

        if "contrast" in metrics:
            contrast_score, contrast_label = _analyze_contrast(image)
            scores["contrast"] = contrast_score
            if contrast_label == "low_contrast":
                issues.append("low_contrast")

        if "resolution" in metrics:
            res_score, res_label = _analyze_resolution(image, img.width, img.height)
            scores["resolution"] = res_score
            if res_label == "low_resolution":
                issues.append("low_resolution")

        if "color" in metrics:
            color_score, color_label = _analyze_color(image)
            scores["color"] = color_score
            if color_label == "poor_color":
                issues.append("poor_color")

        overall = _compute_overall(scores) if scores else None

        if issues:
            flagged_count += 1

        results.append(
            QualityMetrics(
                image_id=img.image_id,
                blur_score=scores.get("blur"),
                blur_label=_score_to_label(scores.get("blur"), "blur"),
                exposure_score=scores.get("exposure"),
                exposure_label=_score_to_label(scores.get("exposure"), "exposure"),
                noise_score=scores.get("noise"),
                contrast_score=scores.get("contrast"),
                resolution_score=scores.get("resolution"),
                color_score=scores.get("color"),
                overall_score=overall,
                issues=issues,
            )
        )

    scores_list = [r.overall_score for r in results if r.overall_score is not None]
    avg_overall = sum(scores_list) / len(scores_list) if scores_list else 0.0

    return QualityReport(
        dataset_id=dataset.dataset_id,
        per_image=results,
        average_overall=round(avg_overall, 2),
        flagged_images=flagged_count,
        summary={
            "metrics_used": metrics,
            "total_images": len(dataset.images),
            "flagged_percentage": round(flagged_count / max(len(dataset.images), 1) * 100, 2),
        },
    )


def _analyze_blur(image: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var > 100:
        label = "sharp"
    elif laplacian_var > 30:
        label = "acceptable"
    else:
        label = "blurry"

    normalized_score = min(laplacian_var / 500.0, 1.0) * 100
    return round(normalized_score, 2), label


def _analyze_exposure(image: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()

    if mean_brightness < 40:
        label = "underexposed"
    elif mean_brightness > 220:
        label = "overexposed"
    else:
        label = "good"

    if mean_brightness < 128:
        normalized_score = mean_brightness / 128.0 * 100
    else:
        normalized_score = (255.0 - mean_brightness) / 127.0 * 100

    return round(normalized_score, 2), label


def _analyze_noise(image: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    low_radius = min(h, w) // 8
    mask = np.zeros_like(magnitude)
    cv2.circle(mask, (center_w, center_h), low_radius, 1, -1)
    low_freq = (magnitude * mask).sum()
    high_freq = magnitude.sum() - low_freq
    noise_ratio = high_freq / max(low_freq, 1)

    if noise_ratio > 0.5:
        label = "noisy"
    elif noise_ratio > 0.2:
        label = "moderate"
    else:
        label = "clean"

    normalized = max(0, min(100, (1.0 - noise_ratio) * 100))
    return round(normalized, 2), label


def _analyze_contrast(image: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rms_contrast = gray.std()
    normalized = min(rms_contrast / 64.0, 1.0) * 100

    if normalized < 20:
        label = "low_contrast"
    elif normalized < 40:
        label = "moderate"
    else:
        label = "good"

    return round(normalized, 2), label


def _analyze_resolution(
    image: np.ndarray, width: int | None, height: int | None
) -> tuple[float, str]:
    h, w = image.shape[:2]
    actual_w = width or w
    actual_h = height or h
    megapixels = (actual_w * actual_h) / 1_000_000

    if megapixels < 0.1:
        label = "low_resolution"
        score = megapixels / 0.1 * 100
    elif megapixels < 0.5:
        label = "acceptable"
        score = 50 + (megapixels / 0.5) * 50
    else:
        label = "good"
        score = min(100, megapixels / 2.0 * 100)

    return round(min(score, 100), 2), label


def _analyze_color(image: np.ndarray) -> tuple[float, str]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    mean_sat = saturation.mean()
    sat_std = saturation.std()

    if mean_sat < 20:
        label = "poor_color"
        score = mean_sat / 20.0 * 100
    elif mean_sat < 50:
        label = "moderate"
        score = 50 + (mean_sat / 100.0) * 50
    else:
        label = "vibrant"
        score = min(100, mean_sat / 128.0 * 100)

    if sat_std < 15:
        score *= 0.85

    return round(min(score, 100), 2), label


def _score_to_label(score: float | None, metric: str) -> str | None:
    if score is None:
        return None
    if score >= 70:
        return "good"
    elif score >= 40:
        return "acceptable"
    return "poor"


def _compute_overall(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 2)
