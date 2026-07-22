from __future__ import annotations

import numpy as np

from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.quality import analyze_quality
from pixlint.analysis.statistics import compute_statistics
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import HealthScore, HealthScoreBreakdown


def dataset_health_score(dataset: CVDataset) -> HealthScore:
    stats = compute_statistics(dataset)
    dist = analyze_distribution(dataset)
    quality = analyze_quality(dataset)
    integrity = check_integrity(dataset)

    class_balance = _compute_class_balance(dist)
    image_quality = quality.average_overall
    integrity_score = _compute_integrity_score(integrity, stats.num_images)
    annotation_consistency = _compute_annotation_consistency(stats)
    diversity = _compute_diversity(class_balance, stats)

    weights = {
        "class_balance": 0.25,
        "image_quality": 0.25,
        "integrity": 0.20,
        "annotation_consistency": 0.15,
        "diversity": 0.15,
    }

    overall = (
        class_balance * weights["class_balance"]
        + image_quality * weights["image_quality"]
        + integrity_score * weights["integrity"]
        + annotation_consistency * weights["annotation_consistency"]
        + diversity * weights["diversity"]
    )

    recommendations: list[str] = []

    if class_balance < 60:
        if dist.imbalance_ratio and dist.imbalance_ratio > 5:
            recommendations.append(
                f"High class imbalance detected (ratio: {dist.imbalance_ratio}). "
                "Consider adding more samples for underrepresented classes."
            )

    if image_quality < 60:
        flagged_pct = quality.summary.get("flagged_percentage", 0)
        if flagged_pct > 10:
            recommendations.append(
                f"{quality.flagged_images} images ({flagged_pct}%) flagged for quality issues. "
                "Consider replacing low-quality images."
            )

    if integrity_score < 80:
        issue_types = integrity.summary
        for issue_type, count in issue_types.items():
            if count > 0:
                recommendations.append(
                    f"{count} {issue_type.replace('_', ' ')} detected. Run check_integrity() for details."
                )

    if annotation_consistency < 60:
        recommendations.append(
            "Low annotation consistency detected. Review annotation quality and completeness."
        )

    if diversity < 50:
        recommendations.append(
            "Low dataset diversity. Consider adding more varied samples or using augmentation."
        )

    if stats.num_images < 100:
        recommendations.append(
            f"Small dataset ({stats.num_images} images). Consider collecting more data for better model performance."
        )

    return HealthScore(
        dataset_id=dataset.dataset_id,
        overall=round(overall, 1),
        breakdown=HealthScoreBreakdown(
            class_balance=round(class_balance, 1),
            image_quality=round(image_quality, 1),
            integrity=round(integrity_score, 1),
            annotation_consistency=round(annotation_consistency, 1),
            diversity=round(diversity, 1),
        ),
        recommendations=recommendations[:5],
    )


def _compute_class_balance(dist) -> float:
    if not dist.class_distribution:
        return 100.0

    counts = np.array([d.count for d in dist.class_distribution])
    if len(counts) <= 1:
        return 100.0

    total = counts.sum()
    if total == 0:
        return 100.0

    proportions = counts / total
    ideal = 1.0 / len(counts)
    kl_div = (proportions * np.log(proportions / ideal + 1e-10)).sum()
    max_kl = np.log(len(counts))
    balance = max(0, 100 * (1 - kl_div / max_kl))
    return balance


def _compute_integrity_score(integrity, num_images: int) -> float:
    total_issues = integrity.total_issues
    if num_images == 0:
        return 100.0
    issue_ratio = total_issues / max(num_images, 1)
    score = max(0, 100 * (1 - issue_ratio))
    return score


def _compute_annotation_consistency(stats) -> float:
    if stats.num_annotations == 0 or stats.num_images == 0:
        return 50.0

    avg_anns = stats.avg_annotations_per_image
    if avg_anns < 0.5:
        return 40.0
    elif avg_anns < 1.0:
        return 60.0
    elif avg_anns < 3.0:
        return 85.0
    else:
        return 95.0


def _compute_diversity(class_balance: float, stats) -> float:
    num_classes = stats.num_classes
    if num_classes == 0:
        return 30.0
    diversity = min(100, num_classes * 10 + class_balance * 0.5)
    return diversity
