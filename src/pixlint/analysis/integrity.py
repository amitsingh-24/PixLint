from __future__ import annotations

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image
from pixlint.utils.schemas import IntegrityIssue, IntegrityReport


def check_integrity(
    dataset: CVDataset,
    checks: list[str] | None = None,
    strict: bool = False,
) -> IntegrityReport:
    if checks is None:
        checks = ["corrupt", "missing_labels"]

    issues: list[IntegrityIssue] = []
    summary: dict[str, int] = {}

    if "corrupt" in checks:
        count = 0
        for img in dataset.images:
            image = read_image(img.path)
            if image is None:
                issues.append(
                    IntegrityIssue(
                        image_id=img.image_id,
                        issue_type="corrupt",
                        description=f"Cannot decode image: {img.path}",
                        severity="error",
                    )
                )
                count += 1
        summary["corrupt_images"] = count

    if "missing_labels" in checks:
        count = 0
        for img in dataset.images:
            if len(img.annotations) == 0:
                issues.append(
                    IntegrityIssue(
                        image_id=img.image_id,
                        issue_type="missing_labels",
                        description=f"No annotations found for image: {img.image_id}",
                        severity="warning",
                    )
                )
                count += 1
        summary["missing_labels"] = count

    if "annotation_bounds" in checks:
        count = 0
        for img in dataset.images:
            if img.width is None or img.height is None:
                continue
            for ann in img.annotations:
                if ann.bbox is not None:
                    x1, y1, x2, y2 = ann.bbox
                    if x2 <= x1 or y2 <= y1:
                        issues.append(
                            IntegrityIssue(
                                image_id=img.image_id,
                                issue_type="degenerate_bbox",
                                description=f"Degenerate bounding box ({x1},{y1},{x2},{y2})",
                                severity="error",
                            )
                        )
                        count += 1
                    elif x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height:
                        issues.append(
                            IntegrityIssue(
                                image_id=img.image_id,
                                issue_type="out_of_bounds",
                                description=f"Bounding box outside image bounds ({x1},{y1},{x2},{y2}) vs ({img.width},{img.height})",
                                severity="warning",
                            )
                        )
                        count += 1
        summary["annotation_bounds"] = count

    return IntegrityReport(
        dataset_id=dataset.dataset_id,
        total_issues=len(issues),
        issues=issues,
        summary=summary,
    )
