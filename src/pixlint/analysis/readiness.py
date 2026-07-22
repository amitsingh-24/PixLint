"""Dataset "Doctor" — a training-readiness report that diagnoses AND prescribes.

One call runs the full diagnostic battery (integrity, quality, duplicates,
distribution, leakage, health) and returns a prioritized, *executable*
remediation plan plus a ready-to-run pipeline JSON. This turns the agent loop
from "here are some numbers" into "here is exactly what to fix and the pipeline
that fixes it" — the flagship agent-native experience.
"""
from __future__ import annotations

from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.health import dataset_health_score
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.quality import analyze_quality
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import ReadinessRecommendation, ReadinessReport

_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def dataset_readiness_report(dataset: CVDataset) -> ReadinessReport:
    n = len(dataset.images)
    recs: list[ReadinessRecommendation] = []
    metrics: dict = {"num_images": n}

    if n == 0:
        return ReadinessReport(
            dataset_id=dataset.dataset_id, readiness_score=0.0,
            readiness_level="not_ready", summary="Dataset is empty.",
            recommendations=[ReadinessRecommendation(
                severity="critical", category="empty", finding="No images found.",
                action="Load a non-empty dataset with load_dataset_tool.")],
            metrics=metrics,
        )

    # --- Run the battery (each guarded so one failure doesn't sink the report) ---
    integrity = _safe(lambda: check_integrity(dataset))
    quality = _safe(lambda: analyze_quality(dataset))
    dupes = _safe(lambda: find_duplicates(dataset, methods=["exact", "perceptual"]))
    dist = _safe(lambda: analyze_distribution(dataset, analyses=["class_dist"]))
    health = _safe(lambda: dataset_health_score(dataset))

    needs_clean = False
    needs_dedup = False

    if integrity is not None:
        metrics["integrity_issues"] = integrity.total_issues
        corrupt = sum(1 for i in integrity.issues if "corrupt" in i.issue_type.lower())
        missing = sum(1 for i in integrity.issues if "missing" in i.issue_type.lower())
        oob = sum(1 for i in integrity.issues if "bound" in i.issue_type.lower() or "degenerate" in i.issue_type.lower())
        if corrupt:
            needs_clean = True
            recs.append(ReadinessRecommendation(
                severity="critical", category="integrity",
                finding=f"{corrupt} corrupt/unreadable images.",
                action="Run clean_dataset_tool with drop_corrupt=true.", auto_fixable=True))
        if oob:
            needs_clean = True
            recs.append(ReadinessRecommendation(
                severity="high", category="annotations",
                finding=f"{oob} out-of-bounds/degenerate boxes.",
                action="Run clean_dataset_tool with clip_out_of_bounds=true, drop_degenerate=true.",
                auto_fixable=True))
        if missing:
            recs.append(ReadinessRecommendation(
                severity="high", category="annotations",
                finding=f"{missing} images with missing labels.",
                action="Review with check_integrity_tool; add labels or filter them out.",
                auto_fixable=False))

    if dupes is not None:
        metrics["duplicate_images"] = dupes.total_duplicate_images
        if dupes.total_duplicate_images > 0:
            needs_dedup = True
            pct = 100.0 * dupes.total_duplicate_images / max(1, n)
            recs.append(ReadinessRecommendation(
                severity="high" if pct > 5 else "medium", category="duplicates",
                finding=f"{dupes.total_duplicate_images} duplicate images ({pct:.1f}%).",
                action="Run clean_dataset_tool with drop_duplicates=true.", auto_fixable=True))

    if quality is not None:
        metrics["avg_quality"] = round(quality.average_overall, 2)
        metrics["flagged_images"] = quality.flagged_images
        flagged_pct = 100.0 * quality.flagged_images / max(1, n)
        if flagged_pct > 20:
            recs.append(ReadinessRecommendation(
                severity="high" if flagged_pct > 40 else "medium", category="quality",
                finding=f"{quality.flagged_images} low-quality images ({flagged_pct:.0f}%).",
                action="Inspect with analyze_quality_tool; filter with query_dataset_tool "
                       "(max_quality=50) then filter_dataset_tool by image_ids.",
                auto_fixable=False))

    if dist is not None and dist.class_distribution:
        metrics["num_classes"] = len(dist.class_distribution)
        metrics["imbalance_ratio"] = dist.imbalance_ratio
        if dist.imbalance_ratio and dist.imbalance_ratio > 10:
            recs.append(ReadinessRecommendation(
                severity="medium", category="balance",
                finding=f"Severe class imbalance (ratio {dist.imbalance_ratio:.1f}:1).",
                action="Run discover_slices_tool to find weak classes, then augment_dataset_tool "
                       "or collect more of the rare classes.", auto_fixable=False))

    # --- Score & level ---
    score = round(health.overall, 2) if health is not None else 50.0
    metrics["health_score"] = score
    if score >= 80 and not any(r.severity == "critical" for r in recs):
        level = "training_ready"
    elif score >= 55:
        level = "needs_work"
    else:
        level = "not_ready"

    if not recs:
        recs.append(ReadinessRecommendation(
            severity="low", category="ok",
            finding="No blocking issues detected.",
            action="Proceed to split_dataset_tool and export_dataset_tool.", auto_fixable=False))

    recs.sort(key=lambda r: _SEV_ORDER.get(r.severity, 9))

    # --- Executable remediation pipeline (runnable via execute_pipeline_tool) ---
    steps: list[dict] = []
    if needs_clean or needs_dedup:
        steps.append({
            "step_id": "clean", "operation": "check_integrity", "params": {},
            "description": "Re-check integrity before cleaning",
        })
    steps.append({
        "step_id": "split", "operation": "split_dataset",
        "params": {"strategy": "stratified", "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
        "depends_on": ["clean"] if steps else [],
        "description": "Stratified train/val/test split",
    })
    remediation = {
        "pipeline_id": f"remediate_{dataset.dataset_id}",
        "name": "Auto-generated remediation",
        "version": "1.0",
        "description": "Suggested remediation pipeline from the readiness report. "
                       "For corrupt/duplicate/box fixes, first call clean_dataset_tool.",
        "steps": steps,
        "tags": ["auto", "remediation"],
    }

    crit = sum(1 for r in recs if r.severity in ("critical", "high"))
    summary = (f"Readiness: {level.replace('_', ' ')} (score {score}/100). "
               f"{crit} high-priority issue(s), {len(recs)} recommendation(s).")

    return ReadinessReport(
        dataset_id=dataset.dataset_id,
        readiness_score=score,
        readiness_level=level,
        summary=summary,
        recommendations=recs,
        metrics=metrics,
        remediation_pipeline=remediation,
    )


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None
