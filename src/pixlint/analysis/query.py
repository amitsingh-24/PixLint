"""Natural-language / predicate dataset query engine.

The agent translates a user's English request ("blurry images with a person on
the left side, at least 2 objects") into a structured predicate dict; this
engine compiles it into a filter over combined signals — class membership,
geometry, object count, image size, quality, spatial position, and (optionally)
CLIP text similarity — and returns matching image IDs that pipe straight into
the curation tools. No competitor exposes this to an agent.
"""
from __future__ import annotations

from typing import Any

from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import QueryResult

# Supported quality issue tokens (match analyze_quality issue strings loosely).
_QUALITY_ISSUE_ALIASES = {
    "blurry": "blur", "blur": "blur",
    "dark": "exposure", "underexposed": "exposure",
    "bright": "exposure", "overexposed": "exposure",
    "noisy": "noise", "noise": "noise",
    "low_contrast": "contrast", "low_resolution": "resolution",
}

_POSITIONS = {"left", "right", "top", "bottom", "center"}


def _box_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _in_region(cx: float, cy: float, w: float, h: float, region: str) -> bool:
    if not w or not h:
        return False
    fx, fy = cx / w, cy / h
    if region == "left":
        return fx < 0.4
    if region == "right":
        return fx > 0.6
    if region == "top":
        return fy < 0.4
    if region == "bottom":
        return fy > 0.6
    if region == "center":
        return 0.33 <= fx <= 0.67 and 0.33 <= fy <= 0.67
    return False


def query_dataset(dataset: CVDataset, filters: dict[str, Any]) -> QueryResult:
    """Return image IDs matching ALL provided predicates.

    Recognized predicate keys (all optional, ANDed):
      classes, exclude_classes: list[str]
      min_annotations, max_annotations: int
      min_width, max_width, min_height, max_height: int
      min_area, max_area: float           (any box within range)
      quality_issues: list[str]           (e.g. ["blurry","dark"])
      min_quality, max_quality: float     (overall quality score 0..100)
      position: str                       (left|right|top|bottom|center — any box)
      position_class: str                 (restrict `position` to this class)
      similar_to_text: str                (CLIP semantic similarity; needs [clip])
      limit: int
    """
    applied: dict[str, Any] = {}
    notes: list[str] = []
    images = list(dataset.images)

    # Precompute quality only if a quality predicate is present.
    quality_by_id: dict[str, Any] = {}
    if any(k in filters for k in ("quality_issues", "min_quality", "max_quality")):
        from pixlint.analysis.quality import analyze_quality
        report = analyze_quality(dataset)
        quality_by_id = {q.image_id: q for q in report.per_image}

    # Optional semantic pre-filter via CLIP text similarity.
    semantic_ids: set[str] | None = None
    if filters.get("similar_to_text"):
        from pixlint.analysis.embeddings import semantic_search
        top_k = int(filters.get("limit", 100)) or 100
        results = semantic_search(dataset, query=str(filters["similar_to_text"]), top_k=top_k)
        semantic_ids = {r.image_id for r in results}
        applied["similar_to_text"] = filters["similar_to_text"]
        if not semantic_ids:
            notes.append("similar_to_text returned no matches (is the [clip] extra installed?).")

    classes = set(filters["classes"]) if filters.get("classes") else None
    exclude = set(filters["exclude_classes"]) if filters.get("exclude_classes") else None
    region = filters.get("position")
    if region and region not in _POSITIONS:
        notes.append(f"Unknown position '{region}' (use one of {sorted(_POSITIONS)}); ignored.")
        region = None
    position_class = filters.get("position_class")

    matched: list[str] = []
    for img in images:
        if semantic_ids is not None and img.image_id not in semantic_ids:
            continue

        labels = [ann.label for ann in img.annotations]
        n_ann = len(img.annotations)

        if classes is not None and not (set(labels) & classes):
            continue
        if exclude is not None and (set(labels) & exclude):
            continue
        if "min_annotations" in filters and n_ann < int(filters["min_annotations"]):
            continue
        if "max_annotations" in filters and n_ann > int(filters["max_annotations"]):
            continue

        w, h = img.width, img.height
        if "min_width" in filters and (not w or w < filters["min_width"]):
            continue
        if "max_width" in filters and (w and w > filters["max_width"]):
            continue
        if "min_height" in filters and (not h or h < filters["min_height"]):
            continue
        if "max_height" in filters and (h and h > filters["max_height"]):
            continue

        if "min_area" in filters or "max_area" in filters:
            areas = []
            for ann in img.annotations:
                if ann.bbox:
                    x1, y1, x2, y2 = ann.bbox
                    areas.append(abs(x2 - x1) * abs(y2 - y1))
            ok = any(
                (("min_area" not in filters or a >= filters["min_area"]) and
                 ("max_area" not in filters or a <= filters["max_area"]))
                for a in areas
            )
            if not ok:
                continue

        if quality_by_id:
            q = quality_by_id.get(img.image_id)
            if q is not None:
                if "min_quality" in filters and (q.overall_score is None or q.overall_score < filters["min_quality"]):
                    continue
                if "max_quality" in filters and (q.overall_score is None or q.overall_score > filters["max_quality"]):
                    continue
                if filters.get("quality_issues"):
                    wanted = {_QUALITY_ISSUE_ALIASES.get(str(x).lower(), str(x).lower())
                              for x in filters["quality_issues"]}
                    issue_text = " ".join(q.issues).lower()
                    if not any(w_ in issue_text for w_ in wanted):
                        continue

        if region:
            hit = False
            for ann in img.annotations:
                if not ann.bbox:
                    continue
                if position_class and ann.label != position_class:
                    continue
                cx, cy = _box_center(ann.bbox)
                if _in_region(cx, cy, w or 0, h or 0, region):
                    hit = True
                    break
            if not hit:
                continue

        matched.append(img.image_id)

    for key in ("classes", "exclude_classes", "min_annotations", "max_annotations",
                "min_width", "max_width", "min_height", "max_height", "min_area",
                "max_area", "quality_issues", "min_quality", "max_quality",
                "position", "position_class"):
        if key in filters:
            applied[key] = filters[key]

    limit = filters.get("limit")
    if limit:
        matched = matched[: int(limit)]

    return QueryResult(
        dataset_id=dataset.dataset_id,
        num_matched=len(matched),
        total_images=len(images),
        matched_image_ids=matched,
        applied_filters=applied,
        notes=notes,
    )
