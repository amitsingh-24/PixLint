# Changelog

All notable changes to PixLint are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-07-26

### Fixed
- **COCO bounding boxes were stored in the wrong convention.** COCO's native
  `[x, y, width, height]` boxes were kept verbatim while every other loader and
  every consumer treats `bbox` as `(x1, y1, x2, y2)`. This corrupted areas,
  spatial statistics, and COCO exports, and made COCO→COCO round-trips lossy.
  COCO boxes are now normalized to `(x1, y1, x2, y2)` at load time.
- **COCO loader no longer crashes on real-world files.** Records missing `id`,
  `image_id`, or `category_id` (and categories missing `name`) are skipped
  instead of raising `KeyError`.
- **VOC loader no longer crashes** on empty or non-integer `<size>`/`<bndbox>`
  values (e.g. `<width></width>`, `500.0`) — these are parsed safely.
- **YOLO loader** now skips malformed label lines instead of aborting the whole
  dataset, and correctly handles YOLO-segmentation (polygon) rows by deriving a
  bounding box from the polygon extent and preserving the polygon.
- **KITTI loader** now excludes `DontCare`/`Misc` ignore-regions from class
  lists and statistics.
- **COCO JSON selection is deterministic** — annotation files under
  `annotations/` are preferred and sorted, instead of picking an arbitrary
  `*.json` (which could silently load the wrong split).
- The `dataset://{id}/quality/scores` resource no longer raises a `TypeError`
  when a dataset contains an undecodable image.
- `analyze_distribution`'s `cooccurrence` analysis is no longer silently
  skipped (name mismatch with the engine); friendly aliases are normalized.
- Removed the no-op `empty_images` integrity check (it is covered by
  `missing_labels`).
- Segmentation preview is guarded against malformed (odd-length) polygons.

### Performance
- Duplicate detection no longer rebuilds an id→path map on every image lookup
  (previously O(n³) inside the pairwise loops).
- Quality analysis converts each image to grayscale once and reuses it across
  the blur/exposure/noise/contrast metrics.
- YOLO/KITTI loaders index images by stem once instead of rescanning the image
  directory per label file (O(n²)→O(n) on load).

### Types & tooling
- `mypy` now passes cleanly on `src/` with no `# type: ignore` added to
  first-party code; a `[tool.mypy]` config scopes checks and silences only the
  optional heavy dependencies. CI now enforces the type check.
- Added real-world-data regression tests: COCO bbox geometry, crowd/RLE
  segmentation, the previously-untested KITTI loader, COCO→COCO round-trip
  integrity, and malformed-input handling for every loader.

## [1.0.3] - 2026-07-25

### Fixed
- COCO RLE crowd annotations (segmentation as a `{"counts", "size"}` dict) no
  longer crash dataset loading.

## [1.0.2] - 2026-07-24

### Added
- MCP registry ownership metadata; published to the official MCP registry.

## [1.0.1]

### Fixed
- Corrected four resource handlers using stale schema attributes.
- `diversity_sampling` no longer errors on datasets smaller than the requested
  sample size.

## [1.0.0]

- Initial public release: 67 tools, 23 resources, 13 prompts for loading,
  analyzing, curating, splitting, augmenting, converting, and exporting
  computer-vision datasets over MCP.
