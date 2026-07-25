"""Streamlit GUI dashboard for PixLint.

Usage:
    streamlit run src/pixlint/dashboard.py
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import cv2
import streamlit as st

from pixlint.analysis.captioning import enrich_metadata, generate_captions
from pixlint.analysis.distribution import analyze_distribution
from pixlint.analysis.duplicates import find_duplicates
from pixlint.analysis.health import dataset_health_score
from pixlint.analysis.integrity import check_integrity
from pixlint.analysis.outliers import detect_outliers
from pixlint.analysis.quality import analyze_quality
from pixlint.augmentation.pipeline import augment_dataset, preview_augmentation
from pixlint.core.loader import CVDataset, load_dataset
from pixlint.core.metadata import register_dataset
from pixlint.splitting.splitter import split_dataset
from pixlint.transformation.resize import resize_dataset

st.set_page_config(page_title="PixLint", page_icon="📷", layout="wide")

if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "current_dataset_id" not in st.session_state:
    st.session_state.current_dataset_id = None


def _load(path: str, fmt: str = "auto", name: str | None = None) -> str:
    ds = load_dataset(path, format=fmt, name=name)
    ds_id = register_dataset(ds.to_info())
    st.session_state.datasets[ds_id] = ds
    st.session_state.current_dataset_id = ds_id
    return ds_id


def _get_ds() -> CVDataset | None:
    ds_id = st.session_state.current_dataset_id
    if ds_id and ds_id in st.session_state.datasets:
        return st.session_state.datasets[ds_id]
    return None


def _render_image_grid(ds: CVDataset, n: int = 12):
    cols = st.columns(4)
    for i, img in enumerate(ds.images[:n]):
        with cols[i % 4]:
            image = cv2.imread(img.path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption=img.image_id[:20], use_container_width=True)
            else:
                st.caption(f"{img.image_id} (unreadable)")


st.sidebar.title("📷 PixLint")
st.sidebar.markdown("---")

with st.sidebar.expander("📂 Load Dataset", expanded=True):
    load_option = st.radio("Source", ["Local Path", "Sample COCO", "Sample VOC"])
    if load_option == "Local Path":
        path = st.text_input("Dataset path", placeholder="/path/to/dataset")
        fmt = st.selectbox("Format", ["auto", "folder", "coco", "voc", "yolo", "kitti"])
        name = st.text_input("Name (optional)")
        if st.button("Load") and path:
            try:
                _load(path, fmt=fmt, name=name or None)
                st.success(f"Loaded: {path}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

dataset_ids = list(st.session_state.datasets.keys())
if dataset_ids:
    selected = st.sidebar.selectbox(
        "Active Dataset", dataset_ids,
        index=dataset_ids.index(st.session_state.current_dataset_id) if st.session_state.current_dataset_id in dataset_ids else 0,
    )
    if selected != st.session_state.current_dataset_id:
        st.session_state.current_dataset_id = selected
        st.rerun()

    ds = _get_ds()
    if ds:
        info = ds.to_info()
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{info.name}**")
        st.sidebar.markdown(f"Images: {info.num_images} | Anns: {info.num_annotations}")
        st.sidebar.markdown(f"Classes: {len(info.class_names)} | Format: {info.format.value}")

st.sidebar.markdown("---")
st.sidebar.markdown("PixLint v1.1.0")

st.title("📷 PixLint Dashboard")

ds = _get_ds()
if ds is None:
    st.info("Load a dataset to get started. Use the sidebar to load from a local path.")
    st.stop()

# Type assertion for mypy - dataset is guaranteed to be not None after st.stop()
if ds is None:
    raise RuntimeError("Dataset should not be None after st.stop()")

tab_overview, tab_analyze, tab_augment, tab_export, tab_viz = st.tabs(
    ["Overview", "Analyze", "Augment", "Export", "Visualize"]
)

with tab_overview:
    info = ds.to_info()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images", info.num_images)
    col2.metric("Annotations", info.num_annotations)
    col3.metric("Classes", len(info.class_names))
    col4.metric("Format", info.format.value.upper())

    if info.class_names:
        st.subheader("Classes")
        st.write(", ".join(info.class_names))

    st.subheader("Image Preview")
    _render_image_grid(ds, n=12)  # type: ignore[arg-type]

with tab_analyze:
    st.subheader("Analysis Tools")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Find Duplicates"):
            with st.spinner("Analyzing..."):
                report = find_duplicates(ds)  # type: ignore[assignment]  # type: ignore[arg-type]  # type: ignore[assignment]
                st.json(report.model_dump())

        if st.button("📊 Quality Report"):
            with st.spinner("Analyzing quality..."):
                report = analyze_quality(ds)  # type: ignore[assignment]  # type: ignore[arg-type]  # type: ignore[assignment]
                st.json(report.model_dump())

        if st.button("🩺 Health Score"):
            with st.spinner("Computing health score..."):
                score = dataset_health_score(ds)  # type: ignore[arg-type]
                st.metric("Overall Health", f"{score.overall:.1f}/100")
                with st.expander("Breakdown"):
                    st.json(score.breakdown.model_dump())
                if score.recommendations:
                    st.info("\n".join(f"- {r}" for r in score.recommendations))

        if st.button("⚠️ Integrity Check"):
            with st.spinner("Checking..."):
                report = check_integrity(ds)  # type: ignore[assignment]  # type: ignore[arg-type]  # type: ignore[assignment]
                st.json(report.model_dump())

    with col2:
        if st.button("📈 Distribution"):
            with st.spinner("Analyzing..."):
                report = analyze_distribution(ds)  # type: ignore[assignment]  # type: ignore[arg-type]  # type: ignore[assignment]
                st.json(report.model_dump())

        if st.button("🔎 Detect Outliers"):
            with st.spinner("Detecting..."):
                report = detect_outliers(ds)  # type: ignore[assignment]  # type: ignore[arg-type]  # type: ignore[assignment]
                st.json(report.model_dump())

        if st.button("🏷️ Auto-Tag Images"):
            with st.spinner("Tagging..."):
                results = generate_captions(ds, model="resnet")  # type: ignore[arg-type]
                st.json(results[:5] if isinstance(results, list) else results)

        if st.button("✏️ Enrich Metadata"):
            with st.spinner("Enriching..."):
                enrich_result: Any = enrich_metadata(ds)  # type: ignore[arg-type]
                st.json(enrich_result)

with tab_augment:
    st.subheader("Augmentation")

    pipeline = st.selectbox("Pipeline", ["yolo_detection", "classification", "segmentation", "custom"])
    multiplier = st.slider("Multiplier", 1, 10, 3)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Augment Dataset"):
            output_dir = tempfile.mkdtemp(prefix="aug_")
            with st.spinner("Augmenting..."):
                augment_result: Any = augment_dataset(ds, pipeline=pipeline, multiplier=multiplier, output_dir=output_dir)  # type: ignore[arg-type, no-redef]
                st.success(f"Generated {augment_result.total_generated} images")
                st.json(augment_result.model_dump())

    with col2:
        if st.button("👁️ Preview"):
            with st.spinner("Generating preview..."):
                preview_result: Any = preview_augmentation(ds, pipeline=pipeline, n_samples=4)  # type: ignore[arg-type, no-redef]
                st.json(preview_result)

    st.subheader("Transformation")
    resize_strategy = st.selectbox("Resize Strategy", ["letterbox", "stretch", "crop"])
    resize_w = st.number_input("Width", 32, 4096, 640, step=32)
    resize_h = st.number_input("Height", 32, 4096, 640, step=32)

    if st.button("📐 Resize Dataset"):
        output_dir = tempfile.mkdtemp(prefix="resize_")
        with st.spinner("Resizing..."):
            resize_result: Any = resize_dataset(ds, size=(resize_w, resize_h), strategy=resize_strategy, output_dir=output_dir)  # type: ignore[arg-type, no-redef]
            st.success(f"Resized {resize_result.num_images_resized} images")
            st.json(resize_result.model_dump())

with tab_export:
    st.subheader("Export / Split")

    col1, col2 = st.columns(2)
    with col1:
        split_strategy = st.selectbox("Split Strategy", ["stratified", "random", "temporal", "grouped"])
        if st.button("✂️ Split Dataset"):
            with st.spinner("Splitting..."):
                split_result: Any = split_dataset(ds, strategy=split_strategy)  # type: ignore[arg-type, no-redef]
                st.json(split_result.model_dump())

    with col2:
        export_format = st.selectbox("Export Format", ["pytorch", "tensorflow", "ultralytics", "hdf5", "webdataset", "cvat_xml", "labelme_json"])
        if st.button("📦 Export"):
            output_dir = tempfile.mkdtemp(prefix=f"export_{export_format}_")
            with st.spinner("Exporting..."):
                if export_format == "webdataset":
                    from pixlint.export.extra_formats import export_webdataset
                    export_result: Any = export_webdataset(ds, output_dir=output_dir)  # type: ignore[arg-type, no-redef]
                elif export_format == "cvat_xml":
                    from pixlint.export.extra_formats import export_cvat_xml
                    export_result: Any = export_cvat_xml(ds, output_path=os.path.join(output_dir, "annotations.xml"))  # type: ignore[arg-type, no-redef]
                elif export_format == "labelme_json":
                    from pixlint.export.extra_formats import export_labelme_json
                    export_result: Any = export_labelme_json(ds, output_dir=output_dir)  # type: ignore[arg-type, no-redef]
                else:
                    from pixlint.export.hdf5 import export_hdf5
                    from pixlint.export.pytorch import export_pytorch
                    from pixlint.export.tensorflow import export_tensorflow
                    from pixlint.export.ultralytics import export_ultralytics
                    if export_format == "pytorch":
                        export_result: Any = export_pytorch(ds, output_dir, image_size=(640, 640))  # type: ignore[arg-type, no-redef]
                    elif export_format == "tensorflow":
                        export_result: Any = export_tensorflow(ds, output_dir)  # type: ignore[arg-type, no-redef]
                    elif export_format == "ultralytics":
                        export_result: Any = export_ultralytics(ds, output_dir)  # type: ignore[arg-type, no-redef]
                    elif export_format == "hdf5":
                        export_result: Any = export_hdf5(ds, output_dir)  # type: ignore[arg-type, no-redef]
                    else:
                        export_result: Any = export_pytorch(ds, output_dir, image_size=(640, 640))  # type: ignore[arg-type, no-redef]
                st.success(f"Exported to {output_dir}")
                st.json(export_result.model_dump() if hasattr(export_result, 'model_dump') else str(export_result))

with tab_viz:
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Distribution Chart (Bar)"):
            from pixlint.visualization.charts import plot_distribution
            result = plot_distribution(ds, chart_type="bar")  # type: ignore[arg-type]
            if "image_base64" in result:
                st.image(result["image_base64"])
            else:
                st.json(result)

        if st.button("🥧 Distribution Chart (Pie)"):
            from pixlint.visualization.charts import plot_distribution
            result = plot_distribution(ds, chart_type="pie")  # type: ignore[arg-type]
            if "image_base64" in result:
                st.image(result["image_base64"])
            else:
                st.json(result)

    with col2:
        if st.button("🌡️ Quality Heatmap"):
            from pixlint.visualization.charts import plot_quality_scores
            result = plot_quality_scores(ds)  # type: ignore[arg-type]
            if "image_base64" in result:
                st.image(result["image_base64"])
            else:
                st.json(result)

        if st.button("🗺️ Spatial Heatmap"):
            from pixlint.visualization.charts import plot_spatial_heatmap
            result = plot_spatial_heatmap(ds)  # type: ignore[arg-type]
            if "image_base64" in result:
                st.image(result["image_base64"])
            else:
                st.json(result)
