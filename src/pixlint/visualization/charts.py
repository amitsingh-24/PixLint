from __future__ import annotations

import io
from typing import Any

import numpy as np

from pixlint.analysis.duplicates import find_duplicates
from pixlint.core.loader import CVDataset
from pixlint.utils.schemas import DistributionReport, QualityReport

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches  # noqa: F401
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

_PLOTLY_AVAILABLE = False
try:
    import plotly.express as px  # noqa: F401
    import plotly.graph_objects as go
    import plotly.io as pio  # noqa: F401
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass


def plot_distribution(
    dataset: CVDataset,
    distribution: DistributionReport | None = None,
    chart_type: str = "bar",
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    if distribution is None:
        from pixlint.analysis.distribution import analyze_distribution
        distribution = analyze_distribution(dataset)

    class_names = [d.class_name for d in distribution.class_distribution]
    counts = [d.count for d in distribution.class_distribution]

    if not class_names:
        return {"error": "No class distribution data"}

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "bar":
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        ax.bar(class_names, counts, color=colors)
        ax.set_ylabel("Image Count")
        ax.set_title("Class Distribution")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    elif chart_type == "pie":
        ax.pie(counts, labels=class_names, autopct="%1.1f%%", startangle=90)
        ax.set_title("Class Distribution (Pie)")

    elif chart_type == "horizontal":
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        ax.barh(class_names, counts, color=colors)
        ax.set_xlabel("Image Count")
        ax.set_title("Class Distribution")

    elif chart_type == "treemap" and _PLOTLY_AVAILABLE:
        plt.close(fig)
        fig = go.Figure(go.Treemap(
            labels=class_names,
            parents=[""] * len(class_names),
            values=counts,
            textinfo="label+value+percent root",
        ))
        fig.update_layout(title="Class Distribution (Treemap)")
        if save_path:
            fig.write_image(save_path)
            return {"chart_type": "treemap", "save_path": save_path}
        img_bytes = fig.to_image(format="png")
        import base64
        return {"chart_type": "treemap", "image_base64": base64.b64encode(img_bytes).decode()}

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"chart_type": chart_type, "save_path": save_path}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"chart_type": chart_type, "image_base64": img_b64, "n_classes": len(class_names)}


def plot_quality_scores(
    dataset: CVDataset,
    quality_report: QualityReport | None = None,
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    if quality_report is None:
        from pixlint.analysis.quality import analyze_quality
        quality_report = analyze_quality(dataset)

    scores = [q.overall_score for q in quality_report.per_image if q.overall_score is not None]
    if not scores:
        return {"error": "No quality scores available"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(scores, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Image Count")
    axes[0].set_title("Quality Score Distribution")
    axes[0].axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.1f}")
    axes[0].legend()

    flagged = quality_report.flagged_images
    total = len(scores)
    clean = total - flagged
    axes[1].pie([clean, flagged], labels=["Pass", "Flagged"], autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90)
    axes[1].set_title(f"Quality Check: {flagged}/{total} Flagged")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"save_path": save_path}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"image_base64": img_b64, "avg_score": round(np.mean(scores), 2), "flagged": flagged}


def plot_spatial_heatmap(
    dataset: CVDataset,
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    from pixlint.analysis.distribution import analyze_distribution
    dist = analyze_distribution(dataset, analyses=["spatial"])
    if not dist.spatial_heatmap:
        return {"error": "No spatial heatmap data"}

    heatmap = np.array(dist.spatial_heatmap.grid)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest", aspect="equal")
    ax.set_title("Spatial Object Distribution")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.colorbar(im, ax=ax, label="Object Density (%)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"save_path": save_path}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"image_base64": img_b64, "grid_size": dist.spatial_heatmap.grid_size}


def plot_duplicate_groups(
    dataset: CVDataset,
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    report = find_duplicates(dataset, methods=["exact", "perceptual"])
    groups = report.duplicate_groups

    if not groups:
        return {"error": "No duplicate groups found", "total_images": report.total_duplicate_images}

    methods = [g.method for g in groups]
    method_counts = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods_names = list(method_counts.keys())
    methods_values = list(method_counts.values())
    axes[0].bar(methods_names, methods_values, color=["#3498db", "#e67e22", "#2ecc71"])
    axes[0].set_xlabel("Detection Method")
    axes[0].set_ylabel("Number of Groups")
    axes[0].set_title("Duplicate Groups by Method")

    group_sizes = [len(g.image_ids) for g in groups]
    axes[1].hist(group_sizes, bins=min(20, max(group_sizes)), color="coral", edgecolor="white")
    axes[1].set_xlabel("Images per Group")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Duplicate Group Size Distribution")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"save_path": save_path, "total_groups": len(groups)}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {
        "image_base64": img_b64,
        "total_groups": len(groups),
        "total_duplicates": report.total_duplicate_images,
    }


def plot_class_balance_radar(
    dataset: CVDataset,
    save_path: str | None = None,
) -> dict[str, Any]:
    if not _MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    from pixlint.analysis.distribution import analyze_distribution
    dist = analyze_distribution(dataset)
    class_names = [d.class_name for d in dist.class_distribution]

    if len(class_names) < 3:
        return {"error": "Need at least 3 classes for radar chart"}

    counts = np.array([d.count for d in dist.class_distribution])
    normalized = counts / counts.max() * 100

    angles = np.linspace(0, 2 * np.pi, len(class_names), endpoint=False).tolist()
    values = normalized.tolist() + [normalized.tolist()[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, "o-", linewidth=2, color="#3498db")
    ax.fill(angles, values, alpha=0.25, color="#3498db")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Class Balance Radar", pad=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"save_path": save_path}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"image_base64": img_b64, "n_classes": len(class_names)}
