"""
2D UMAP of all concept steering vectors across all layers.

Each concept gets a trajectory: a sequence of 2D points (one per layer),
colored and connected to show how the concept direction evolves with depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from torch import Tensor

import config
from compute_vectors import load_vectors


def _fit_umap(
    vectors: Tensor,
) -> Tuple[np.ndarray, object]:
    """
    Fit UMAP on all (concept, layer) vectors stacked together.

    Args:
        vectors: (n_concepts, n_layers, d_model)

    Returns:
        (embedding, umap_model)
        embedding shape: (n_concepts * n_layers, 2)
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("pip install umap-learn") from e

    n_concepts, n_layers, d_model = vectors.shape
    flat = vectors.reshape(n_concepts * n_layers, d_model).float().numpy()

    reducer = umap.UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.UMAP_RANDOM_STATE,
        n_components=2,
    )
    embedding = reducer.fit_transform(flat)  # (n_concepts * n_layers, 2)
    return embedding, reducer


def plot_umap_trajectories(
    vectors: Tensor,
    concept_names: List[str],
    save_path: Optional[Path] = None,
    highlight_concepts: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot 2D UMAP trajectories for all concepts across layers.

    Each concept's path is drawn as a colored line from layer 0 (small dot)
    to the final layer (large dot). Line opacity encodes layer depth.

    Args:
        vectors: (n_concepts, n_layers, d_model)
        concept_names: Names for the legend.
        save_path: If given, save the figure here.
        highlight_concepts: If given, draw these concepts thicker.

    Returns:
        matplotlib Figure.
    """
    embedding, _ = _fit_umap(vectors)

    n_concepts, n_layers, _ = vectors.shape
    traj = embedding.reshape(n_concepts, n_layers, 2)

    highlight_concepts = set(highlight_concepts or [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")

    for i, name in enumerate(concept_names):
        color = config.color_for(name)
        is_hl = name in highlight_concepts
        lw = 3.5 if is_hl else 1.8
        marker_alpha = 1.0 if is_hl else 0.85

        points = traj[i]  # (n_layers, 2)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.25, 1.0, len(segments))
        # color is hex; convert to RGBA
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color)
        colors_with_alpha = [(*rgb, a) for a in alphas]

        lc = LineCollection(segments, colors=colors_with_alpha, linewidths=lw)
        ax.add_collection(lc)

        # Start marker (small open circle)
        ax.scatter(
            *points[0], facecolors="white", edgecolors=color,
            s=40, zorder=5, alpha=0.8, linewidths=1.2,
        )
        # End marker (filled diamond)
        ax.scatter(
            *points[-1], color=color, s=110, zorder=6,
            marker="D", edgecolors="white", linewidths=1.0, alpha=marker_alpha,
        )
        # Endpoint label, larger for poster
        ax.annotate(
            name.replace("_", " "),
            xy=points[-1],
            fontsize=9,
            fontweight="medium",
            alpha=0.95,
            xytext=(6, 5),
            textcoords="offset points",
            color="#222222",
        )

    ax.autoscale()
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title(
        "Concept Steering Vector Trajectories Across Layers",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, alpha=0.25)

    # Category legend (8 colors instead of 25)
    cat_handles = [
        Line2D([0], [0], color=config.CATEGORY_COLORS[c], linewidth=3, label=c)
        for c in config.CATEGORY_ORDER
    ]
    leg1 = ax.legend(
        handles=cat_handles,
        title="Category",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        title_fontsize=11,
        frameon=False,
    )
    ax.add_artist(leg1)

    # Marker legend explaining start/end
    marker_handles = [
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black",
               markersize=8, linestyle="", label="Layer 0 (start)"),
        Line2D([0], [0], marker="D", color="black",
               markersize=10, linestyle="", label=f"Layer {n_layers-1} (end)"),
    ]
    ax.legend(
        handles=marker_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved UMAP trajectories -> {save_path}")

    return fig


def render_umap_trajectories(
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    output_dir: Path = config.VIZ_DIR,
    highlight_concepts: Optional[List[str]] = None,
) -> Path:
    """Load vectors and render the UMAP trajectory plot."""
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "umap_trajectories.png"
    fig = plot_umap_trajectories(
        vectors, concept_names,
        save_path=out_path,
        highlight_concepts=highlight_concepts,
    )
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    render_umap_trajectories()
