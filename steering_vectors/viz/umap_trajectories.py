"""
2D UMAP of all concept steering vectors across all layers.

Each concept gets a trajectory: a sequence of 2D points (one per layer),
colored and connected to show how the concept direction evolves with depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

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
    # Reshape to (n_concepts, n_layers, 2)
    traj = embedding.reshape(n_concepts, n_layers, 2)

    cmap = cm.get_cmap("tab20", n_concepts)
    highlight_concepts = set(highlight_concepts or [])

    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = []
    for i, name in enumerate(concept_names):
        color = cmap(i)
        lw = 3.0 if name in highlight_concepts else 1.5
        alpha = 1.0 if name in highlight_concepts else 0.75

        # Draw trajectory as a gradient line (darker = later layers)
        points = traj[i]  # (n_layers, 2)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        # Alpha ramps from 0.2 (early layers) to 1.0 (late layers)
        alphas = np.linspace(0.2, 1.0, len(segments))
        colors_with_alpha = [(*color[:3], a) for a in alphas]

        lc = LineCollection(segments, colors=colors_with_alpha, linewidths=lw)
        ax.add_collection(lc)

        # Start marker (small circle)
        ax.scatter(*points[0], color=color, s=30, zorder=5, alpha=0.5)
        # End marker (larger circle)
        ax.scatter(*points[-1], color=color, s=80, zorder=6, marker="D")
        # Concept label near endpoint
        ax.annotate(
            name.replace("_", " "),
            xy=points[-1],
            fontsize=6.5,
            alpha=0.9,
            xytext=(3, 3),
            textcoords="offset points",
        )

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=lw, label=name.replace("_", " "))
        )

    ax.autoscale()
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Concept Steering Vector Trajectories Across Layers (UMAP)", fontsize=13)
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        ncol=2,
        framealpha=0.8,
    )

    # Colorbar to indicate layer depth
    sm = cm.ScalarMappable(
        cmap="Greys",
        norm=plt.Normalize(vmin=0, vmax=n_layers - 1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Layer index")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
