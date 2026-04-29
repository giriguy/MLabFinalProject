"""
Cosine similarity heatmap across concept directions at a chosen layer.

Produces a clustered heatmap with dendrogram using seaborn.clustermap.
Optionally launches an interactive slider to choose the layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.widgets import Slider
from torch import Tensor

import config
from compute_vectors import cosine_similarity_matrix, layer_similarity_matrices, load_vectors


def plot_cosine_heatmap(
    vectors: Tensor,
    concept_names: List[str],
    layer_idx: int = 0,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> sns.matrix.ClusterGrid:
    """
    Render a clustered cosine-similarity heatmap for all concepts at `layer_idx`.

    Args:
        vectors: (n_concepts, n_layers, d_model)
        concept_names: Names aligned with dimension 0 of vectors.
        layer_idx: Which layer to visualize.
        save_path: If given, save the figure to this path.
        title: Override the default figure title.

    Returns:
        seaborn ClusterGrid object.
    """
    vecs_at_layer = vectors[:, layer_idx, :].float().numpy()  # (n_concepts, d_model)
    sim_matrix = cosine_similarity_matrix(vectors[:, layer_idx, :]).numpy()

    labels = [name.replace("_", "\n") for name in concept_names]

    g = sns.clustermap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(14, 12),
        annot=len(concept_names) <= 15,  # only annotate if few enough concepts
        fmt=".2f",
        linewidths=0.5,
        dendrogram_ratio=0.15,
    )
    _title = title or f"Concept Cosine Similarity — Layer {layer_idx}"
    g.figure.suptitle(_title, y=1.01, fontsize=14, fontweight="bold")

    if save_path is not None:
        g.figure.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved heatmap -> {save_path}")

    return g


def interactive_layer_heatmap(
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> None:
    """
    Display an interactive heatmap with a layer slider.

    Loads vectors from disk if not provided.
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    n_layers = vectors.shape[1]
    sim_matrices = layer_similarity_matrices(vectors).numpy()  # (n_layers, n_concepts, n_concepts)

    labels = [name.replace("_", " ") for name in concept_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)

    im = ax.imshow(sim_matrices[0], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    title_text = ax.set_title(f"Concept Cosine Similarity — Layer 0")

    # Layer slider
    ax_slider = plt.axes([0.2, 0.04, 0.6, 0.03])
    slider = Slider(ax_slider, "Layer", 0, n_layers - 1, valinit=0, valstep=1)

    def update(val):
        layer = int(slider.val)
        im.set_data(sim_matrices[layer])
        title_text.set_text(f"Concept Cosine Similarity — Layer {layer}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def save_all_layer_heatmaps(
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    output_dir: Path = config.VIZ_DIR,
    stride: int = 4,
) -> None:
    """
    Save static heatmap images for layers 0, stride, 2*stride, ...

    Useful for batch inspection without the interactive slider.
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    n_layers = vectors.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(0, n_layers, stride):
        path = output_dir / f"cosine_heatmap_layer{layer_idx:02d}.png"
        plot_cosine_heatmap(vectors, concept_names, layer_idx=layer_idx, save_path=path)
        plt.close("all")


if __name__ == "__main__":
    interactive_layer_heatmap()
