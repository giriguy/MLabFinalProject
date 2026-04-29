"""
Animate the cosine similarity matrix evolving across transformer layers.

Exports a GIF or MP4 using matplotlib.animation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from torch import Tensor

import config
from compute_vectors import layer_similarity_matrices, load_vectors


def build_animation(
    vectors: Tensor,
    concept_names: List[str],
    interval_ms: int = 200,
    figsize: tuple = (10, 9),
) -> FuncAnimation:
    """
    Build a matplotlib FuncAnimation showing the similarity matrix at each layer.

    Args:
        vectors: (n_concepts, n_layers, d_model)
        concept_names: Names for axis labels.
        interval_ms: Milliseconds between frames.

    Returns:
        FuncAnimation object (call .save() or plt.show() on it).
    """
    n_layers = vectors.shape[1]
    sim_matrices = layer_similarity_matrices(vectors).numpy()  # (n_layers, n_c, n_c)
    labels = [name.replace("_", " ") for name in concept_names]
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_matrices[0], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="Cosine similarity", fraction=0.04)
    title = ax.set_title("Layer 0")
    fig.tight_layout()

    def update(frame: int):
        im.set_data(sim_matrices[frame])
        title.set_text(f"Cosine Similarity — Layer {frame}/{n_layers - 1}")
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_layers,
        interval=interval_ms,
        blit=True,
    )
    return anim


def save_as_gif(
    anim: FuncAnimation,
    path: Path,
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Save animation as an animated GIF using Pillow."""
    writer = PillowWriter(fps=fps)
    anim.save(str(path), writer=writer, dpi=dpi)
    print(f"Saved GIF -> {path}")


def save_as_mp4(
    anim: FuncAnimation,
    path: Path,
    fps: int = 5,
    dpi: int = 100,
) -> None:
    """Save animation as MP4. Requires ffmpeg in PATH."""
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(str(path), writer=writer, dpi=dpi)
    print(f"Saved MP4 -> {path}")


def render_layer_animation(
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    output_dir: Path = config.VIZ_DIR,
    fmt: str = "gif",
    fps: int = 5,
) -> Path:
    """
    Full pipeline: load vectors, build animation, export to file.

    Args:
        fmt: "gif" or "mp4".

    Returns:
        Path to the saved animation file.
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    anim = build_animation(vectors, concept_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"layer_animation.{fmt}"

    if fmt == "gif":
        save_as_gif(anim, out_path, fps=fps)
    elif fmt == "mp4":
        save_as_mp4(anim, out_path, fps=fps)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Choose 'gif' or 'mp4'.")

    plt.close("all")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", default="gif", choices=["gif", "mp4"])
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    render_layer_animation(fmt=args.fmt, fps=args.fps)
