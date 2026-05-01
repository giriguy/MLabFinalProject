"""
Layer specificity profile — line plot of steering-vector strength across layers.

Two modes:

  - "norm" (default, no model needed): plots the L2 norm of the unnormalized
    steering vector at each layer. Activations are loaded from disk and the
    raw mean-difference (without L2 normalize) is recomputed per concept; the
    resulting norm reflects how strongly the concept separates the residual
    stream at each layer.

  - "effectiveness": uses cached effectiveness sweep data (from running
    effectiveness_map.py --run-sweep) to plot effectiveness across layers at
    the optimal coefficient.

Run:
    cd steering_vectors
    python viz/layer_profile.py                   # norm mode
    python viz/layer_profile.py --mode effectiveness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

import config
from compute_vectors import load_vectors


# Concepts to highlight in the profile (one per major category)
DEFAULT_CONCEPTS: List[str] = [
    "formal_casual",        # Pragmatics
    "optimistic_pessimistic",  # Sentiment
    "technical_simple",     # Register
    "english_french",       # Register
    "refuse_comply",        # Intent
]


def _separability(concept_name: str) -> np.ndarray:
    """
    Per-layer Fisher-style separability ratio:
        ||μ_A - μ_B|| / (sqrt(tr(Σ_A) + tr(Σ_B)))

    This normalizes the between-class distance by the within-class spread,
    so it isn't dominated by the monotonic growth of the residual stream
    norm with depth. Higher = the two sides are more cleanly separated at
    that layer. Returns array of length N_LAYERS.
    """
    pair = next(p for p in config.CONCEPT_PAIRS if p[0] == concept_name)
    _, side_a, side_b = pair
    a = np.load(config.ACTIVATIONS_DIR / f"{concept_name}__{side_a}.npy")
    b = np.load(config.ACTIVATIONS_DIR / f"{concept_name}__{side_b}.npy")
    # a, b: (n_examples, n_layers, d_model)
    diff_norm = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0), axis=1)  # (n_layers,)
    # within-class scatter: total variance = sum of per-dim variances
    within = np.sqrt(a.var(axis=0).sum(axis=-1) + b.var(axis=0).sum(axis=-1))
    return diff_norm / (within + 1e-9)


def plot_layer_profile_norms(
    concepts: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize=(11, 6.5),
) -> plt.Figure:
    concepts = concepts or DEFAULT_CONCEPTS
    n_layers = config.N_LAYERS
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")

    for c in concepts:
        try:
            y = _separability(c)
        except FileNotFoundError:
            print(f"Skipping {c}: activations not found")
            continue
        y_norm = y / y.max()
        color = config.color_for(c)
        ax.plot(layers, y_norm, color=color, linewidth=2.6,
                label=c.replace("_", " "), marker="o", markersize=5)
        peak = int(np.argmax(y_norm))
        ax.scatter(peak, y_norm[peak], s=220, marker="*",
                   color=color, edgecolor="black", linewidth=1.0, zorder=10)
        ax.annotate(
            f"layer {peak}",
            xy=(peak, y_norm[peak]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Class Separability (normalized Fisher ratio)", fontsize=12)
    ax.set_title(
        "Different Concepts Live at Different Depths",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.set_xticks(layers[::2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer profile -> {save_path}")
    return fig


def plot_layer_profile_effectiveness(
    concepts: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize=(11, 6.5),
) -> Optional[plt.Figure]:
    """Plot effectiveness across layers using cached sweep data."""
    concepts = concepts or ["formal_casual", "optimistic_pessimistic", "technical_simple"]
    layers_swept = config.LAYER_SWEEP

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")

    rendered = 0
    for c in concepts:
        path = config.VIZ_DIR / f"_effectiveness_{c}.npy"
        if not path.exists():
            print(f"Cached sweep missing for {c}: {path}")
            continue
        sm = np.load(path)  # (n_layers_swept, n_coeffs)
        # Take max across coefficients for each layer
        per_layer = sm.max(axis=1)
        color = config.color_for(c)
        ax.plot(layers_swept, per_layer, color=color, linewidth=2.6,
                label=c.replace("_", " "), marker="o", markersize=6)
        peak = int(np.argmax(per_layer))
        ax.scatter(layers_swept[peak], per_layer[peak], s=220, marker="*",
                   color=color, edgecolor="black", linewidth=1.0, zorder=10)
        rendered += 1

    if rendered == 0:
        plt.close(fig)
        print("No cached effectiveness data found. Run effectiveness_map.py --run-sweep first.")
        return None

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Injection Layer", fontsize=12)
    ax.set_ylabel("Best Effectiveness Score (max over coefficient)", fontsize=12)
    ax.set_title(
        "Steering Effectiveness Peaks Mid-Network",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.set_xticks(layers_swept)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer profile -> {save_path}")
    return fig


def render_layer_profile(
    mode: str = "norm",
    output_dir: Path = config.VIZ_DIR,
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"layer_profile_{mode}.png"
    if mode == "norm":
        fig = plot_layer_profile_norms(save_path=out)
    elif mode == "effectiveness":
        fig = plot_layer_profile_effectiveness(save_path=out)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if fig is not None:
        plt.close(fig)
        return out
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="norm",
                        choices=["norm", "effectiveness"])
    args = parser.parse_args()
    render_layer_profile(mode=args.mode)
