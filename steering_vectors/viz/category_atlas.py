"""
Concept atlas — radial layout of all 25 steering concepts grouped by category.

Acts as the legend / orientation panel for the poster: every other panel uses
the same Okabe-Ito category colors defined in config.CATEGORY_COLORS.

Run:
    cd steering_vectors
    python viz/category_atlas.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from compute_vectors import load_vectors


POSTER_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _ordered_concepts() -> List[tuple]:
    """Return [(concept_name, category, color), ...] grouped by category order."""
    out = []
    for cat in config.CATEGORY_ORDER:
        for name in config.CONCEPT_NAMES:
            if config.category_for(name) == cat:
                out.append((name, cat, config.CATEGORY_COLORS[cat]))
    return out


def plot_concept_atlas(
    save_path: Optional[Path] = None,
    figsize=(10, 10),
    use_norms: bool = True,
) -> plt.Figure:
    """
    Polar atlas: 25 concepts arranged around a circle, grouped by category.

    Each concept is drawn as a colored wedge whose radial length is proportional
    to the L2 norm of its mid-layer steering vector (a proxy for "how strong"
    the direction is when use_norms=True). Set use_norms=False for equal wedges.
    """
    plt.rcParams.update(POSTER_RC)
    ordered = _ordered_concepts()
    n = len(ordered)

    radii = np.ones(n)
    if use_norms:
        try:
            vectors, names = load_vectors()
            mid = config.N_LAYERS // 2
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            norms = np.array(
                [vectors[name_to_idx[c], mid].float().norm().item() for c, _, _ in ordered]
            )
            radii = 0.55 + 0.45 * (norms - norms.min()) / (np.ptp(norms) + 1e-9)
        except (FileNotFoundError, KeyError):
            radii = np.ones(n)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    width = 2 * np.pi / n
    theta = np.array([i * width for i in range(n)])
    colors = [c for _, _, c in ordered]

    bars = ax.bar(
        theta,
        radii,
        width=width * 0.92,
        color=colors,
        edgecolor="white",
        linewidth=2.0,
        bottom=0.15,
        alpha=0.92,
    )

    # Concept labels around the rim
    for ang, r, (name, _cat, _col) in zip(theta, radii, ordered):
        rotation_deg = np.degrees(ang)
        # flip text on the bottom half so it reads outward
        if 90 < rotation_deg < 270:
            rot = rotation_deg + 180
            ha = "right"
        else:
            rot = rotation_deg
            ha = "left"
        ax.text(
            ang,
            r + 0.22,
            name.replace("_", " "),
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=10,
            fontweight="medium",
        )

    # Inner category ring (color band, no overlapping text)
    inner_r = 0.08
    cats_in_order = [c for _, c, _ in ordered]
    seen_cat_starts = {}
    for i, cat in enumerate(cats_in_order):
        seen_cat_starts.setdefault(cat, []).append(i)

    for cat, idxs in seen_cat_starts.items():
        start = theta[idxs[0]] - width / 2
        end = theta[idxs[-1]] + width / 2
        arc_theta = np.linspace(start, end, 60)
        ax.fill_between(
            arc_theta,
            inner_r,
            inner_r + 0.045,
            color=config.CATEGORY_COLORS[cat],
            alpha=0.95,
        )

    ax.set_ylim(0, 1.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title(
        "Concept Atlas — 25 Steering Directions in 8 Semantic Categories",
        pad=28,
        fontsize=15,
    )

    # Legend in the corner
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.CATEGORY_COLORS[c])
        for c in config.CATEGORY_ORDER
    ]
    ax.legend(
        handles,
        config.CATEGORY_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved concept atlas -> {save_path}")

    return fig


def render_concept_atlas(output_dir: Path = config.VIZ_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "concept_atlas.png"
    fig = plot_concept_atlas(save_path=out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    render_concept_atlas()
