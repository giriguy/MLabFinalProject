"""
Panel C3 — Composition Heatmap.

Stack two steering vectors at varying coefficients on the same residual stream
(matching generate.py's `_build_injection_hooks` summation behavior) and plot
the resulting concept score on a 2D heatmap.

  X-axis: coefficient of vector A (e.g. formal_casual)
  Y-axis: coefficient of vector B (e.g. happy_sad)
  Color : score on a chosen target concept

Demonstrates that steering vectors compose like dials on a mixer — the corners
match the single-vector sweeps from Panel C1.

Run:
    cd steering_vectors
    python viz/composition_heatmap.py
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import Tensor
from tqdm import tqdm

import config
from compute_vectors import load_vectors
from scoring import score_batch
from validate import SteeringHook, generate_text


CACHE_DIR = config.VIZ_DIR / "_composition_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Default focal prompt — same family as C1/C2 for narrative consistency
FOCAL_PROMPTS = [
    "Tell me about your weekend.",
    "Describe what makes a perfect day.",
]


# Three composition pairs to render side-by-side. Each tuple is
# (vec_a, vec_b, target_for_color) — color shows the score for `target`.
DEFAULT_PAIRS = [
    ("formal_casual",          "happy_sad",         "formal_casual"),
    ("optimistic_pessimistic", "verbose_concise",   "optimistic_pessimistic"),
    ("english_french",         "formal_casual",     "english_french"),
]


# Coefficient grid — 7×7 keeps it under 50 generations per pair
DEFAULT_COEFFS = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0]


def _sweep_pair(
    vec_a: str,
    vec_b: str,
    target: str,
    coeffs: List[float],
    layer: int,
    prompts: List[str],
    model,
    tokenizer,
    vectors: Tensor,
    concept_names: List[str],
) -> np.ndarray:
    """Generate at every (cA, cB) cell and return mean target score."""
    idx_a = concept_names.index(vec_a)
    idx_b = concept_names.index(vec_b)
    va = vectors[idx_a, layer]
    vb = vectors[idx_b, layer]

    n = len(coeffs)
    grid = np.zeros((n, n), dtype=np.float32)

    pbar = tqdm(total=n * n, desc=f"{vec_a}+{vec_b}")
    for i, cb in enumerate(coeffs):       # rows = vec_b coefficient
        for j, ca in enumerate(coeffs):   # cols = vec_a coefficient
            steered_texts: List[str] = []
            with ExitStack() as stack:
                if abs(ca) > 1e-6:
                    stack.enter_context(SteeringHook(model, va, layer_idx=layer, coeff=ca))
                if abs(cb) > 1e-6:
                    stack.enter_context(SteeringHook(model, vb, layer_idx=layer, coeff=cb))
                for p in prompts:
                    steered_texts.append(generate_text(p, model, tokenizer))
            scores = score_batch(steered_texts, target)
            grid[i, j] = float(np.mean(scores))
            pbar.update(1)
    pbar.close()
    return grid


def render_pair(
    vec_a: str, vec_b: str, target: str,
    grid: np.ndarray, coeffs: List[float],
    ax: plt.Axes,
    cbar: bool = False,
):
    color_a = config.color_for(vec_a)
    color_b = config.color_for(vec_b)
    color_target = config.color_for(target)

    sns.heatmap(
        grid,
        xticklabels=[f"{c:g}" for c in coeffs],
        yticklabels=[f"{c:g}" for c in coeffs],
        cmap="magma", vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"fontsize": 8, "color": "white"},
        linewidths=0.4, linecolor="white",
        cbar=cbar,
        cbar_kws={"label": f"{target.replace('_', ' ')} score"} if cbar else None,
        ax=ax,
    )
    # Star at the cell with maximum target score
    bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((bj, bi), 1, 1, fill=False,
                                edgecolor="cyan", linewidth=2.5))
    ax.text(bj + 0.5, bi + 0.5, "★",
            ha="center", va="center", fontsize=18,
            color="cyan", fontweight="bold")
    ax.invert_yaxis()  # so positive Y goes up like a math axis

    ax.set_xlabel(f"{vec_a.replace('_', ' ')}  coeff",
                  fontsize=11, color=color_a, fontweight="bold")
    ax.set_ylabel(f"{vec_b.replace('_', ' ')}  coeff",
                  fontsize=11, color=color_b, fontweight="bold")
    ax.set_title(
        f"color = {target.replace('_', ' ')}",
        fontsize=11, color=color_target, fontweight="bold",
    )


def render_composition_grid(
    pairs_data: List[Tuple[str, str, str, np.ndarray]],
    coeffs: List[float],
    save_path: Optional[Path] = None,
) -> Path:
    n = len(pairs_data)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Composition Heatmap — Steering Vectors Stack Like Dials",
        fontsize=15, fontweight="bold", y=1.02,
    )
    for i, (ax, (vec_a, vec_b, target, grid)) in enumerate(zip(axes, pairs_data)):
        render_pair(vec_a, vec_b, target, grid, coeffs, ax,
                    cbar=(i == n - 1))

    fig.tight_layout()
    save_path = save_path or (config.VIZ_DIR / "composition_grid.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
    return save_path


def run_pair(
    vec_a: str, vec_b: str, target: str,
    model, tokenizer,
    layer: int = 14,
    coeffs: Optional[List[float]] = None,
    prompts: Optional[List[str]] = None,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    coeffs = coeffs or DEFAULT_COEFFS
    prompts = prompts or FOCAL_PROMPTS
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    cache_path = CACHE_DIR / f"{vec_a}__{vec_b}__{target}__layer{layer}.npy"
    if use_cache and cache_path.exists():
        print(f"Loaded cache -> {cache_path}")
        return np.load(cache_path)

    grid = _sweep_pair(vec_a, vec_b, target, coeffs, layer, prompts,
                       model, tokenizer, vectors, concept_names)
    np.save(cache_path, grid)
    return grid


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()
    coeffs = DEFAULT_COEFFS

    if args.from_cache:
        pairs_data = []
        for va, vb, tgt in DEFAULT_PAIRS:
            cache_path = CACHE_DIR / f"{va}__{vb}__{tgt}__layer{args.layer}.npy"
            if cache_path.exists():
                pairs_data.append((va, vb, tgt, np.load(cache_path)))
            else:
                print(f"Missing cache for {va}+{vb}: {cache_path}")
        if pairs_data:
            render_composition_grid(pairs_data, coeffs)
        return

    from extraction import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    pairs_data = []
    for va, vb, tgt in DEFAULT_PAIRS:
        grid = run_pair(va, vb, tgt, model, tokenizer,
                        layer=args.layer, coeffs=coeffs,
                        vectors=vectors, concept_names=concept_names)
        pairs_data.append((va, vb, tgt, grid))
    render_composition_grid(pairs_data, coeffs)


if __name__ == "__main__":
    main_cli()
