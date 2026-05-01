"""
Panel C1 — "Steering Dial" controllability curves.

For one focal prompt, sweep the steering coefficient from -coeff_max to +coeff_max
and plot the resulting concept score(s) as smooth curves.

Two stories per concept:
  1. Target concept curve — should be sigmoid-like, monotonic in coefficient.
  2. Off-target concept curves — should stay flat near 0.5, demonstrating
     that steering is directional, not just generic distortion.

Run:
    cd steering_vectors
    python viz/controllability_curves.py --concept formal_casual
    python viz/controllability_curves.py --all      # the 3 flagship concepts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from tqdm import tqdm

import config
from compute_vectors import get_vector, load_vectors
from scoring import CONCEPT_SCORERS, get_scorer, score_batch
from validate import SteeringHook, generate_text


# Cache directory for sweep arrays — re-render without re-running model
CACHE_DIR = config.VIZ_DIR / "_curves_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Default focal prompts — same prompt across panels keeps the story consistent
FOCAL_PROMPTS: Dict[str, List[str]] = {
    "formal_casual": [
        "Tell me about your weekend.",
        "What did you have for breakfast?",
        "Describe your favorite movie.",
    ],
    "optimistic_pessimistic": [
        "What do you think about the future of climate change?",
        "How will AI affect society in 20 years?",
        "Describe what your career might look like in a decade.",
    ],
    "english_french": [
        "Describe what makes a perfect day.",
        "Tell me about your favorite city.",
        "What is the best way to learn something new?",
    ],
    "happy_sad": [
        "Tell me about your day.",
        "How are you feeling right now?",
        "Describe a recent memory.",
    ],
    "verbose_concise": [
        "What is the meaning of life?",
        "Why do people enjoy music?",
        "Explain why exercise is important.",
    ],
}


# Coefficients to sweep. Symmetric around 0 so the curve goes from full side B
# to baseline to full side A.
DEFAULT_COEFFS = [-12.0, -8.0, -5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 8.0, 12.0]

# Off-target concepts to overlay. Pick ones that use the SAME scorer family
# (sentiment vs length) so they're sensitive enough to detect leakage if it
# exists, plus one regex scorer as a stable baseline.
OFF_TARGET_BY_CONCEPT: Dict[str, List[str]] = {
    "formal_casual":          ["happy_sad", "verbose_concise", "certain_uncertain"],
    "optimistic_pessimistic": ["happy_sad", "verbose_concise", "certain_uncertain"],
    "english_french":         ["formal_casual", "happy_sad", "verbose_concise"],
    "happy_sad":              ["optimistic_pessimistic", "verbose_concise", "certain_uncertain"],
    "verbose_concise":        ["happy_sad", "long_short", "certain_uncertain"],
}


def _scorers_for(concept: str) -> List[str]:
    target = [concept] if concept in CONCEPT_SCORERS else []
    off = [c for c in OFF_TARGET_BY_CONCEPT.get(concept, []) if c in CONCEPT_SCORERS]
    return target + off


def _sweep(
    concept: str,
    prompts: List[str],
    coeffs: List[float],
    layer: int,
    model,
    tokenizer,
    vectors: Tensor,
    concept_names: List[str],
) -> Dict[str, np.ndarray]:
    """For each coefficient, generate steered outputs and score them across multiple concepts."""
    target_idx = concept_names.index(concept)
    scoreable = _scorers_for(concept)
    # results[scorer_name] = array of shape (n_coeffs,) — mean score across prompts
    results: Dict[str, np.ndarray] = {s: np.zeros(len(coeffs)) for s in scoreable}

    for ci, coeff in enumerate(tqdm(coeffs, desc=f"{concept} sweep")):
        steered_texts: List[str] = []
        if abs(coeff) < 1e-6:
            # Baseline: no hook
            for p in prompts:
                steered_texts.append(generate_text(p, model, tokenizer))
        else:
            vec = vectors[target_idx, layer]
            hook = SteeringHook(model, vec, layer_idx=layer, coeff=coeff)
            for p in prompts:
                steered_texts.append(generate_text(p, model, tokenizer, steering_hook=hook))

        # Score with every relevant scorer
        for s in scoreable:
            scores = score_batch(steered_texts, s)
            results[s][ci] = float(np.mean(scores))

    return results


def render_curves(
    concept: str,
    results: Dict[str, np.ndarray],
    coeffs: List[float],
    save_path: Optional[Path] = None,
    figsize=(8, 5.5),
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafafa")

    # Plot off-target curves first (thinner, lower alpha)
    target_color = config.color_for(concept)
    for name, vals in results.items():
        if name == concept:
            continue
        c = config.color_for(name)
        ax.plot(coeffs, vals, color=c, linewidth=1.6, alpha=0.55,
                marker="o", markersize=4,
                label=name.replace("_", " "))

    # Target concept curve last so it sits on top
    if concept in results:
        ax.plot(coeffs, results[concept], color=target_color, linewidth=3.5,
                marker="o", markersize=8,
                label=f"{concept.replace('_', ' ')}  (target)",
                zorder=5)
        # Dot at coeff=0 for baseline emphasis
        zi = coeffs.index(0.0) if 0.0 in coeffs else None
        if zi is not None:
            ax.scatter([0], [results[concept][zi]], s=180, marker="*",
                       color=target_color, edgecolor="black", linewidth=1.0, zorder=6)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Concept Score", fontsize=12)
    ax.set_title(
        f"Steering Dial — {concept.replace('_', ' ')}",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=False)

    # Side annotations: which side is which
    pair = next(p for p in config.CONCEPT_PAIRS if p[0] == concept)
    _, side_a, side_b = pair
    ax.text(0.02, 0.97, f"↑ {side_a.replace('_', ' ')}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=target_color, va="top")
    ax.text(0.02, 0.03, f"↓ {side_b.replace('_', ' ')}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=target_color, va="bottom")

    fig.tight_layout()

    save_path = save_path or (config.VIZ_DIR / f"controllability_{concept}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
    return save_path


def render_curves_grid(
    concepts: List[str],
    all_results: Dict[str, Dict[str, np.ndarray]],
    coeffs: List[float],
    save_path: Optional[Path] = None,
) -> Path:
    """Render multiple concepts as a horizontal small-multiples strip for the poster."""
    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Steering Dial — Smooth, Directional Control via Coefficient",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for ax, concept in zip(axes, concepts):
        ax.set_facecolor("#fafafa")
        results = all_results[concept]
        target_color = config.color_for(concept)

        for name, vals in results.items():
            if name == concept:
                continue
            c = config.color_for(name)
            ax.plot(coeffs, vals, color=c, linewidth=1.6, alpha=0.55,
                    marker="o", markersize=4,
                    label=name.replace("_", " "))

        if concept in results:
            ax.plot(coeffs, results[concept], color=target_color, linewidth=3.5,
                    marker="o", markersize=8,
                    label=f"{concept.replace('_', ' ')}  (target)", zorder=5)

        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xlabel("Steering Coefficient", fontsize=11)
        ax.set_ylabel("Concept Score", fontsize=11)
        ax.set_title(concept.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=False)

        pair = next(p for p in config.CONCEPT_PAIRS if p[0] == concept)
        _, side_a, side_b = pair
        ax.text(0.02, 0.97, f"↑ {side_a.replace('_', ' ')}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=target_color, va="top")
        ax.text(0.02, 0.03, f"↓ {side_b.replace('_', ' ')}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=target_color, va="bottom")

    fig.tight_layout()
    save_path = save_path or (config.VIZ_DIR / "controllability_grid.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
    return save_path


def run_for_concept(
    concept: str,
    model,
    tokenizer,
    layer: int = 14,
    coeffs: Optional[List[float]] = None,
    prompts: Optional[List[str]] = None,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    coeffs = coeffs or DEFAULT_COEFFS
    prompts = prompts or FOCAL_PROMPTS.get(concept, ["Tell me about your day."])
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    cache_path = CACHE_DIR / f"{concept}_layer{layer}.npz"
    if use_cache and cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        results = {k: npz[k] for k in npz.files}
        print(f"Loaded cached curves -> {cache_path}")
        return results, coeffs

    results = _sweep(concept, prompts, coeffs, layer, model, tokenizer, vectors, concept_names)
    np.savez(cache_path, **results)
    print(f"Cached curves -> {cache_path}")
    return results, coeffs


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, default=None,
                        help="Single concept to sweep")
    parser.add_argument("--all", action="store_true",
                        help="Run for the 3 flagship concepts and produce a grid figure")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--from-cache", action="store_true",
                        help="Render only from cached sweeps (no model)")
    args = parser.parse_args()

    # Flagships chosen for SCORER SENSITIVITY: sentiment + length scorers are
    # graded continuously, regex scorers (formal/refuse/french) are sparse
    # and produce flat curves that don't show the smooth control story.
    flagships = ["happy_sad", "verbose_concise", "optimistic_pessimistic"]

    if args.from_cache:
        # Render whatever is cached
        all_results = {}
        for c in flagships:
            cache = CACHE_DIR / f"{c}_layer{args.layer}.npz"
            if cache.exists():
                npz = np.load(cache, allow_pickle=True)
                all_results[c] = {k: npz[k] for k in npz.files}
        if not all_results:
            print("No cached curves found. Run without --from-cache first.")
            return
        coeffs = DEFAULT_COEFFS
        concepts_present = [c for c in flagships if c in all_results]
        for c in concepts_present:
            render_curves(c, all_results[c], coeffs)
        if len(concepts_present) > 1:
            render_curves_grid(concepts_present, all_results, coeffs)
        return

    from extraction import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    if args.all:
        all_results = {}
        for c in flagships:
            results, coeffs = run_for_concept(
                c, model, tokenizer, layer=args.layer,
                vectors=vectors, concept_names=concept_names,
            )
            all_results[c] = results
            render_curves(c, results, coeffs)
        render_curves_grid(flagships, all_results, coeffs)
    elif args.concept:
        results, coeffs = run_for_concept(
            args.concept, model, tokenizer, layer=args.layer,
            vectors=vectors, concept_names=concept_names,
        )
        render_curves(args.concept, results, coeffs)
    else:
        print("Pass --concept <name> or --all")


if __name__ == "__main__":
    main_cli()
