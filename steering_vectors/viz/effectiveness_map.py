"""
Steering effectiveness heatmap: sweep (injection_layer × coefficient), score outputs.

For each (layer, coeff) cell we generate steered outputs for a fixed prompt set
and score them, then plot the score as a heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import Tensor
from tqdm import tqdm

import config
from compute_vectors import get_vector, load_vectors
from scoring import score_batch
from validate import SteeringHook, generate_text


def sweep_layer_coeff(
    concept_name: str,
    test_prompts: List[str],
    model,
    tokenizer,
    layers: Optional[List[int]] = None,
    coeffs: Optional[List[float]] = None,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Run the full (layer × coeff) sweep and return a score matrix.

    Args:
        concept_name: Which concept's steering vector to use.
        test_prompts: Prompts used for scoring.
        model, tokenizer: Loaded model.
        layers: Layer indices to sweep (default: config.LAYER_SWEEP).
        coeffs: Coefficient values to sweep (default: config.COEFF_SWEEP).

    Returns:
        score_matrix: np.ndarray of shape (n_layers, n_coeffs) with scores in [0, 1].
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()
    if layers is None:
        layers = config.LAYER_SWEEP
    if coeffs is None:
        coeffs = config.COEFF_SWEEP

    score_matrix = np.zeros((len(layers), len(coeffs)), dtype=np.float32)

    for li, layer_idx in enumerate(tqdm(layers, desc=f"Sweeping {concept_name}")):
        vector = get_vector(concept_name, layer_idx, vectors, concept_names)
        for ci, coeff in enumerate(coeffs):
            steered_texts = []
            hook = SteeringHook(model, vector, layer_idx=layer_idx, coeff=coeff)
            for prompt in test_prompts:
                text = generate_text(prompt, model, tokenizer, steering_hook=hook)
                steered_texts.append(text)
            scores = score_batch(steered_texts, concept_name)
            score_matrix[li, ci] = float(np.mean(scores))

    return score_matrix


def plot_effectiveness_heatmap(
    score_matrix: np.ndarray,
    layers: List[int],
    coeffs: List[float],
    concept_name: str,
    save_path: Optional[Path] = None,
    baseline_score: Optional[float] = None,
) -> plt.Figure:
    """
    Render a heatmap of steering effectiveness scores.

    Args:
        score_matrix: (n_layers, n_coeffs) in [0, 1].
        layers: Layer indices (y-axis).
        coeffs: Coefficient values (x-axis).
        concept_name: Title label.
        baseline_score: If given, annotate with a contour at this score level.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        score_matrix,
        xticklabels=[f"{c:.0f}" for c in coeffs],
        yticklabels=layers,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Effectiveness score (side A)"},
    )

    if baseline_score is not None:
        # Mark cells that are meaningfully above baseline
        threshold = baseline_score + 0.1
        for li in range(len(layers)):
            for ci in range(len(coeffs)):
                if score_matrix[li, ci] >= threshold:
                    ax.add_patch(plt.Rectangle((ci, li), 1, 1, fill=False, edgecolor="blue", lw=2))

    ax.set_xlabel("Coefficient strength")
    ax.set_ylabel("Injection layer")
    ax.set_title(f"Steering Effectiveness — {concept_name.replace('_', ' ')}", fontsize=13)

    best_li, best_ci = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    ax.text(
        best_ci + 0.5,
        best_li + 0.5,
        "★",
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved effectiveness map -> {save_path}")

    return fig


def render_effectiveness_map(
    concept_name: str,
    test_prompts: List[str],
    model,
    tokenizer,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    coeffs: Optional[List[float]] = None,
    output_dir: Path = config.VIZ_DIR,
) -> Tuple[np.ndarray, Path]:
    """
    Full pipeline: sweep, plot, save.

    Returns:
        (score_matrix, path_to_saved_figure)
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()
    if layers is None:
        layers = config.LAYER_SWEEP
    if coeffs is None:
        coeffs = config.COEFF_SWEEP

    score_matrix = sweep_layer_coeff(
        concept_name, test_prompts, model, tokenizer,
        layers=layers, coeffs=coeffs,
        vectors=vectors, concept_names=concept_names,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"effectiveness_{concept_name}.png"
    fig = plot_effectiveness_heatmap(score_matrix, layers, coeffs, concept_name, save_path=out_path)
    plt.close(fig)

    return score_matrix, out_path


def render_all_effectiveness_maps(
    concepts_and_prompts: Dict[str, List[str]],
    model,
    tokenizer,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    output_dir: Path = config.VIZ_DIR,
) -> Dict[str, np.ndarray]:
    """
    Render effectiveness maps for multiple concepts.

    Args:
        concepts_and_prompts: {concept_name: [test_prompt, ...]}

    Returns:
        {concept_name: score_matrix}
    """
    results = {}
    for concept_name, prompts in concepts_and_prompts.items():
        score_matrix, _ = render_effectiveness_map(
            concept_name, prompts, model, tokenizer,
            vectors=vectors, concept_names=concept_names,
            output_dir=output_dir,
        )
        results[concept_name] = score_matrix
    return results


# ---------------------------------------------------------------------------
# Poster panel: 3 concepts side-by-side with shared colorbar
# ---------------------------------------------------------------------------

POSTER_CONCEPTS = ["formal_casual", "optimistic_pessimistic", "technical_simple"]
POSTER_PROMPTS = {
    "formal_casual": [
        "Tell me about your weekend.",
        "What is the meaning of life?",
        "Describe your favorite meal.",
        "How do you feel about deadlines?",
        "What advice would you give a new student?",
    ],
    "optimistic_pessimistic": [
        "What do you think about the future of climate change?",
        "How will AI affect the job market?",
        "Describe the next decade for humanity.",
        "What are your thoughts on space exploration?",
        "How will cities evolve over the next century?",
    ],
    "technical_simple": [
        "Explain how a transformer model works.",
        "How does a refrigerator work?",
        "Why does the sky look blue?",
        "What causes inflation in an economy?",
        "How do vaccines train the immune system?",
    ],
}


def render_three_concept_grid(
    model,
    tokenizer,
    concepts: Optional[List[str]] = None,
    prompts: Optional[Dict[str, List[str]]] = None,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    coeffs: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Path]:
    """Sweep 3 flagship concepts and render them as a single 1×3 figure with shared colorbar."""
    concepts = concepts or POSTER_CONCEPTS
    prompts = prompts or POSTER_PROMPTS
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()
    if layers is None:
        layers = config.LAYER_SWEEP
    if coeffs is None:
        coeffs = config.COEFF_SWEEP

    score_mats: Dict[str, np.ndarray] = {}
    for c in concepts:
        score_mats[c] = sweep_layer_coeff(
            c, prompts[c], model, tokenizer,
            layers=layers, coeffs=coeffs,
            vectors=vectors, concept_names=concept_names,
        )
        # Persist intermediate results so a crash doesn't lose the sweep
        np.save(config.VIZ_DIR / f"_effectiveness_{c}.npy", score_mats[c])

    fig, axes = plt.subplots(1, len(concepts), figsize=(6 * len(concepts), 6.5))
    if len(concepts) == 1:
        axes = [axes]

    fig.suptitle(
        "Steering Effectiveness — Where Does Each Concept Live?",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for ax, c in zip(axes, concepts):
        sm = score_mats[c]
        sns.heatmap(
            sm,
            xticklabels=[f"{x:.0f}" for x in coeffs],
            yticklabels=layers,
            cmap="magma",
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            linewidths=0.4, linecolor="white",
            ax=ax,
            cbar=False,
            annot_kws={"fontsize": 8, "color": "white"},
        )
        best_li, best_ci = np.unravel_index(np.argmax(sm), sm.shape)
        ax.add_patch(plt.Rectangle((best_ci, best_li), 1, 1,
                                    fill=False, edgecolor="cyan", lw=2.5))
        ax.text(best_ci + 0.5, best_li + 0.5, "★",
                ha="center", va="center", fontsize=18,
                color="cyan", fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=11)
        ax.set_ylabel("Injection Layer", fontsize=11)
        ax.set_title(c.replace("_", " ").title(), fontsize=13, fontweight="bold")

    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"),
                        cax=cbar_ax, label="Effectiveness score")

    fig.tight_layout()

    save_path = save_path or (config.VIZ_DIR / "effectiveness_grid_poster.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved effectiveness grid -> {save_path}")
    return score_mats, save_path


def render_three_concept_grid_from_cached(
    concepts: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    coeffs: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
) -> Optional[Path]:
    """Build the poster grid from already-cached _effectiveness_<concept>.npy files."""
    concepts = concepts or POSTER_CONCEPTS
    layers = layers or config.LAYER_SWEEP
    coeffs = coeffs or config.COEFF_SWEEP

    score_mats: Dict[str, np.ndarray] = {}
    for c in concepts:
        path = config.VIZ_DIR / f"_effectiveness_{c}.npy"
        if not path.exists():
            print(f"Missing cached sweep for {c} at {path}")
            return None
        score_mats[c] = np.load(path)

    fig, axes = plt.subplots(1, len(concepts), figsize=(6 * len(concepts), 6.5))
    if len(concepts) == 1:
        axes = [axes]
    fig.suptitle(
        "Steering Effectiveness — Where Does Each Concept Live?",
        fontsize=16, fontweight="bold", y=1.02,
    )
    for ax, c in zip(axes, concepts):
        sm = score_mats[c]
        sns.heatmap(
            sm, xticklabels=[f"{x:.0f}" for x in coeffs], yticklabels=layers,
            cmap="magma", vmin=0, vmax=1, annot=True, fmt=".2f",
            linewidths=0.4, linecolor="white", ax=ax, cbar=False,
            annot_kws={"fontsize": 8, "color": "white"},
        )
        best_li, best_ci = np.unravel_index(np.argmax(sm), sm.shape)
        ax.add_patch(plt.Rectangle((best_ci, best_li), 1, 1,
                                    fill=False, edgecolor="cyan", lw=2.5))
        ax.text(best_ci + 0.5, best_li + 0.5, "★",
                ha="center", va="center", fontsize=18,
                color="cyan", fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=11)
        ax.set_ylabel("Injection Layer", fontsize=11)
        ax.set_title(c.replace("_", " ").title(), fontsize=13, fontweight="bold")

    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"),
                 cax=cbar_ax, label="Effectiveness score")
    fig.tight_layout()
    save_path = save_path or (config.VIZ_DIR / "effectiveness_grid_poster.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved effectiveness grid -> {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true",
                        help="Render poster grid from cached .npy sweeps (no model needed)")
    parser.add_argument("--run-sweep", action="store_true",
                        help="Run the full sweep + render (loads model, slow)")
    args = parser.parse_args()

    if args.from_cache:
        render_three_concept_grid_from_cached()
    elif args.run_sweep:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from extraction import load_model_and_tokenizer
        m, t = load_model_and_tokenizer()
        render_three_concept_grid(m, t)
    else:
        print("Pass --run-sweep (slow) or --from-cache (fast).")
