"""
Steering effectiveness heatmap: sweep (injection_layer × coefficient), score outputs.

For each (layer, coeff) cell we generate steered outputs for a fixed prompt set
and score them, then plot the score as a heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
