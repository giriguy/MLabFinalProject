"""
Panel C2 — Concept Radar.

For one focal prompt, score the model's response on every measurable concept
(those in scoring.CONCEPT_SCORERS) under multiple steering conditions, and
plot the result as overlapping polygons on a polar (radar) axis.

Each radar shows:
  - GRAY polygon: baseline output's concept profile
  - COLORED polygon: a single-concept-steered output's profile

Small multiples row of 3 radars demonstrates that steering surgically
distorts the targeted axis while leaving most others near baseline.

Run:
    cd steering_vectors
    python viz/concept_radar.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from tqdm import tqdm

import config
from compute_vectors import load_vectors
from scoring import CONCEPT_SCORERS, score_batch
from validate import SteeringHook, generate_text


CACHE_DIR = config.VIZ_DIR / "_radar_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Concepts displayed on the radar axes. Reordered so visually-related axes
# are adjacent (cleaner polygon shapes).
RADAR_AXES = [
    "formal_casual",
    "verbose_concise",
    "long_short",
    "happy_sad",
    "positive_negative",
    "optimistic_pessimistic",
    "emotional_neutral",
    "certain_uncertain",
    "english_french",
    "refuse_comply",
]


# Steering conditions to render: (target_concept, coeff)
DEFAULT_STEER_CONDITIONS = [
    ("formal_casual", 8.0),
    ("optimistic_pessimistic", 8.0),
    ("english_french", 8.0),
]


# Focal prompts averaged together for stability
FOCAL_PROMPTS = [
    "Tell me about your weekend.",
    "Describe what makes a perfect day.",
    "What did you have for breakfast?",
]


def _profile_for_outputs(texts: List[str]) -> Dict[str, float]:
    """Return mean concept score across `texts` for every axis on the radar."""
    out = {}
    for c in RADAR_AXES:
        if c not in CONCEPT_SCORERS:
            out[c] = 0.5
            continue
        scores = score_batch(texts, c)
        out[c] = float(np.mean(scores))
    return out


def _gen_steered(
    target_concept: str, coeff: float, layer: int,
    prompts: List[str], model, tokenizer,
    vectors: Tensor, concept_names: List[str],
) -> List[str]:
    idx = concept_names.index(target_concept)
    vec = vectors[idx, layer]
    out = []
    hook = SteeringHook(model, vec, layer_idx=layer, coeff=coeff)
    for p in prompts:
        out.append(generate_text(p, model, tokenizer, steering_hook=hook))
    return out


def gather_data(
    model,
    tokenizer,
    layer: int = 14,
    conditions: Optional[List] = None,
    prompts: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Dict:
    """Generate baseline + each steered condition and score everything."""
    conditions = conditions or DEFAULT_STEER_CONDITIONS
    prompts = prompts or FOCAL_PROMPTS

    cache_path = CACHE_DIR / f"radar_layer{layer}.npz"
    if use_cache and cache_path.exists():
        data = dict(np.load(cache_path, allow_pickle=True))
        print(f"Loaded radar cache -> {cache_path}")
        return {k: data[k].item() if data[k].dtype == object else data[k]
                for k in data.files}

    vectors, concept_names = load_vectors()

    print("Generating baseline outputs...")
    baseline_texts = [generate_text(p, model, tokenizer) for p in prompts]
    baseline_profile = _profile_for_outputs(baseline_texts)

    steered_profiles = {}
    for target, coeff in tqdm(conditions, desc="Steering conditions"):
        texts = _gen_steered(target, coeff, layer, prompts, model, tokenizer,
                             vectors, concept_names)
        steered_profiles[(target, coeff)] = _profile_for_outputs(texts)

    data = {
        "baseline": baseline_profile,
        "steered": steered_profiles,
        "axes": RADAR_AXES,
    }
    # Save with object arrays for the dicts
    np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in data.items()})
    print(f"Cached radar -> {cache_path}")
    return data


def render_radar_grid(
    data: Dict,
    save_path: Optional[Path] = None,
) -> Path:
    axes_concepts = data["axes"]
    baseline = data["baseline"]
    steered = data["steered"]
    conditions = list(steered.keys())

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5),
                             subplot_kw={"projection": "polar"})
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Concept Radar — Steering Distorts Target Axis, Spares Others",
        fontsize=15, fontweight="bold", y=1.05,
    )

    n_axes = len(axes_concepts)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    for ax, cond in zip(axes, conditions):
        target, coeff = cond
        target_color = config.color_for(target)

        baseline_vals = [baseline[c] for c in axes_concepts] + [baseline[axes_concepts[0]]]
        steered_vals = [steered[cond][c] for c in axes_concepts] + [steered[cond][axes_concepts[0]]]

        ax.plot(angles_closed, baseline_vals, color="grey", linewidth=2, alpha=0.85,
                label="Baseline")
        ax.fill(angles_closed, baseline_vals, color="grey", alpha=0.15)

        ax.plot(angles_closed, steered_vals, color=target_color, linewidth=2.5,
                label=f"Steered  +{coeff:g}")
        ax.fill(angles_closed, steered_vals, color=target_color, alpha=0.25)

        # Highlight the target axis
        if target in axes_concepts:
            t_idx = axes_concepts.index(target)
            ax.scatter([angles[t_idx]], [steered[cond][target]],
                       s=180, color=target_color, edgecolor="black",
                       linewidth=1.5, zorder=10, marker="*")

        ax.set_xticks(angles)
        ax.set_xticklabels([c.replace("_", "\n") for c in axes_concepts], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=7, color="grey")
        ax.set_title(
            f"{target.replace('_', ' ')}",
            fontsize=12, fontweight="bold", color=target_color, pad=22,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
                  fontsize=9, frameon=False)
        ax.grid(alpha=0.4)

    fig.tight_layout()
    save_path = save_path or (config.VIZ_DIR / "concept_radar.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
    return save_path


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()

    if args.from_cache:
        cache_path = CACHE_DIR / f"radar_layer{args.layer}.npz"
        if not cache_path.exists():
            print(f"No cache at {cache_path}")
            return
        data_raw = dict(np.load(cache_path, allow_pickle=True))
        data = {
            "baseline": data_raw["baseline"].item(),
            "steered": data_raw["steered"].item(),
            "axes": list(data_raw["axes"].item()) if data_raw["axes"].dtype == object
                    else list(data_raw["axes"]),
        }
        render_radar_grid(data)
        return

    from extraction import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer()
    data = gather_data(model, tokenizer, layer=args.layer)
    render_radar_grid(data)


if __name__ == "__main__":
    main_cli()
