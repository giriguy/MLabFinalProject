"""
Qualitative gallery — side-by-side baseline vs steered text outputs.

Usage:
    cd steering_vectors

    # Step 1 (slow, requires model). Fills baseline + steered into the JSON.
    python viz/qualitative_gallery.py --generate

    # Step 2 (fast). Renders the gallery PNG from the populated JSON.
    python viz/qualitative_gallery.py --render

The JSON lives at data/poster_examples.json. Hand-edit to curate the most
striking examples after step 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config

EXAMPLES_PATH = config.DATA_DIR / "poster_examples.json"


def _load_examples() -> dict:
    with open(EXAMPLES_PATH) as f:
        return json.load(f)


def _save_examples(data: dict) -> None:
    with open(EXAMPLES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def generate_examples() -> None:
    """Run the model on each (concept, prompt) and fill in baseline + steered text."""
    from compute_vectors import load_vectors
    from extraction import load_model_and_tokenizer
    from validate import SteeringHook, generate_text

    data = _load_examples()
    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    for ex in data["examples"]:
        concept = ex["concept"]
        if concept not in concept_names:
            print(f"Skipping unknown concept: {concept}")
            continue
        idx = concept_names.index(concept)
        layer = ex.get("layer", config.N_LAYERS // 2)
        coeff = ex.get("coeff", config.DEFAULT_COEFF)
        prompt = ex["prompt"]
        sv = vectors[idx, layer]

        print(f"\n=== {concept} (layer={layer}, coeff={coeff}) ===")
        print(f"Prompt: {prompt}")

        baseline = generate_text(prompt, model, tokenizer)
        print(f"Baseline: {baseline[:120]}...")
        ex["baseline"] = baseline.strip()

        with SteeringHook(model, sv, layer_idx=layer, coeff=coeff) as hook:
            steered = generate_text(prompt, model, tokenizer, steering_hook=hook)
        print(f"Steered:  {steered[:120]}...")
        ex["steered"] = steered.strip()

        # Persist after each one so a crash doesn't lose work
        _save_examples(data)

    print(f"\nDone. Examples saved to {EXAMPLES_PATH}")


def _wrap(text: str, width: int = 55) -> str:
    if not text:
        return "(empty — run --generate)"
    paragraphs = text.split("\n")
    wrapped = []
    for p in paragraphs:
        wrapped.append(textwrap.fill(p, width=width))
    return "\n".join(wrapped)


def render_gallery(save_path: Path = None) -> Path:
    data = _load_examples()
    examples = [e for e in data["examples"] if e.get("baseline") or e.get("steered")]
    if not examples:
        print("No populated examples in JSON — run --generate first.")
        return None

    n = len(examples)
    n_cols = 2  # baseline | steered
    n_rows = n
    fig_h = 2.4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, fig_h))
    if n_rows == 1:
        axes = axes[None, :]

    fig.suptitle(
        "Steering in Action — Baseline vs Steered Output",
        fontsize=16, fontweight="bold", y=1.005,
    )

    for row, ex in enumerate(examples):
        cat_color = config.color_for(ex["concept"])
        # Concept header strip on the left
        for col, key, label in [(0, "baseline", "BASELINE"), (1, "steered", "STEERED")]:
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Header bar
            bar = mpatches.Rectangle((0, 0.92), 1, 0.08,
                                     transform=ax.transAxes,
                                     color=cat_color if col == 1 else "#cccccc",
                                     alpha=0.85, clip_on=False)
            ax.add_patch(bar)
            label_text = label
            if col == 0:
                concept_label = ex['concept'].replace('_', ' ')
                label_text = f"{label}  —  prompt: \"{ex['prompt']}\""
                ax.text(0.01, 0.96, label_text, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", color="#222222", va="center")
                ax.text(-0.02, 0.5, concept_label.upper(),
                        transform=ax.transAxes,
                        fontsize=11, fontweight="bold",
                        rotation=90, ha="right", va="center",
                        color=cat_color)
            else:
                meta = f"{label}   (layer {ex['layer']}, coeff {ex['coeff']})"
                ax.text(0.01, 0.96, meta, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", color="white", va="center")

            text = _wrap(ex.get(key, ""), width=58)
            ax.text(0.02, 0.85, text,
                    transform=ax.transAxes,
                    fontsize=9.5, va="top", ha="left",
                    fontfamily="monospace", color="#222222",
                    wrap=True)

            # Subtle border around the cell
            border = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         fill=False, edgecolor="#dddddd", linewidth=1)
            ax.add_patch(border)

    fig.tight_layout()

    save_path = save_path or (config.VIZ_DIR / "qualitative_gallery.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gallery -> {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Run model and populate baseline/steered text.")
    parser.add_argument("--render", action="store_true",
                        help="Render gallery PNG from populated JSON.")
    args = parser.parse_args()
    if args.generate:
        generate_examples()
    if args.render or not args.generate:
        render_gallery()
