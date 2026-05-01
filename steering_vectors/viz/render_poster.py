"""
Render every poster panel in one shot.

Run from /steering_vectors:
    python viz/render_poster.py            # render all that don't need inference
    python viz/render_poster.py --all      # also run inference-heavy panels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def render_no_inference() -> None:
    from viz.category_atlas import render_concept_atlas
    from viz.cosine_heatmap import render_poster_heatmap
    from viz.umap_trajectories import render_umap_trajectories
    from viz.layer_profile import render_layer_profile
    from viz.lora_analysis import (
        plot_training_loss,
        plot_weight_analysis,
        _find_trained_adapters,
        OUT_DIR,
    )
    from viz.effectiveness_map import render_three_concept_grid_from_cached
    from viz.qualitative_gallery import render_gallery

    print("\n[1/7] Concept atlas...")
    render_concept_atlas()

    print("\n[2/7] Cosine heatmap (layer 14)...")
    render_poster_heatmap(layer_idx=14)

    print("\n[3/7] UMAP trajectories...")
    render_umap_trajectories()

    print("\n[4/7] Effectiveness grid (from cache, if available)...")
    render_three_concept_grid_from_cached()

    print("\n[5/7] Layer profile (separability)...")
    render_layer_profile(mode="norm")
    print("\n[5b/7] Layer profile (effectiveness, from cache)...")
    render_layer_profile(mode="effectiveness")

    adapters = _find_trained_adapters()
    if adapters:
        print("\n[6a/7] LoRA training loss...")
        plot_training_loss(adapters, OUT_DIR / "lora_training_loss.png")
        print("\n[6b/7] LoRA weight analysis...")
        plot_weight_analysis(adapters, OUT_DIR / "lora_weight_analysis.png")
    else:
        print("\n[6/7] No LoRA adapters found, skipping.")

    print("\n[7/7] Qualitative gallery (from JSON, if populated)...")
    render_gallery()

    # Controllability cluster — render from cache if available
    print("\n[C1] Controllability curves (from cache)...")
    try:
        import subprocess
        subprocess.run(
            ["python", "viz/controllability_curves.py", "--from-cache"],
            cwd=str(Path(__file__).parent.parent), check=False,
        )
    except Exception as e:
        print(f"  skipped: {e}")

    print("\n[C2] Concept radar (from cache)...")
    try:
        import subprocess
        subprocess.run(
            ["python", "viz/concept_radar.py", "--from-cache"],
            cwd=str(Path(__file__).parent.parent), check=False,
        )
    except Exception as e:
        print(f"  skipped: {e}")

    print("\n[C3] Composition heatmap (from cache)...")
    try:
        import subprocess
        subprocess.run(
            ["python", "viz/composition_heatmap.py", "--from-cache"],
            cwd=str(Path(__file__).parent.parent), check=False,
        )
    except Exception as e:
        print(f"  skipped: {e}")

    print(f"\nAll outputs in {config.VIZ_DIR}")


def render_with_inference() -> None:
    """Inference-heavy panels: effectiveness sweep, LoRA score comparison, controllability cluster."""
    from extraction import load_model_and_tokenizer
    from compute_vectors import load_vectors
    from viz.effectiveness_map import render_three_concept_grid
    from viz.lora_analysis import plot_score_comparison, _find_trained_adapters, OUT_DIR
    from viz.qualitative_gallery import generate_examples, render_gallery
    from viz import controllability_curves as cc
    from viz import concept_radar as cr
    from viz import composition_heatmap as ch

    print("\n>>> Loading model once for all inference panels...")
    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    print("\n[I-1] Effectiveness sweep (3 concepts × layers × coeffs)...")
    render_three_concept_grid(model, tokenizer,
                              vectors=vectors, concept_names=concept_names)

    print("\n[I-2] Steering Dial controllability curves (C1)...")
    flagships = ["formal_casual", "optimistic_pessimistic", "english_french"]
    all_curves = {}
    for c in flagships:
        results, coeffs = cc.run_for_concept(
            c, model, tokenizer, layer=14,
            vectors=vectors, concept_names=concept_names,
        )
        all_curves[c] = results
        cc.render_curves(c, results, coeffs)
    cc.render_curves_grid(flagships, all_curves, coeffs)

    print("\n[I-3] Concept Radar (C2)...")
    radar_data = cr.gather_data(model, tokenizer, layer=14)
    cr.render_radar_grid(radar_data)

    print("\n[I-4] Composition Heatmap (C3)...")
    pairs_data = []
    for va, vb, tgt in ch.DEFAULT_PAIRS:
        grid = ch.run_pair(va, vb, tgt, model, tokenizer, layer=14,
                           vectors=vectors, concept_names=concept_names)
        pairs_data.append((va, vb, tgt, grid))
    ch.render_composition_grid(pairs_data, ch.DEFAULT_COEFFS)

    print("\n[I-5] Generating qualitative gallery examples...")
    generate_examples()
    render_gallery()

    adapters = _find_trained_adapters()
    if adapters:
        print("\n[I-6] LoRA score comparison (with CIs)...")
        plot_score_comparison(
            adapters, OUT_DIR / "lora_score_comparison.png",
            n_eval=32, coeff=8.0,
        )
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Also run inference-heavy panels (loads model).")
    args = parser.parse_args()

    if args.all:
        render_with_inference()
    render_no_inference()
