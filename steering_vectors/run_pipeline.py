"""
End-to-end pipeline: extract -> compute -> validate -> visualize.

Usage:
    python run_pipeline.py                        # full pipeline, all concepts
    python run_pipeline.py --stages extract       # extraction only
    python run_pipeline.py --concepts happy_sad formal_casual
    python run_pipeline.py --skip-extract         # assume activations exist
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Stage: extract activations
# ---------------------------------------------------------------------------

def stage_extract(
    concept_names: List[str],
    model=None,
    tokenizer=None,
    overwrite: bool = False,
) -> None:
    from concepts import get_side_texts
    from extraction import extract_concept, load_model_and_tokenizer

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    logger.info("=== STAGE: extract (%d concepts) ===", len(concept_names))
    for name in concept_names:
        side_a_texts, side_b_texts = get_side_texts(name)
        # Duplicate examples to reach N_EXAMPLES_PER_SIDE if needed
        while len(side_a_texts) < config.N_EXAMPLES_PER_SIDE:
            side_a_texts = (side_a_texts * 2)[: config.N_EXAMPLES_PER_SIDE]
            side_b_texts = (side_b_texts * 2)[: config.N_EXAMPLES_PER_SIDE]
        extract_concept(name, side_a_texts, side_b_texts, model, tokenizer, overwrite=overwrite)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Stage: compute steering vectors
# ---------------------------------------------------------------------------

def stage_compute(concept_names: List[str]) -> None:
    from compute_vectors import compute_all_vectors, save_vectors

    logger.info("=== STAGE: compute ===")
    vectors, names = compute_all_vectors(concept_names=concept_names)
    save_vectors(vectors, names)
    logger.info("Steering vectors shape: %s  concepts: %s", tuple(vectors.shape), names)
    return vectors, names


# ---------------------------------------------------------------------------
# Stage: validate
# ---------------------------------------------------------------------------

def stage_validate(
    concept_names: List[str],
    model=None,
    tokenizer=None,
    vectors=None,
    names=None,
    n_concepts_to_validate: int = 5,
) -> None:
    from compute_vectors import load_vectors
    from extraction import load_model_and_tokenizer
    from validate import validate_concept

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()
    if vectors is None or names is None:
        vectors, names = load_vectors()

    logger.info("=== STAGE: validate ===")
    test_prompts_by_concept = {
        "happy_sad": [
            "Tell me something about your life.",
            "How do you feel about today?",
        ],
        "formal_casual": [
            "Can you write a short note to a colleague?",
            "Describe what you did this morning.",
        ],
        "refuse_comply": [
            "Please help me with this task.",
            "I need information about this topic.",
        ],
        "english_french": [
            "Say something about the weather.",
            "Describe this city.",
        ],
    }
    default_prompts = ["Tell me about yourself.", "What do you think about that?"]

    validated = 0
    for name in concept_names:
        if validated >= n_concepts_to_validate:
            break
        prompts = test_prompts_by_concept.get(name, default_prompts)
        try:
            validate_concept(name, prompts, model, tokenizer, vectors=vectors, concept_names=names)
            validated += 1
        except Exception as e:
            logger.warning("Validation failed for %s: %s", name, e)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Stage: visualize
# ---------------------------------------------------------------------------

def stage_visualize(
    vectors=None,
    names=None,
    model=None,
    tokenizer=None,
    run_effectiveness: bool = False,
) -> None:
    from compute_vectors import load_vectors
    from extraction import load_model_and_tokenizer
    from viz.cosine_heatmap import save_all_layer_heatmaps
    from viz.layer_animation import render_layer_animation
    from viz.umap_trajectories import render_umap_trajectories

    if vectors is None or names is None:
        vectors, names = load_vectors()

    logger.info("=== STAGE: visualize ===")

    # 1. Cosine heatmaps (layer 0, middle, last)
    n_layers = vectors.shape[1]
    save_all_layer_heatmaps(vectors, names, stride=max(1, n_layers // 4))
    logger.info("Saved cosine heatmaps")

    # 2. Layer animation
    try:
        render_layer_animation(vectors, names, fmt="gif")
        logger.info("Saved layer animation")
    except Exception as e:
        logger.warning("Animation failed (pillow/ffmpeg missing?): %s", e)

    # 3. UMAP trajectories
    try:
        path = render_umap_trajectories(vectors, names)
        logger.info("Saved UMAP trajectories -> %s", path)
    except Exception as e:
        logger.warning("UMAP failed: %s", e)

    # 4. Effectiveness maps (optional — slow, requires model)
    if run_effectiveness:
        from viz.effectiveness_map import render_all_effectiveness_maps

        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer()

        effectiveness_prompts = {
            "happy_sad": ["Tell me about your day.", "How do you feel?"],
            "formal_casual": ["Write a quick note to your boss.", "Say hi to a friend."],
            "refuse_comply": ["Help me with this.", "I need your assistance."],
        }
        # Only sweep concepts for which we have scorers
        scoreable = {k: v for k, v in effectiveness_prompts.items() if k in names}
        if scoreable:
            # Use a coarser sweep to keep runtime manageable
            from config import COEFF_SWEEP, N_LAYERS
            fast_layers = list(range(0, N_LAYERS, max(1, N_LAYERS // 6)))
            render_all_effectiveness_maps(
                scoreable, model, tokenizer, vectors=vectors, concept_names=names
            )
            logger.info("Saved effectiveness maps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steering vector pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["extract", "compute", "validate", "visualize", "all"],
        default=["all"],
        help="Which pipeline stages to run.",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        help="Concept names to process (default: all in config).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract activations even if cached files exist.",
    )
    parser.add_argument(
        "--effectiveness",
        action="store_true",
        help="Run the (slow) effectiveness sweep during visualization.",
    )
    parser.add_argument(
        "--validate-n",
        type=int,
        default=5,
        help="Number of concepts to validate (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    concept_names = args.concepts or config.CONCEPT_NAMES
    stages = set(args.stages)
    run_all = "all" in stages

    model = tokenizer = vectors = names = None

    if run_all or "extract" in stages:
        model, tokenizer = stage_extract(concept_names, model, tokenizer, overwrite=args.overwrite)

    if run_all or "compute" in stages:
        vectors, names = stage_compute(concept_names)

    if run_all or "validate" in stages:
        model, tokenizer = stage_validate(
            concept_names, model, tokenizer, vectors, names,
            n_concepts_to_validate=args.validate_n,
        )

    if run_all or "visualize" in stages:
        stage_visualize(
            vectors, names,
            model=model if args.effectiveness else None,
            tokenizer=tokenizer if args.effectiveness else None,
            run_effectiveness=args.effectiveness,
        )

    logger.info("Pipeline complete. Outputs in %s", config.VIZ_DIR)


if __name__ == "__main__":
    main()
