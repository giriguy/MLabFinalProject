"""
Generate supervised fine-tuning training data by collecting steered model outputs.

For each concept, runs the steered model over a diverse prompt set and saves
(user_prompt, assistant_response) pairs as JSONL for use with train_lora.py.

Usage:
    python generate_training_data.py --concepts technical_simple optimistic_pessimistic
    python generate_training_data.py --concepts all --coeff 15.0 --n-prompts 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import config
from compute_vectors import get_vector, load_vectors
from extraction import load_model_and_tokenizer
from scoring import CONCEPT_SCORERS, get_scorer
from validate import SteeringHook, generate_text

logger = logging.getLogger(__name__)

LORA_DATA_DIR = config.DATA_DIR / "lora_training"

# Diverse neutral prompts covering a wide range of topics and formats so the
# fine-tuned LoRA generalises beyond the concept-specific training examples.
DIVERSE_PROMPTS: List[str] = [
    # Science & Nature
    "Explain how photosynthesis works.",
    "What causes thunder and lightning?",
    "How do vaccines work?",
    "Describe the water cycle.",
    "What are black holes?",
    "Explain the theory of evolution.",
    "How does the human immune system work?",
    "What is quantum mechanics?",
    "Explain DNA and genetic inheritance.",
    "How do earthquakes occur?",
    "What causes the seasons to change?",
    "How does a nuclear reactor generate electricity?",
    "What is the greenhouse effect?",
    "Explain how batteries store and release energy.",
    "How do animals navigate during migration?",
    # Technology & Society
    "What is machine learning?",
    "How does social media affect mental health?",
    "What are the benefits and risks of artificial intelligence?",
    "How do I start learning programming?",
    "What is cryptocurrency and how does it work?",
    "Explain how GPS navigation works.",
    "What are renewable energy sources?",
    "How does facial recognition technology work?",
    "How do search engines like Google work?",
    "What is the impact of automation on jobs?",
    "Explain how the internet transmits data.",
    "What is cybersecurity and why does it matter?",
    "How do electric vehicles compare to petrol cars?",
    "What are the challenges of space exploration?",
    "How does 3D printing work?",
    # History & Culture
    "What caused World War I?",
    "Describe the significance of the French Revolution.",
    "What was the Space Race and why did it matter?",
    "Explain the causes of the Great Depression.",
    "What is the significance of the printing press?",
    "Describe the impact of the Industrial Revolution.",
    "What were the main causes of the fall of the Roman Empire?",
    "Explain the Civil Rights Movement in the United States.",
    "What is the significance of the Silk Road?",
    "Describe the impact of colonialism on the world.",
    "How did the Renaissance change European society?",
    "What was the Cold War?",
    # Health & Lifestyle
    "What are the benefits of regular exercise?",
    "How does sleep affect health and wellbeing?",
    "What constitutes a balanced diet?",
    "How does chronic stress affect the body?",
    "What are some evidence-based tips for improving mental health?",
    "How does meditation affect the brain?",
    "What are the long-term risks of smoking?",
    "How do antibiotics work and why is resistance a problem?",
    "What is mindfulness and how is it practised?",
    "Explain the importance of hydration for health.",
    "How does the gut microbiome affect overall health?",
    "What are the stages of grief?",
    # Philosophy & Society
    "What is the meaning of happiness?",
    "How should we think about the future of humanity?",
    "What is the nature of consciousness?",
    "Is technology making our lives better or worse overall?",
    "What is the value of a liberal arts education?",
    "How should we approach failure and setbacks?",
    "What makes a good leader?",
    "Why do people hold fundamentally different political views?",
    "How has globalisation changed society?",
    "What ethical questions does artificial intelligence raise?",
    "What is the relationship between freedom and responsibility?",
    "How do cultural differences shape our worldview?",
    # Practical & Everyday
    "How do I manage my time more effectively?",
    "What are practical strategies for saving money?",
    "How do I improve my writing skills?",
    "What makes a friendship last over time?",
    "How do I handle conflict constructively?",
    "What are effective study techniques backed by research?",
    "How do I build and sustain good habits?",
    "What are the basics of cooking healthy meals at home?",
    "What should I consider when starting a small business?",
    "How do I make better decisions under uncertainty?",
    # Open-ended / Speculative
    "What will cities look like in 50 years?",
    "What is the most important invention of the 20th century?",
    "How might humans explore deep space in the future?",
    "What lessons can modern societies learn from ancient civilisations?",
    "How do animals communicate with each other?",
    "What is creativity and how can we cultivate it?",
    "Describe what a truly sustainable economy might look like.",
    "What would it mean for humanity to achieve world peace?",
    "How might advances in biology change medicine in the next decade?",
    "What responsibilities do wealthy nations have toward poorer ones?",
]


def generate_dataset_for_concept(
    concept_name: str,
    model,
    tokenizer,
    vectors,
    concept_names: List[str],
    n_prompts: int = 200,
    coeff: float = config.DEFAULT_COEFF,
    layer_idx: Optional[int] = None,
    min_score: float = 0.6,
    seed: int = 42,
) -> List[dict]:
    """
    Generate steered (prompt, response) pairs for one concept.

    Returns a list of message dicts: {"messages": [user_turn, assistant_turn]}.
    """
    random.seed(seed)

    if layer_idx is None:
        layer_idx = config.N_LAYERS // 2

    vector = get_vector(concept_name, layer_idx, vectors, concept_names)
    hook = SteeringHook(model, vector, layer_idx=layer_idx, coeff=coeff)

    scorer = get_scorer(concept_name)
    # Concepts without a registered scorer return 0.5 always; don't filter them.
    has_scorer = concept_name in CONCEPT_SCORERS
    effective_min_score = min_score if has_scorer else 0.0

    # Sample prompts; repeat the pool if more are requested than available.
    if n_prompts <= len(DIVERSE_PROMPTS):
        prompts = random.sample(DIVERSE_PROMPTS, n_prompts)
    else:
        reps = n_prompts // len(DIVERSE_PROMPTS) + 1
        prompts = random.sample(DIVERSE_PROMPTS * reps, n_prompts)

    records: List[dict] = []
    skipped = 0
    for prompt in prompts:
        response = generate_text(prompt, model, tokenizer, steering_hook=hook)
        score = scorer(response)
        if score < effective_min_score:
            skipped += 1
            logger.debug(
                "Skipped (score=%.2f < %.2f): %r", score, effective_min_score, response[:80]
            )
            continue
        records.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })

    logger.info(
        "Concept %s: %d records kept, %d skipped (score < %.2f, layer=%d, coeff=%.1f)",
        concept_name, len(records), skipped, effective_min_score, layer_idx, coeff,
    )
    return records


def save_dataset(records: List[dict], concept_name: str, output_dir: Path = LORA_DATA_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{concept_name}.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Saved %d records -> %s", len(records), path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate LoRA training data via steered model generation"
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["technical_simple", "optimistic_pessimistic"],
        help="Concept names to generate data for, or 'all' for all 25.",
    )
    parser.add_argument(
        "--coeff", type=float, default=config.DEFAULT_COEFF,
        help="Steering coefficient (default: %(default)s).",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=200,
        help="Number of prompts to generate per concept (default: %(default)s).",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.6,
        help=(
            "Minimum scorer threshold to keep an example [0, 1]. "
            "Set to 0 to disable filtering. Only applies to concepts with a registered scorer."
        ),
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Injection layer index (default: N_LAYERS // 2).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=LORA_DATA_DIR,
        help="Directory to write JSONL files (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.concepts == ["all"]:
        concepts = config.CONCEPT_NAMES
    else:
        concepts = args.concepts

    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    for concept in concepts:
        if concept not in concept_names:
            logger.error(
                "Unknown concept '%s'. Available: %s", concept, concept_names
            )
            continue

        logger.info("=== Generating data for: %s ===", concept)
        records = generate_dataset_for_concept(
            concept,
            model,
            tokenizer,
            vectors,
            concept_names,
            n_prompts=args.n_prompts,
            coeff=args.coeff,
            layer_idx=args.layer,
            min_score=args.min_score,
        )

        if not records:
            logger.warning(
                "No records passed the score filter for '%s'. "
                "Try lowering --min-score or increasing --coeff.",
                concept,
            )
            continue

        save_dataset(records, concept, args.output_dir)


if __name__ == "__main__":
    main()
