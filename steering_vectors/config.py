from pathlib import Path
from typing import List, Tuple

import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# float16 on CUDA, float32 elsewhere (mps has spotty fp16 support)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Qwen2.5-1.5B architecture defaults; overridden at runtime after model load
N_LAYERS = 28
D_MODEL = 1536

BATCH_SIZE = 8
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 150
N_EXAMPLES_PER_SIDE = 20  # prompt examples per concept side for vector computation

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"
VECTORS_DIR = DATA_DIR / "vectors"
OUTPUTS_DIR = DATA_DIR / "outputs"
VIZ_DIR = BASE_DIR / "viz" / "outputs"

for _d in [ACTIVATIONS_DIR, VECTORS_DIR, OUTPUTS_DIR, VIZ_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# (concept_name, side_A_label, side_B_label)
CONCEPT_PAIRS: List[Tuple[str, str, str]] = [
    ("formal_casual", "formal", "casual"),
    ("certain_uncertain", "certain", "uncertain"),
    ("polite_rude", "polite", "rude"),
    ("english_french", "english", "french"),
    ("happy_sad", "happy", "sad"),
    ("refuse_comply", "refuse", "comply"),
    ("technical_simple", "technical", "simple"),
    ("verbose_concise", "verbose", "concise"),
    ("past_present", "past_tense", "present_tense"),
    ("positive_negative", "positive", "negative"),
    ("specific_vague", "specific", "vague"),
    ("optimistic_pessimistic", "optimistic", "pessimistic"),
    ("assertive_hesitant", "assertive", "hesitant"),
    ("scientific_colloquial", "scientific", "colloquial"),
    ("emotional_neutral", "emotional", "neutral"),
    ("first_third_person", "first_person", "third_person"),
    ("instructional_narrative", "instructional", "narrative"),
    ("urgent_calm", "urgent", "calm"),
    ("hypothetical_factual", "hypothetical", "factual"),
    ("creative_literal", "creative", "literal"),
    ("long_short", "long_form", "short_form"),
    ("question_statement", "question", "statement"),
    ("active_passive", "active_voice", "passive_voice"),
    ("inclusive_exclusive", "inclusive", "exclusive"),
    ("future_past", "future_oriented", "past_oriented"),
]

CONCEPT_NAMES: List[str] = [name for name, _, _ in CONCEPT_PAIRS]

# Semantic grouping for visualization. Eight categories mapped to a
# colorblind-safe Okabe-Ito palette (used across the poster panels).
CONCEPT_CATEGORIES: dict = {
    "formal_casual": "Pragmatics",
    "polite_rude": "Pragmatics",
    "urgent_calm": "Pragmatics",
    "question_statement": "Pragmatics",
    "positive_negative": "Sentiment",
    "happy_sad": "Sentiment",
    "optimistic_pessimistic": "Sentiment",
    "emotional_neutral": "Sentiment",
    "active_passive": "Syntax",
    "past_present": "Syntax",
    "first_third_person": "Syntax",
    "technical_simple": "Register",
    "scientific_colloquial": "Register",
    "english_french": "Register",
    "verbose_concise": "Style",
    "long_short": "Style",
    "specific_vague": "Style",
    "creative_literal": "Style",
    "instructional_narrative": "Content",
    "hypothetical_factual": "Content",
    "refuse_comply": "Intent",
    "certain_uncertain": "Intent",
    "assertive_hesitant": "Intent",
    "inclusive_exclusive": "Scope",
    "future_past": "Scope",
}

CATEGORY_ORDER: List[str] = [
    "Pragmatics",
    "Sentiment",
    "Syntax",
    "Register",
    "Style",
    "Content",
    "Intent",
    "Scope",
]

# Okabe-Ito palette (colorblind-safe, 8 colors). Used categorically.
CATEGORY_COLORS: dict = {
    "Pragmatics": "#E69F00",
    "Sentiment": "#56B4E9",
    "Syntax": "#009E73",
    "Register": "#F0E442",
    "Style": "#0072B2",
    "Content": "#D55E00",
    "Intent": "#CC79A7",
    "Scope": "#999999",
}


def category_for(concept_name: str) -> str:
    return CONCEPT_CATEGORIES.get(concept_name, "Other")


def color_for(concept_name: str) -> str:
    return CATEGORY_COLORS.get(category_for(concept_name), "#888888")


# Steering sweep parameters
DEFAULT_COEFF = 15.0
COEFF_SWEEP = [5.0, 10.0, 15.0, 20.0, 30.0]
# Sweep every other layer; full sweep is expensive
LAYER_SWEEP = list(range(0, N_LAYERS, 2))

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# Scoring thresholds
SENTIMENT_BATCH_SIZE = 32
