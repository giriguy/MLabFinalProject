"""
Simple, fast scorers for measuring steering effectiveness.

Each scorer takes a list of generated texts and returns a float in [0, 1]
where 1 means "clearly side-A of the concept" and 0 means "clearly side-B".
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Keyword / regex scorers
# ---------------------------------------------------------------------------

REFUSAL_PHRASES = [
    r"I (cannot|can't|am unable to|won't|will not|must (refuse|decline))",
    r"(that|this) (request|question|prompt) (is|violates|goes against)",
    r"(not (able|in a position)|unable) to (help|assist|provide)",
    r"I (must|have to) (decline|refuse|stop)",
    r"against my (guidelines|values|ethics|training)",
]

COMPLY_PHRASES = [
    r"(Sure|Of course|Absolutely|Certainly|Happy to|I'd be glad|No problem)",
    r"(Here( i|'| i)s|Here you go|Here are)",
    r"Let me (help|assist|explain|show)",
]

FRENCH_WORDS = [
    r"\b(le|la|les|du|des|un|une|et|est|sont|avec|pour|dans|sur|par)\b",
    r"\b(je|tu|il|elle|nous|vous|ils|elles)\b",
    r"\b(bonjour|merci|s'il vous plaît|oui|non|voilà)\b",
]

FORMAL_MARKERS = [
    r"\b(hereby|henceforth|aforementioned|pursuant|notwithstanding|whereby)\b",
    r"\b(I would like to|I am writing to|Please be advised|Kindly)\b",
    r"\b(sincerely|respectfully|regards|yours truly)\b",
]

CASUAL_MARKERS = [
    r"\b(hey|hi|yeah|yep|nope|gonna|wanna|gotta|kinda|sorta)\b",
    r"\b(ok|okay|sure thing|no worries|cool|awesome|totally)\b",
    r"(!{2,}|\.{3,}|\?!)",
]

UNCERTAINTY_MARKERS = [
    r"\b(maybe|perhaps|possibly|might|could|seems?|appears?|likely)\b",
    r"\b(I('m| am) not (sure|certain|entirely)|unclear|uncertain)\b",
    r"\b(it('s| is) (possible|probable|hard to say))\b",
]

CERTAINTY_MARKERS = [
    r"\b(definitely|certainly|absolutely|clearly|undoubtedly|without (a )?doubt)\b",
    r"\b(I (am|'m) (sure|certain|confident)|I know|the fact is)\b",
    r"\b(conclusively|definitively|unquestionably|unambiguously)\b",
]


def regex_score(text: str, positive_patterns: List[str], negative_patterns: List[str]) -> float:
    """
    Score a text based on regex matches.

    Returns:
        Score in [0, 1] where positive matches push toward 1, negative toward 0.
        Returns 0.5 when no patterns match (neutral/ambiguous).
    """
    text_lower = text.lower()
    pos_hits = sum(1 for p in positive_patterns if re.search(p, text_lower, re.IGNORECASE))
    neg_hits = sum(1 for p in negative_patterns if re.search(p, text_lower, re.IGNORECASE))
    total = pos_hits + neg_hits
    if total == 0:
        return 0.5
    return pos_hits / total


def score_refusal(text: str) -> float:
    """1.0 = clearly a refusal, 0.0 = clearly compliance."""
    return regex_score(text, REFUSAL_PHRASES, COMPLY_PHRASES)


def score_formal(text: str) -> float:
    """1.0 = formal, 0.0 = casual."""
    return regex_score(text, FORMAL_MARKERS, CASUAL_MARKERS)


def score_certainty(text: str) -> float:
    """1.0 = certain, 0.0 = uncertain."""
    return regex_score(text, CERTAINTY_MARKERS, UNCERTAINTY_MARKERS)


def score_french(text: str) -> float:
    """1.0 = French, 0.0 = English (based on French function word density)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    french_hits = sum(
        1 for w in words
        if re.match(r"^(le|la|les|du|des|un|une|et|est|sont|avec|pour|dans|sur|par|je|tu|il|elle|nous|vous|ils|elles)$", w)
    )
    return min(1.0, french_hits / max(1, len(words)) * 10)  # scale: ~10% French words -> score 1.0


def score_length(text: str, long_threshold: int = 200, short_threshold: int = 50) -> float:
    """1.0 = long, 0.0 = short."""
    n = len(text)
    if n >= long_threshold:
        return 1.0
    if n <= short_threshold:
        return 0.0
    return (n - short_threshold) / (long_threshold - short_threshold)


# ---------------------------------------------------------------------------
# Sentiment scorer (lightweight, no heavy deps)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = [
    "good", "great", "excellent", "wonderful", "happy", "joy", "love", "amazing",
    "fantastic", "thrilled", "excited", "beautiful", "perfect", "best", "brilliant",
    "delighted", "pleased", "grateful", "positive", "success", "win",
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "horrible", "sad", "hate", "angry", "disgusting",
    "worst", "failure", "disaster", "depressed", "miserable", "dreadful", "painful",
    "hopeless", "devastating", "tragic", "ruined", "broken",
]


def score_sentiment(text: str) -> float:
    """
    Simple lexicon-based sentiment: 1.0 = positive, 0.0 = negative.
    Falls back to a HuggingFace pipeline if available.
    """
    try:
        return _hf_sentiment_score(text)
    except Exception:
        return _lexicon_sentiment_score(text)


def _lexicon_sentiment_score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    return (pos / total) if total > 0 else 0.5


_hf_pipeline = None


def _hf_sentiment_score(text: str) -> float:
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # CPU always to avoid interfering with main model
        )
    result = _hf_pipeline(text[:512])[0]
    score = result["score"]
    return score if result["label"] == "POSITIVE" else 1.0 - score


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

# Maps concept name -> scorer function. Scorer returns float in [0,1]
# where 1 = side A of the concept is present.
CONCEPT_SCORERS: Dict[str, Callable[[str], float]] = {
    "formal_casual": score_formal,
    "certain_uncertain": score_certainty,
    "refuse_comply": score_refusal,
    "english_french": lambda t: 1.0 - score_french(t),  # 1 = English
    "happy_sad": score_sentiment,
    "positive_negative": score_sentiment,
    "optimistic_pessimistic": score_sentiment,
    "verbose_concise": score_length,
    "long_short": score_length,
    "emotional_neutral": lambda t: 1.0 - score_certainty(t),  # emotional = less measured
}


def get_scorer(concept_name: str) -> Callable[[str], float]:
    """
    Return the scorer for a concept, or a neutral scorer if not registered.
    """
    return CONCEPT_SCORERS.get(concept_name, lambda _: 0.5)


def score_batch(
    texts: List[str],
    concept_name: str,
) -> List[float]:
    """Score a batch of generated texts for a concept."""
    scorer = get_scorer(concept_name)
    return [scorer(t) for t in texts]


def effectiveness_score(
    baseline_texts: List[str],
    steered_texts: List[str],
    concept_name: str,
) -> Tuple[float, float, float]:
    """
    Compare baseline vs steered generation to measure steering effectiveness.

    Returns:
        (baseline_mean, steered_mean, delta)  — delta > 0 means steering pushed toward side A.
    """
    baseline_scores = score_batch(baseline_texts, concept_name)
    steered_scores = score_batch(steered_texts, concept_name)
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    steered_mean = sum(steered_scores) / len(steered_scores)
    return baseline_mean, steered_mean, steered_mean - baseline_mean
