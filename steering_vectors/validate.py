"""
Validate steering vectors by injecting them during generation and comparing outputs.

Injection: add `coeff * steering_vector[layer]` to the residual stream at the
chosen layer via a forward hook during autoregressive generation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import config
from compute_vectors import get_vector, load_vectors
from extraction import apply_chat_template, load_model_and_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """
    Injects a steering vector into a specific transformer layer during generation.

    Usage:
        with SteeringHook(model, vector, layer_idx=10, coeff=15.0):
            output = model.generate(...)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        vector: Tensor,
        layer_idx: int,
        coeff: float = config.DEFAULT_COEFF,
    ) -> None:
        self.vector = vector.to(model.device, dtype=next(model.parameters()).dtype)
        self.layer_idx = layer_idx
        self.coeff = coeff
        self._hook_handle = None
        self._model = model

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Inject at every sequence position during generation; the model sees
        # the full context on the first pass, then single tokens on subsequent passes.
        hidden = hidden + self.coeff * self.vector.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def _get_layer(self) -> torch.nn.Module:
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers[self.layer_idx]
        raise AttributeError(f"Cannot locate layer {self.layer_idx}")

    def __enter__(self) -> "SteeringHook":
        layer = self._get_layer()
        self._hook_handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *_) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    apply_template: bool = True,
    steering_hook: Optional[SteeringHook] = None,
) -> str:
    """Generate a completion for `prompt`, optionally with a SteeringHook active."""
    if apply_template:
        formatted = apply_chat_template([prompt], tokenizer)[0]
    else:
        formatted = prompt

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_INPUT_LENGTH,
    ).to(model.device)

    ctx = steering_hook if steering_hook is not None else _null_ctx()
    with ctx:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_steered_vs_baseline(
    prompt: str,
    concept_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: Optional[int] = None,
    coeff: float = config.DEFAULT_COEFF,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Generate baseline and steered outputs for a single prompt.

    Returns:
        (baseline_text, steered_text)
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    if layer_idx is None:
        # Default to middle layer — typically most effective for semantics
        layer_idx = config.N_LAYERS // 2

    vector = get_vector(concept_name, layer_idx, vectors, concept_names)

    baseline = generate_text(prompt, model, tokenizer)
    hook = SteeringHook(model, vector, layer_idx=layer_idx, coeff=coeff)
    steered = generate_text(prompt, model, tokenizer, steering_hook=hook)

    return baseline, steered


def validate_concept(
    concept_name: str,
    test_prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: Optional[int] = None,
    coeff: float = config.DEFAULT_COEFF,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> None:
    """
    Print side-by-side comparisons for each test prompt.
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    if layer_idx is None:
        layer_idx = config.N_LAYERS // 2

    print(f"\n{'='*70}")
    print(f"Steering concept: {concept_name!r}  |  layer={layer_idx}  coeff={coeff}")
    print(f"{'='*70}")

    for i, prompt in enumerate(test_prompts):
        baseline, steered = compare_steered_vs_baseline(
            prompt, concept_name, model, tokenizer,
            layer_idx=layer_idx, coeff=coeff,
            vectors=vectors, concept_names=concept_names,
        )
        print(f"\n[{i+1}] Prompt: {prompt!r}")
        print(f"  BASELINE : {baseline}")
        print(f"  STEERED  : {steered}")


def sweep_coefficients(
    prompt: str,
    concept_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    coeffs: List[float] = config.COEFF_SWEEP,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> List[Tuple[float, str]]:
    """
    Generate outputs at multiple coefficient strengths for a single prompt and layer.

    Returns:
        [(coeff, generated_text), ...]
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()

    vector = get_vector(concept_name, layer_idx, vectors, concept_names)
    results = []
    for coeff in coeffs:
        hook = SteeringHook(model, vector, layer_idx=layer_idx, coeff=coeff)
        text = generate_text(prompt, model, tokenizer, steering_hook=hook)
        results.append((coeff, text))
    return results


def sweep_layers(
    prompt: str,
    concept_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    coeff: float = config.DEFAULT_COEFF,
    layer_indices: Optional[List[int]] = None,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> List[Tuple[int, str]]:
    """
    Generate steered outputs at multiple injection layers for a fixed coefficient.

    Returns:
        [(layer_idx, generated_text), ...]
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()
    if layer_indices is None:
        layer_indices = config.LAYER_SWEEP

    results = []
    for layer_idx in layer_indices:
        vector = get_vector(concept_name, layer_idx, vectors, concept_names)
        hook = SteeringHook(model, vector, layer_idx=layer_idx, coeff=coeff)
        text = generate_text(prompt, model, tokenizer, steering_hook=hook)
        results.append((layer_idx, text))
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Validate steering vectors")
    parser.add_argument("--concept", required=True, help="Concept name, e.g. happy_sad")
    parser.add_argument("--layer", type=int, default=None, help="Injection layer (default: N_LAYERS//2)")
    parser.add_argument("--coeff", type=float, default=config.DEFAULT_COEFF)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Tell me about your day.", "What do you think about the future?"],
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()

    validate_concept(
        args.concept,
        args.prompts,
        model,
        tokenizer,
        layer_idx=args.layer,
        coeff=args.coeff,
        vectors=vectors,
        concept_names=concept_names,
    )
