"""
Forward-pass hook infrastructure for residual stream activation extraction.

Design:
  - Register a post-hook on each decoder layer (model.model.layers[i]).
  - The hook captures the hidden state output at the last non-padding token position.
  - Activations are saved as a memory-mapped numpy array of shape
    (n_prompts, n_layers, d_model) per concept side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = config.MODEL_NAME,
    device: str = config.DEVICE,
    dtype: torch.dtype = config.DTYPE,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    logger.info("Loading model %s on %s (%s)", model_name, device, dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning("Failed to load %s (%s), trying fallback %s", model_name, e, config.FALLBACK_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(config.FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.FALLBACK_MODEL,
            torch_dtype=dtype,
            device_map=device,
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad so last real token is always at position -1

    # Update global architecture constants from the loaded model
    config.N_LAYERS = model.config.num_hidden_layers
    config.D_MODEL = model.config.hidden_size
    config.LAYER_SWEEP = list(range(0, config.N_LAYERS, 2))
    logger.info("Model has %d layers, hidden_size=%d", config.N_LAYERS, config.D_MODEL)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Hook infrastructure
# ---------------------------------------------------------------------------

class ActivationCache:
    """Accumulates per-layer residual stream outputs during a forward pass."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        # layer_idx -> Tensor(batch, seq, d_model)
        self._cache: Dict[int, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def register(self, model: PreTrainedModel) -> "ActivationCache":
        """Attach hooks to every transformer block."""
        layers = _get_layers(model)
        if len(layers) != self.n_layers:
            logger.warning("Expected %d layers, found %d", self.n_layers, len(layers))

        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)
        return self

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._cache.clear()

    def __enter__(self) -> "ActivationCache":
        return self

    def __exit__(self, *_) -> None:
        self.remove()

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # HuggingFace decoder layers return a tuple; first element is hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Detach and move to CPU immediately to avoid accumulating GPU tensors
            self._cache[layer_idx] = hidden.detach().cpu()
        return hook

    def get_last_token_activations(
        self,
        attention_mask: Tensor,
        n_layers: Optional[int] = None,
    ) -> Tensor:
        """
        Extract the residual stream at the last non-padding token for each example.

        Args:
            attention_mask: (batch, seq) — 1 for real tokens, 0 for padding.
            n_layers: How many layers to include (default: all).

        Returns:
            Tensor of shape (batch, n_layers, d_model).
        """
        n_layers = n_layers or self.n_layers
        outputs = []
        for layer_idx in range(n_layers):
            hidden = self._cache[layer_idx]  # (batch, seq, d_model) — already on CPU
            # With left-padding, position -1 is always the last real token for every example.
            token_hidden = hidden[:, -1, :]  # (batch, d_model)
            outputs.append(token_hidden)

        return torch.stack(outputs, dim=1)  # (batch, n_layers, d_model)


def _get_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Return the list of transformer decoder blocks regardless of model family."""
    # Qwen2 / LLaMA / Mistral all expose model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback for GPT-NeoX style
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def apply_chat_template(texts: List[str], tokenizer: PreTrainedTokenizerBase) -> List[str]:
    """
    Wrap each text in the model's chat template as a user message.
    If the tokenizer has no chat template, returns texts unchanged.
    """
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return texts
    formatted = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        # add_generation_prompt=True appends the assistant turn opening token
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(prompt)
    return formatted


def tokenize_batch(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    device: str = config.DEVICE,
    max_length: int = config.MAX_INPUT_LENGTH,
) -> Dict[str, Tensor]:
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations_for_texts(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = config.BATCH_SIZE,
    apply_template: bool = True,
) -> np.ndarray:
    """
    Run forward passes and collect residual stream activations.

    Args:
        texts: Raw prompt strings.
        model: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        batch_size: Prompts per forward pass.
        apply_template: Whether to wrap each text in the chat template.

    Returns:
        numpy array of shape (n_texts, n_layers, d_model), float32.
    """
    n_layers = config.N_LAYERS
    if apply_template:
        texts = apply_chat_template(texts, tokenizer)

    all_activations: List[np.ndarray] = []
    cache = ActivationCache(n_layers)
    cache.register(model)

    try:
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[batch_start : batch_start + batch_size]
            inputs = tokenize_batch(batch_texts, tokenizer)

            model(**inputs)

            batch_acts = cache.get_last_token_activations(
                inputs["attention_mask"], n_layers=n_layers
            )
            all_activations.append(batch_acts.float().numpy())
            cache._cache.clear()  # free memory between batches

    finally:
        cache.remove()

    return np.concatenate(all_activations, axis=0)  # (n_texts, n_layers, d_model)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def activations_path(concept_name: str, side: str) -> Path:
    return config.ACTIVATIONS_DIR / f"{concept_name}__{side}.npy"


def save_activations(
    activations: np.ndarray,
    concept_name: str,
    side: str,
) -> Path:
    """Save activations as a memory-mapped .npy file."""
    path = activations_path(concept_name, side)
    np.save(path, activations.astype(np.float32))
    logger.info("Saved activations %s -> %s %s", activations.shape, path.name, side)
    return path


def load_activations(concept_name: str, side: str) -> np.ndarray:
    """Load activations as a read-only memory map."""
    path = activations_path(concept_name, side)
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")
    return np.load(path, mmap_mode="r")


def activations_exist(concept_name: str, side: str) -> bool:
    return activations_path(concept_name, side).exists()


# ---------------------------------------------------------------------------
# High-level extraction pipeline
# ---------------------------------------------------------------------------

def extract_concept(
    concept_name: str,
    side_a_texts: List[str],
    side_b_texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and save activations for both sides of a concept.

    Returns:
        (acts_a, acts_b) each of shape (n_texts, n_layers, d_model).
    """
    _, side_a_label, side_b_label = _find_concept_labels(concept_name)

    results = []
    for label, texts in [(side_a_label, side_a_texts), (side_b_label, side_b_texts)]:
        if not overwrite and activations_exist(concept_name, label):
            logger.info("Loading cached activations for %s/%s", concept_name, label)
            acts = load_activations(concept_name, label)
        else:
            logger.info("Extracting activations for %s/%s (%d texts)", concept_name, label, len(texts))
            acts = extract_activations_for_texts(texts, model, tokenizer)
            save_activations(acts, concept_name, label)
        results.append(acts)

    return results[0], results[1]


def extract_all_concepts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    concept_pairs_dict: Dict[str, Tuple[List[str], List[str]]],
    overwrite: bool = False,
) -> None:
    """Extract activations for every concept in concept_pairs_dict."""
    for concept_name, (side_a_texts, side_b_texts) in concept_pairs_dict.items():
        logger.info("--- Concept: %s ---", concept_name)
        extract_concept(
            concept_name, side_a_texts, side_b_texts, model, tokenizer, overwrite=overwrite
        )


def _find_concept_labels(concept_name: str) -> Tuple[str, str, str]:
    """Return (name, side_a_label, side_b_label) from config.CONCEPT_PAIRS."""
    for name, a, b in config.CONCEPT_PAIRS:
        if name == concept_name:
            return name, a, b
    raise KeyError(f"Concept '{concept_name}' not in config.CONCEPT_PAIRS")
