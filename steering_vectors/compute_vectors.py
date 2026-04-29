"""
Compute L2-normalized mean-difference steering vectors from saved activations.

For each concept pair at each layer:
    vector[concept, layer] = normalize(mean(acts_A) - mean(acts_B))

Output tensor shape: (n_concepts, n_layers, d_model)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import config
from extraction import load_activations, _find_concept_labels

logger = logging.getLogger(__name__)

VECTORS_FILE = config.VECTORS_DIR / "steering_vectors.pt"
METADATA_FILE = config.VECTORS_DIR / "metadata.pt"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_mean_diff(acts_a: np.ndarray, acts_b: np.ndarray) -> np.ndarray:
    """
    Compute mean-difference vectors per layer.

    Args:
        acts_a: (n_a, n_layers, d_model)
        acts_b: (n_b, n_layers, d_model)

    Returns:
        (n_layers, d_model) float32 — raw (unnormalized) mean differences.
    """
    mean_a = acts_a.mean(axis=0)  # (n_layers, d_model)
    mean_b = acts_b.mean(axis=0)
    return (mean_a - mean_b).astype(np.float32)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize along the last axis.

    Args:
        vectors: (..., d_model)
    Returns:
        Unit vectors of the same shape.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + eps)


def compute_vector_for_concept(
    concept_name: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load saved activations and compute the steering vector for one concept.

    Returns:
        (n_layers, d_model) float32.
    """
    _, side_a_label, side_b_label = _find_concept_labels(concept_name)
    acts_a = np.array(load_activations(concept_name, side_a_label), dtype=np.float32)
    acts_b = np.array(load_activations(concept_name, side_b_label), dtype=np.float32)

    vectors = compute_mean_diff(acts_a, acts_b)  # (n_layers, d_model)
    if normalize:
        vectors = l2_normalize(vectors)
    logger.info(
        "Computed vectors for %s: shape=%s, norm[L15]=%f",
        concept_name,
        vectors.shape,
        float(np.linalg.norm(vectors[min(15, len(vectors) - 1)])),
    )
    return vectors


def compute_all_vectors(
    concept_names: Optional[List[str]] = None,
    normalize: bool = True,
) -> Tuple[Tensor, List[str]]:
    """
    Compute steering vectors for all (or specified) concepts.

    Returns:
        vectors:  Tensor of shape (n_concepts, n_layers, d_model)
        names:    List of concept names in the same order as the first dimension.
    """
    if concept_names is None:
        concept_names = config.CONCEPT_NAMES

    all_vectors = []
    successful_names = []

    for name in concept_names:
        try:
            v = compute_vector_for_concept(name, normalize=normalize)
            all_vectors.append(v)
            successful_names.append(name)
        except FileNotFoundError as e:
            logger.warning("Skipping %s — activations not found (%s)", name, e)

    if not all_vectors:
        raise RuntimeError("No activation files found. Run extraction first.")

    stacked = torch.from_numpy(np.stack(all_vectors, axis=0))  # (n_concepts, n_layers, d_model)
    return stacked, successful_names


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vectors(
    vectors: Tensor,
    concept_names: List[str],
    path: Path = VECTORS_FILE,
    metadata_path: Path = METADATA_FILE,
) -> None:
    """Save the steering vector tensor and metadata."""
    torch.save(vectors, path)
    torch.save(
        {
            "concept_names": concept_names,
            "n_concepts": vectors.shape[0],
            "n_layers": vectors.shape[1],
            "d_model": vectors.shape[2],
        },
        metadata_path,
    )
    logger.info("Saved vectors %s -> %s", tuple(vectors.shape), path)


def load_vectors(
    path: Path = VECTORS_FILE,
    metadata_path: Path = METADATA_FILE,
) -> Tuple[Tensor, List[str]]:
    """Load the steering vector tensor and concept name list."""
    if not path.exists():
        raise FileNotFoundError(f"Vectors file not found: {path}. Run compute_vectors first.")
    vectors = torch.load(path, map_location="cpu", weights_only=True)
    meta = torch.load(metadata_path, map_location="cpu", weights_only=True)
    return vectors, meta["concept_names"]


def get_vector(
    concept_name: str,
    layer: int,
    vectors: Optional[Tensor] = None,
    concept_names: Optional[List[str]] = None,
) -> Tensor:
    """
    Retrieve the steering vector for a specific concept and layer.

    Args:
        concept_name: e.g. "happy_sad"
        layer: Transformer layer index.
        vectors: Pre-loaded tensor (loads from disk if None).
        concept_names: Corresponding name list (loads from disk if None).

    Returns:
        Tensor of shape (d_model,).
    """
    if vectors is None or concept_names is None:
        vectors, concept_names = load_vectors()
    if concept_name not in concept_names:
        raise KeyError(f"Concept '{concept_name}' not found. Available: {concept_names}")
    concept_idx = concept_names.index(concept_name)
    return vectors[concept_idx, layer]  # (d_model,)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(vectors_at_layer: Tensor) -> Tensor:
    """
    Compute pairwise cosine similarity between concept vectors at a single layer.

    Args:
        vectors_at_layer: (n_concepts, d_model) — already L2-normalized.

    Returns:
        (n_concepts, n_concepts) similarity matrix.
    """
    # Vectors should already be unit norm, but normalize defensively
    norms = vectors_at_layer.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = vectors_at_layer / norms
    return torch.mm(normalized, normalized.t())


def layer_similarity_matrices(vectors: Tensor) -> Tensor:
    """
    Compute cosine similarity matrices across all layers.

    Args:
        vectors: (n_concepts, n_layers, d_model)

    Returns:
        (n_layers, n_concepts, n_concepts)
    """
    n_layers = vectors.shape[1]
    matrices = []
    for layer in range(n_layers):
        sim = cosine_similarity_matrix(vectors[:, layer, :])
        matrices.append(sim)
    return torch.stack(matrices, dim=0)


def vector_norms_by_layer(vectors: Tensor) -> Tensor:
    """
    Compute the norm of raw (unnormalized) vectors per concept per layer.
    Useful for diagnosing which layers carry the most information.

    Args:
        vectors: (n_concepts, n_layers, d_model)

    Returns:
        (n_concepts, n_layers)
    """
    return vectors.norm(dim=-1)
