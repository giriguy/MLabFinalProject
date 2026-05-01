"""
Visualisations for trained LoRA adapters — three figures:

  1. lora_training_loss.png  — training loss curves from trainer_state.json
  2. lora_weight_analysis.png — per-layer Frobenius norm and steering-vector
                                 alignment for each adapter (no inference needed)
  3. lora_score_comparison.png — concept scores: baseline vs. LoRA vs. steered
                                  model on held-out prompts (requires inference)

Usage:
    cd steering_vectors
    python viz/lora_analysis.py                   # all three figures
    python viz/lora_analysis.py --skip-inference  # only figs 1 & 2 (fast)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
from safetensors.torch import load_file

# Allow running from the steering_vectors/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from compute_vectors import load_vectors
from scoring import get_scorer

logger = logging.getLogger(__name__)

LORA_ADAPTERS_DIR = config.DATA_DIR / "lora_adapters"
OUT_DIR = config.VIZ_DIR

# Held-out eval prompts — distinct from the DIVERSE_PROMPTS used during training.
EVAL_PROMPTS: List[str] = [
    "What do you think about the future of space exploration?",
    "Describe what makes a perfect day.",
    "What is your opinion on remote work?",
    "How do you feel about the current state of technology?",
    "What would make the world a better place?",
    "What are your thoughts on artificial intelligence?",
    "How do people find meaning in their work?",
    "What makes a city a great place to live?",
    "How has the internet changed society?",
    "What is the most interesting scientific discovery of recent decades?",
    "Describe a memorable journey you have taken.",
    "What lessons can we learn from history?",
    "How do you approach learning a new skill?",
    "What role does art play in society?",
    "Describe an ideal education system.",
    "What does success mean to you?",
    "How can communities support mental health?",
    "What are the trade-offs of social media?",
    "Describe a meaningful conversation.",
    "What habits make a person happy?",
    "How will transportation change in 50 years?",
    "Describe what good leadership looks like.",
    "What are the benefits of reading regularly?",
    "Explain why people should travel more.",
    "Discuss the role of curiosity in life.",
    "What changes would you make to your city?",
    "Describe how to cook a simple meal.",
    "What advice would you give a teenager?",
    "Explain the value of friendship.",
    "Describe a sustainable lifestyle.",
    "What makes a story memorable?",
    "Discuss the future of healthcare.",
]

# Colour palette consistent with the rest of the viz outputs
PALETTE = sns.color_palette("Set2", 8)
COL_BASELINE = PALETTE[0]
COL_LORA     = PALETTE[1]
COL_STEERED  = PALETTE[2]
COL_TS       = PALETTE[3]   # technical_simple accent
COL_OP       = PALETTE[4]   # optimistic_pessimistic accent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trained_adapters() -> Dict[str, Path]:
    """Return {concept_name: adapter_path} for every trained final adapter."""
    adapters = {}
    for d in sorted(LORA_ADAPTERS_DIR.iterdir()):
        final = d / "final"
        if final.is_dir() and (final / "adapter_model.safetensors").exists():
            adapters[d.name] = final
    return adapters


def _load_trainer_state(concept: str) -> List[dict]:
    pattern = str(LORA_ADAPTERS_DIR / concept / "checkpoint-*" / "trainer_state.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return []
    with open(paths[-1]) as f:
        return json.load(f).get("log_history", [])


def _load_lora_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    return load_file(str(adapter_path / "adapter_model.safetensors"))


def _compute_delta_w(weights: Dict[str, torch.Tensor], layer: int, module: str) -> torch.Tensor:
    """Compute ΔW = B @ A for a given layer and module (q_proj or v_proj)."""
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
    A = weights[f"{prefix}.lora_A.weight"].float()   # (r, d_in)
    B = weights[f"{prefix}.lora_B.weight"].float()   # (d_out, r)
    return B @ A   # (d_out, d_in)


# ---------------------------------------------------------------------------
# Figure 1: Training loss curves
# ---------------------------------------------------------------------------

def plot_training_loss(adapters: Dict[str, Path], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(adapters), figsize=(5 * len(adapters), 4), sharey=False)
    if len(adapters) == 1:
        axes = [axes]

    accent_colours = [COL_TS, COL_OP, PALETTE[5], PALETTE[6]]
    fig.suptitle("LoRA Training Loss", fontsize=14, fontweight="bold", y=1.02)

    for ax, (concept, path), colour in zip(axes, adapters.items(), accent_colours):
        logs = _load_trainer_state(concept)
        steps = [e["step"] for e in logs if "loss" in e]
        losses = [e["loss"] for e in logs if "loss" in e]

        if not steps:
            ax.text(0.5, 0.5, "No log data", transform=ax.transAxes, ha="center")
        else:
            ax.plot(steps, losses, color=colour, linewidth=2, marker="o", markersize=4)
            ax.fill_between(steps, losses, alpha=0.12, color=colour)

        ax.set_title(concept.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Figure 2: LoRA weight analysis — no inference needed
# ---------------------------------------------------------------------------

def _weight_stats_per_layer(
    weights: Dict[str, torch.Tensor],
    steering_vectors: torch.Tensor,
    concept_idx: int,
    n_layers: int = 28,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each layer compute:
      fnorm_q  — Frobenius norm of ΔW_q (q_proj)
      fnorm_v  — Frobenius norm of ΔW_v (v_proj)
      align_q  — cosine similarity between ΔW_q @ steering_v and steering_v
                 (measures how well LoRA maps the steering direction to itself)
    """
    fnorm_q = np.zeros(n_layers)
    fnorm_v = np.zeros(n_layers)
    align_q = np.zeros(n_layers)

    for l in range(n_layers):
        sv = steering_vectors[concept_idx, l].float()   # (d_model,)

        dw_q = _compute_delta_w(weights, l, "q_proj")   # (1536, 1536)
        dw_v = _compute_delta_w(weights, l, "v_proj")   # (256, 1536)

        fnorm_q[l] = dw_q.norm(p="fro").item()
        fnorm_v[l] = dw_v.norm(p="fro").item()

        # Projection of steering vector through LoRA
        response = dw_q @ sv                           # (1536,)
        eps = 1e-8
        cos = (response @ sv) / (response.norm() + eps) / (sv.norm() + eps)
        align_q[l] = cos.item()

    return fnorm_q, fnorm_v, align_q


def plot_weight_analysis(adapters: Dict[str, Path], save_path: Path) -> None:
    vectors, concept_names = load_vectors()
    n = len(adapters)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[None, :]   # ensure 2-D indexing

    fig.suptitle("LoRA Weight Analysis per Layer", fontsize=14, fontweight="bold")
    layers = np.arange(config.N_LAYERS)
    accent_colours = [COL_TS, COL_OP, PALETTE[5], PALETTE[6]]

    for row, (concept, adapter_path), colour in zip(range(n), adapters.items(), accent_colours):
        if concept not in concept_names:
            continue
        concept_idx = concept_names.index(concept)
        weights = _load_lora_weights(adapter_path)

        fnorm_q, fnorm_v, align_q = _weight_stats_per_layer(
            weights, vectors, concept_idx, config.N_LAYERS
        )

        # Left: Frobenius norms
        ax_l = axes[row, 0]
        ax_l.plot(layers, fnorm_q, color=colour, linewidth=1.8, label="q_proj")
        ax_l.plot(layers, fnorm_v, color=colour, linewidth=1.8, linestyle="--",
                  alpha=0.6, label="v_proj")
        ax_l.set_title(f"{concept.replace('_', ' ').title()} — ΔW Frobenius Norm", fontsize=10)
        ax_l.set_xlabel("Layer")
        ax_l.set_ylabel("‖ΔW‖_F")
        ax_l.legend(fontsize=8)
        ax_l.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax_l)

        # Right: alignment
        ax_r = axes[row, 1]
        ax_r.plot(layers, align_q, color=colour, linewidth=1.8)
        ax_r.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax_r.set_title(
            f"{concept.replace('_', ' ').title()} — Steering-Vector Alignment (q_proj)", fontsize=10
        )
        ax_r.set_xlabel("Layer")
        ax_r.set_ylabel("cosine sim(ΔW·v, v)")
        ax_r.set_ylim(-1.1, 1.1)
        ax_r.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax_r)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Figure 3: Score comparison — requires model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate(
    prompt: str,
    model,
    tokenizer,
    steering_vec: Optional[torch.Tensor] = None,
    coeff: float = 0.0,
    layer_idx: Optional[int] = None,
) -> str:
    """Generate from any model (base or PeftModel), with optional steering."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=config.MAX_INPUT_LENGTH
    ).to(next(model.parameters()).device)

    handle = None
    if steering_vec is not None and coeff != 0.0 and layer_idx is not None:
        # Find transformer layers regardless of PeftModel wrapping depth
        m = model
        while not hasattr(m, "layers"):
            m = m.model
        vec = steering_vec.to(next(model.parameters()).device,
                               dtype=next(model.parameters()).dtype)

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + coeff * vec.unsqueeze(0).unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = m.layers[layer_idx].register_forward_hook(_hook)

    try:
        out_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    new_tok = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()


def _score_prompts(
    prompts: List[str],
    model,
    tokenizer,
    concept: str,
    vectors: torch.Tensor,
    concept_names: List[str],
    layer_idx: int,
    coeff: float,
    peft_model=None,
) -> Dict[str, List[float]]:
    """
    Generate and score outputs for one concept under three conditions:
    baseline, LoRA, and steered (no LoRA).
    """
    from peft import PeftModel

    scorer = get_scorer(concept)
    concept_idx = concept_names.index(concept)
    sv = vectors[concept_idx, layer_idx]

    scores: Dict[str, List[float]] = {"baseline": [], "lora": [], "steered": []}

    for prompt in prompts:
        # --- baseline (no LoRA, no steering) ---
        if peft_model is not None:
            peft_model.disable_adapter_layers()
            text = _generate(prompt, peft_model, tokenizer)
        else:
            text = _generate(prompt, model, tokenizer)
        scores["baseline"].append(scorer(text))

        # --- LoRA (LoRA active, no steering) ---
        if peft_model is not None:
            peft_model.enable_adapter_layers()
            peft_model.set_adapter(concept)
            text = _generate(prompt, peft_model, tokenizer)
            scores["lora"].append(scorer(text))
        else:
            scores["lora"].append(np.nan)

        # --- steered (no LoRA, steering vector injected) ---
        if peft_model is not None:
            peft_model.disable_adapter_layers()
        text = _generate(prompt, model, tokenizer, sv, coeff, layer_idx)
        scores["steered"].append(scorer(text))

    return scores


def plot_score_comparison(
    adapters: Dict[str, Path],
    save_path: Path,
    n_eval: int = 10,
    coeff: float = config.DEFAULT_COEFF,
) -> None:
    from extraction import load_model_and_tokenizer
    from peft import PeftModel

    model, tokenizer = load_model_and_tokenizer()
    vectors, concept_names = load_vectors()
    layer_idx = config.N_LAYERS // 2
    prompts = EVAL_PROMPTS[:n_eval]

    # Load all trained adapters under different names so we can switch cheaply
    peft_model = None
    for concept, adapter_path in adapters.items():
        if concept not in concept_names:
            continue
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=concept)
        else:
            peft_model.load_adapter(str(adapter_path), adapter_name=concept)

    n = len(adapters)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Concept Score: Baseline vs. LoRA vs. Steered Model", fontsize=13, fontweight="bold")
    accent_colours = [COL_TS, COL_OP, PALETTE[5], PALETTE[6]]

    from scipy import stats as scistats

    for ax, (concept, _), colour in zip(axes, adapters.items(), accent_colours):
        if concept not in concept_names:
            continue

        logger.info("Scoring concept: %s (%d prompts)", concept, len(prompts))
        scores = _score_prompts(
            prompts, model, tokenizer, concept,
            vectors, concept_names, layer_idx, coeff, peft_model
        )

        conditions = ["baseline", "lora", "steered"]
        colours_bar = [COL_BASELINE, colour, COL_STEERED]
        labels = ["Baseline", "LoRA", "Steered"]
        means = [np.nanmean(scores[c]) for c in conditions]
        # 95% CI using t-distribution
        cis = []
        for c in conditions:
            vals = np.array([v for v in scores[c] if not np.isnan(v)])
            if len(vals) < 2:
                cis.append(0.0)
                continue
            sem = scistats.sem(vals)
            h = sem * scistats.t.ppf(0.975, len(vals) - 1)
            cis.append(h)

        ax.bar(labels, means, yerr=cis, color=colours_bar,
               capsize=8, edgecolor="white", linewidth=1.0, alpha=0.92)

        # Scatter individual points with jitter
        for i, c in enumerate(conditions):
            vals = [v for v in scores[c] if not np.isnan(v)]
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="black", s=22, alpha=0.45, zorder=5,
                       edgecolors="white", linewidths=0.6)

        # Paired t-tests: baseline vs LoRA, baseline vs Steered
        def _annotate_sig(i_lo, i_hi, base_vals, comp_vals, height):
            base_arr = np.array([v for v in base_vals if not np.isnan(v)])
            comp_arr = np.array([v for v in comp_vals if not np.isnan(v)])
            if len(base_arr) < 2 or len(comp_arr) < 2:
                return
            n = min(len(base_arr), len(comp_arr))
            try:
                _, p = scistats.ttest_rel(base_arr[:n], comp_arr[:n])
            except Exception:
                return
            if np.isnan(p):
                return
            mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.plot([i_lo, i_hi], [height, height], color="black", linewidth=1)
            ax.plot([i_lo, i_lo], [height - 0.015, height], color="black", linewidth=1)
            ax.plot([i_hi, i_hi], [height - 0.015, height], color="black", linewidth=1)
            ax.text((i_lo + i_hi) / 2, height + 0.01, mark,
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        _annotate_sig(0, 1, scores["baseline"], scores["lora"], 0.93)
        _annotate_sig(0, 2, scores["baseline"], scores["steered"], 1.00)

        n_eval_actual = sum(1 for v in scores["baseline"] if not np.isnan(v))
        ax.set_title(
            f"{concept.replace('_', ' ').title()}  (n={n_eval_actual})",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Concept score  (0 = side-B, 1 = side-A)", fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate LoRA visualisations")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the score comparison figure (no model loading needed).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=10,
        help="Number of eval prompts for score comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--coeff", type=float, default=config.DEFAULT_COEFF,
        help="Steering coefficient used for the steered condition (default: %(default)s).",
    )
    args = parser.parse_args()

    adapters = _find_trained_adapters()
    if not adapters:
        logger.error(
            "No trained adapters found in %s. "
            "Run train_lora.py first.",
            LORA_ADAPTERS_DIR,
        )
        return

    logger.info("Found adapters: %s", list(adapters.keys()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_loss(adapters, OUT_DIR / "lora_training_loss.png")
    plot_weight_analysis(adapters, OUT_DIR / "lora_weight_analysis.png")

    if not args.skip_inference:
        plot_score_comparison(
            adapters, OUT_DIR / "lora_score_comparison.png",
            n_eval=args.n_eval,
            coeff=args.coeff,
        )
    else:
        logger.info("Skipping score comparison (--skip-inference).")

    logger.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
