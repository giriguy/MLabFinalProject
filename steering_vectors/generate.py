"""
Compare original vs. steered model outputs.

Steering methods:
  inject  — add steering vectors to the residual stream via forward hooks
  lora    — load a trained LoRA adapter (no runtime injection)
  both    — run injection and LoRA side-by-side for comparison

Examples:
  python generate.py --prompt "Tell me about climate change" --steer formal:3.0
  python generate.py --prompt "Tell me about climate change" --steer formal:3.0 --steer polite:2.0 --steer certain:4.0
  python generate.py --prompt "Write me a recipe" --steer technical --method lora
  python generate.py --prompt "Explain gravity" --steer formal:3.0 --method both --layer 14
  python generate.py --list
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config
from compute_vectors import compute_vector_for_concept, get_vector, load_vectors
from extraction import apply_chat_template, load_model_and_tokenizer
from validate import SteeringHook

LORA_ADAPTERS_DIR = config.DATA_DIR / "lora_adapters"
DEFAULT_COEFF = 3.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_steer(s: str) -> Tuple[str, float]:
    """Parse 'concept:coeff' or bare 'concept' into (concept, coeff)."""
    if ":" in s:
        concept, coeff = s.rsplit(":", 1)
        return concept.strip(), float(coeff)
    return s.strip(), DEFAULT_COEFF


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_available() -> None:
    _, concept_names = load_vectors()
    lora_ready = set()
    if LORA_ADAPTERS_DIR.exists():
        lora_ready = {
            d.name for d in LORA_ADAPTERS_DIR.iterdir()
            if (d / "final" / "adapter_model.safetensors").exists()
        }

    print(f"\n  {'Concept':<35}  Methods")
    print("  " + "-" * 52)
    for name in concept_names:
        methods = ["inject"]
        if name in lora_ready:
            methods.append("lora")
        print(f"  {name:<35}  {', '.join(methods)}")
    print()


# ---------------------------------------------------------------------------
# Best-layer auto-pick
# ---------------------------------------------------------------------------

def _best_layer(concept: str) -> int:
    """Return the layer index with the highest raw mean-difference norm."""
    raw = compute_vector_for_concept(concept, normalize=False)   # (n_layers, d_model)
    return int(np.argmax(np.linalg.norm(raw, axis=-1)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate(
    prompt: str,
    model,
    tokenizer,
    hooks: Optional[List[SteeringHook]] = None,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    formatted = apply_chat_template([prompt], tokenizer)[0]
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_INPUT_LENGTH,
    ).to(next(model.parameters()).device)

    do_sample = temperature > 0
    with ExitStack() as stack:
        for hook in (hooks or []):
            stack.enter_context(hook)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tok = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _build_injection_hooks(
    steer_specs: List[Tuple[str, float]],
    vectors,
    concept_names: List[str],
    layer_override: Optional[int],
    model,
) -> Tuple[List[SteeringHook], str]:
    """
    Build SteeringHooks from steer_specs.

    When multiple concepts share a layer their scaled vectors are summed into
    one hook. When they land on different layers, one hook is registered per
    unique layer.

    Returns (hooks, human-readable description).
    """
    # Assign a layer to each concept
    layer_for: Dict[str, int] = {
        concept: (layer_override if layer_override is not None else _best_layer(concept))
        for concept, _ in steer_specs
    }

    # Sum scaled vectors at each unique layer
    combined: Dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(config.D_MODEL))
    for concept, coeff in steer_specs:
        l = layer_for[concept]
        v = get_vector(concept, l, vectors, concept_names).float()
        combined[l] = combined[l] + coeff * v

    # SteeringHook._get_layer checks self._model.model.layers.
    # For the base Qwen2ForCausalLM that is satisfied directly (.model = Qwen2Model).
    # For a PeftModel, .model is Qwen2ForCausalLM (not Qwen2Model), so we unwrap
    # one level — but only when the immediate .model child lacks .layers.
    m = model
    if hasattr(m, "model") and not hasattr(m.model, "layers"):
        m = m.model   # PeftModel → Qwen2ForCausalLM
    model_for_hook = m

    hooks = [
        SteeringHook(model_for_hook, vec, layer_idx=layer, coeff=1.0)
        for layer, vec in sorted(combined.items())
    ]

    desc_parts = [
        f"{c} (coeff={coeff:.1f}, layer={layer_for[c]})"
        for c, coeff in steer_specs
    ]
    return hooks, " + ".join(desc_parts)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def _load_lora(base_model, concepts: List[str]):
    """Load and activate LoRA adapter(s) on top of base_model. Returns PeftModel."""
    from peft import PeftModel

    valid: List[Tuple[str, Path]] = []
    for concept in concepts:
        path = LORA_ADAPTERS_DIR / concept / "final"
        if not path.exists():
            print(f"  [warn] No LoRA adapter found for '{concept}' — skipping.", file=sys.stderr)
        else:
            valid.append((concept, path))

    if not valid:
        raise RuntimeError(
            "No trained LoRA adapters found for the requested concepts. "
            "Run train_lora.py first."
        )

    first_name, first_path = valid[0]
    peft_model = PeftModel.from_pretrained(base_model, str(first_path), adapter_name=first_name)

    for name, path in valid[1:]:
        peft_model.load_adapter(str(path), adapter_name=name)

    if len(valid) == 1:
        peft_model.set_adapter(first_name)
    else:
        # Combine by concatenating ΔW matrices — equivalent to summing the deltas.
        names = [n for n, _ in valid]
        peft_model.add_weighted_adapter(
            adapters=names,
            weights=[1.0] * len(names),
            adapter_name="combined",
            combination_type="cat",
        )
        peft_model.set_adapter("combined")

    return peft_model


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_result(
    prompt: str,
    steering_desc: str,
    method: str,
    original: str,
    steered: Dict[str, str],
    width: int = 88,
    output_path: Optional[Path] = None,
) -> None:
    sep = "─" * width
    lines: List[str] = []

    lines += [f"\n{sep}", f"PROMPT  : {prompt}", f"STEERING: {steering_desc}", f"METHOD  : {method}", sep]

    def block(label: str, text: str) -> None:
        lines.append(f"\n--- {label} ---")
        lines.extend(textwrap.wrap(text, width=width) or ["(empty response)"])

    block("ORIGINAL", original)
    for label, text in steered.items():
        block(label, text)
    lines.append(f"\n{sep}\n")

    output = "\n".join(lines)
    print(output)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode) as f:
            f.write(output)
        print(f"Saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original vs. steered model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", help="Input prompt (required unless --list).")
    parser.add_argument(
        "--steer",
        action="append",
        default=[],
        metavar="concept[:coeff]",
        help=(
            "Steering direction. Repeatable for stacking. "
            "Format: concept:coeff (coeff optional, default 3.0)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["inject", "lora", "both"],
        default="inject",
        help="Steering method (default: inject).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Injection layer index. Default: auto-pick the layer with highest raw-vector norm.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max new tokens to generate (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}). Use 0 for greedy.",
    )
    parser.add_argument("--list", action="store_true", help="List available concepts and exit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Append results to this text file (created if absent).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name or path override (default: {config.MODEL_NAME}).",
    )
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if not args.prompt:
        parser.error("--prompt is required (or use --list)")
    if not args.steer:
        parser.error("at least one --steer is required")

    steer_specs = [_parse_steer(s) for s in args.steer]
    steer_concepts = [c for c, _ in steer_specs]

    model_name = args.model or config.MODEL_NAME
    model, tokenizer = load_model_and_tokenizer(model_name=model_name)
    model.eval()

    vectors, concept_names = load_vectors()

    for concept, _ in steer_specs:
        if concept not in concept_names:
            print(
                f"Error: unknown concept '{concept}'. Run --list to see available concepts.",
                file=sys.stderr,
            )
            sys.exit(1)

    gen_kw = dict(max_new_tokens=args.max_tokens, temperature=args.temperature)

    # Baseline
    print("Generating original response…", file=sys.stderr)
    original = _generate(args.prompt, model, tokenizer, **gen_kw)

    steered: Dict[str, str] = {}
    inject_desc: Optional[str] = None

    neg_steer_specs = [(c, -coeff) for c, coeff in steer_specs]

    # Activation injection
    if args.method in ("inject", "both"):
        pos_label = "STEERED+ (injection)" if args.method == "both" else "STEERED+"
        neg_label = "STEERED- (injection)" if args.method == "both" else "STEERED-"

        print("Generating injection-steered response (+)…", file=sys.stderr)
        hooks, inject_desc = _build_injection_hooks(
            steer_specs, vectors, concept_names, args.layer, model
        )
        steered[pos_label] = _generate(args.prompt, model, tokenizer, hooks=hooks, **gen_kw)

        print("Generating injection-steered response (-)…", file=sys.stderr)
        neg_hooks, _ = _build_injection_hooks(
            neg_steer_specs, vectors, concept_names, args.layer, model
        )
        steered[neg_label] = _generate(args.prompt, model, tokenizer, hooks=neg_hooks, **gen_kw)

    # LoRA
    if args.method in ("lora", "both"):
        print("Loading LoRA adapter(s)…", file=sys.stderr)
        peft_model = _load_lora(model, steer_concepts)
        peft_model.eval()
        label = "STEERED (LoRA)" if args.method == "both" else "STEERED"
        print("Generating LoRA-steered response…", file=sys.stderr)
        steered[label] = _generate(args.prompt, peft_model, tokenizer, **gen_kw)

    # Build display description
    if inject_desc and args.method == "both":
        full_desc = f"{inject_desc}  +  LoRA: {', '.join(steer_concepts)}"
        method_str = "activation injection + LoRA (side-by-side)"
    elif inject_desc:
        full_desc = inject_desc
        method_str = "activation injection"
    else:
        full_desc = ", ".join(steer_concepts)
        method_str = "LoRA"

    _print_result(args.prompt, full_desc, method_str, original, steered, output_path=args.output)


if __name__ == "__main__":
    main()
