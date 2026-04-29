# Claude Code Prompt: Steering Vector Analysis Project

Copy everything below the line into Claude Code as your opening prompt.

---

I'm building a mechanistic interpretability project that computes and visualizes steering vectors from a small instruct-tuned LLM. I have 48 hours. I need you to scaffold the full project structure and write the core infrastructure code. Here's the spec:

## What this project does

1. Generate contrastive prompt pairs for ~25 concept pairs (formal/casual, certain/uncertain, polite/rude, English/French, happy/sad, refuse/comply, etc.)
2. Run forward passes through a small instruct model (start with Qwen 2.5 1.5B Instruct), extracting residual stream activations at every layer
3. Compute mean-difference steering vectors: for each concept pair at each layer, subtract mean activations of side B from side A
4. Produce four visualizations:
   - Cosine similarity heatmap across all concept directions at a chosen layer (with layer slider)
   - Per-layer animation of the similarity matrix evolving across layers
   - 2D UMAP of all concept directions across all layers, showing per-concept trajectories
   - Steering effectiveness heatmap: sweep (injection layer × coefficient strength), score outputs, plot where each concept is most steerable
5. Validate steering vectors actually work by injecting them during generation and confirming output shifts

## Project structure I want

```
steering_vectors/
├── config.py                 # All constants: model name, device, layer count, concept pairs list, paths
├── concepts.py               # Concept pair definitions and prompt generation (hand-crafted examples + GPT-4 expansion)
├── extraction.py             # Forward pass hooks, activation extraction, batched inference
├── compute_vectors.py        # Mean-difference computation, normalization, saving the (concepts × layers × d_model) tensor
├── validate.py               # Inject steering vectors during generation, print before/after examples
├── viz/
│   ├── cosine_heatmap.py     # Clustered heatmap with dendrogram, layer selection
│   ├── layer_animation.py    # Animate similarity matrix across layers, export GIF/MP4
│   ├── umap_trajectories.py  # UMAP reduction, per-concept paths colored and sized by layer
│   └── effectiveness_map.py  # Sweep layer × coefficient, score outputs, plot heatmaps
├── scoring.py                # Simple classifiers/keyword scorers for steering effectiveness evaluation
├── run_pipeline.py           # End-to-end: extract → compute → validate → visualize
└── requirements.txt
```

## Technical decisions (follow these, don't ask me to choose)

- Model: `Qwen/Qwen2.5-1.5B-Instruct` via HuggingFace transformers. Fall back to `meta-llama/Llama-3.2-1B-Instruct` if needed.
- Extract activations from the **residual stream after each transformer block** (the output of each decoder layer before it feeds into the next). Use PyTorch forward hooks.
- Take activations at the **last non-padding token position** for each prompt.
- Store raw activations as memory-mapped numpy arrays or torch tensors on disk to avoid recomputation.
- Steering vectors should be L2-normalized to unit vectors after mean-differencing.
- For UMAP use the `umap-learn` package. For heatmaps use seaborn with `clustermap`. For animation use `matplotlib.animation`.
- Steering injection: add `coeff * steering_vector` to the residual stream at the target layer during a forward hook.
- Scoring for effectiveness: use keyword/regex classifiers for surface concepts (language, tense), a small sentiment classifier for evaluative concepts, and refusal-phrase matching for stance concepts. Don't overthink this — fast and simple.

## What I need you to write now

1. **Full `config.py`** with all concept pairs defined, model config, device setup, all hyperparameters with sensible defaults.
2. **Full `extraction.py`** with the hook infrastructure, batched activation extraction, and disk saving. This is the most load-bearing code — make it clean and robust.
3. **Full `compute_vectors.py`** that reads saved activations and produces the normalized steering vector tensor.
4. **Full `validate.py`** that loads a steering vector, injects it during generation, and prints side-by-side comparisons.
5. **Skeleton versions** of the four viz modules and `scoring.py` — function signatures, docstrings, key logic pseudocoded, but not necessarily fully fleshed out.
6. **`concepts.py`** with 3-5 hand-crafted example pairs per concept and a function stub for GPT-4 expansion.
7. **`requirements.txt`** with pinned versions.
8. **`run_pipeline.py`** that chains everything together.

Write real, runnable code for items 1-4. The extraction and compute code must actually work — I'll be running it within hours. For the viz and scoring modules, give me enough that I can fill in the details quickly.

Don't explain the theory back to me. Just write the code. Use type hints. Keep functions small and composable. Add comments only where the "why" isn't obvious from the code.
