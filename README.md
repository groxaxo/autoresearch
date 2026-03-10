# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Two modes

This repository supports two autonomous-research workflows in a single project:

| Mode | GPU | What happens | Key metric |
|------|-----|-------------|------------|
| **From-scratch pretraining** | H100 (80 GB) | Agent edits `train.py`, trains a GPT from scratch for 5 min | `val_bpb` (lower is better) |
| **Qwen 3.5 LoRA fine-tuning** | RTX 3090 / ≤ 24 GB | Automated loop fine-tunes Qwen3.5-4B with Unsloth bf16 LoRA | `composite_score` (higher is better) |

The **Qwen fine-tuning mode** is designed for users with RTX 3090s or other GPUs with ≤ 24 GB VRAM. It runs Qwen3.5-4B with bf16 LoRA (not 4-bit QLoRA) via Unsloth, using a constrained search loop that automatically proposes, trains, evaluates, and scores configurations.

---

## Quick start — Qwen 3.5 fine-tuning (RTX 3090 / low-VRAM)

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install Unsloth first (pins its own torch/transformers versions)
pip install --upgrade pip setuptools wheel
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run a single optimisation iteration (~15 min)
python run_loop.py

# 5. Run many iterations unattended
bash run_many.sh 20   # 20 iterations
```

Each iteration of the loop:

1. **`propose_next.py`** — mutates the current best config within the search space.
2. **`validate_config.py`** — checks constraints (VRAM safety, runtime budget).
3. **`train_qwen35_unsloth.py`** — fine-tunes Qwen3.5-4B with Unsloth bf16 LoRA.
4. **`eval.py`** — generates predictions and scores them with a token-overlap judge (optionally an LLM-as-judge via `--use-llm-judge`).
5. **`score_run.py`** — computes a weighted composite score.

Results are appended to `results.jsonl`.

### RTX 3090 VRAM budget

| Component | Approx VRAM |
|-----------|------------|
| Qwen3.5-4B bf16 weights | ~8 GB |
| LoRA adapters + optimizer | ~1–2 GB |
| Activations (gradient checkpointing) | ~4–8 GB |
| **Headroom** | **~6–11 GB** |

The constraint `max_micro_tokens ≤ 8192` (seq_len × batch_size) keeps peak memory well within 24 GB.

---

## Quick start — from-scratch pretraining (H100)

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (includes kernels for Flash Attention 3)
uv sync --extra pretrain

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better.

---

## Running the agent

Spin up your Claude/Codex or any agent in this repo (disable all permissions), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is a lightweight "skill" that tells the agent what to do.

## Project structure

```
# Qwen 3.5 fine-tuning (RTX 3090 / low-VRAM)
train_qwen35_unsloth.py — Qwen3.5-4B SFT with Unsloth bf16 LoRA
eval.py                 — generate predictions + judge scoring
propose_next.py         — propose next config by mutating current best
run_loop.py             — orchestrate one full iteration
score_run.py            — compute weighted composite score
validate_config.py      — validate config against search space
run_many.sh             — run N iterations unattended
search_space.yaml       — hyperparameter search space and constraints
configs/base.yaml       — base configuration for Phase 1
data/                   — training and evaluation JSONL datasets
adapters/               — saved fine-tuned LoRA adapters
results.jsonl           — experiment log
requirements.txt        — pip dependencies for Qwen fine-tuning

# From-scratch pretraining (H100)
prepare.py              — constants, data prep + runtime utilities (do not modify)
train.py                — model, optimizer, training loop (agent modifies this)

# Shared
program.md              — agent instructions
pyproject.toml          — project metadata and dependency groups
```

## Design choices

- **Single file to modify** (pretraining mode). The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **Constrained search space** (fine-tuning mode). The agent proposes configs within `search_space.yaml` rather than editing code directly. Phased search avoids early structural changes.
- **Fixed time budget.** Pretraining mode: 5 minutes. Fine-tuning mode: ~15 minutes per iteration. Makes experiments directly comparable.
- **Self-contained.** One GPU, one metric, no distributed training.
- **Real judge scoring.** `eval.py` uses token-overlap F1, keyword recall, and length-ratio metrics to score predictions against reference answers. An optional `--use-llm-judge` flag adds model-based rating (blended 60 % token / 40 % LLM).

## Platform support

| Platform | Mode | Status |
|----------|------|--------|
| NVIDIA H100 (80 GB) | From-scratch pretraining | ✅ Fully supported |
| NVIDIA RTX 3090 (24 GB) | Qwen 3.5 fine-tuning | ✅ Fully supported |
| NVIDIA RTX 4090 (24 GB) | Qwen 3.5 fine-tuning | ✅ Should work |
| Other NVIDIA ≤ 24 GB | Qwen 3.5 fine-tuning | ⚠️ May need smaller batch/seq |

For even smaller GPUs (< 16 GB), reduce `max_micro_tokens` in `search_space.yaml` and start with `per_device_train_batch_size: 1`, `max_seq_length: 1024`.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
