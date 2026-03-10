# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Two modes

This repository supports two autonomous-research workflows in a single project:

| Mode | GPU | What happens | Key metric |
|------|-----|-------------|------------|
| **Qwen 3.5 LoRA fine-tuning** | **Ampere RTX 3090 (24 GB) / RTX 3060 (12 GB)** | Automated loop fine-tunes Qwen3.5-4B with Unsloth bf16 LoRA | `composite_score` (higher is better) |
| **From-scratch pretraining** | H100 (80 GB) | Agent edits `train.py`, trains a GPT from scratch for 5 min | `val_bpb` (lower is better) |

The **default focus of this repo is Ampere fine-tuning**, especially **RTX 3090** and **RTX 3060** class GPUs. It runs Qwen3.5-4B with bf16 LoRA (not 4-bit QLoRA) via Unsloth, using a constrained self-improving loop that automatically proposes, trains, evaluates, and scores configurations. The H100 pretraining path is still available, but it is the advanced / high-budget branch of the project rather than the default starting point.

---

## Quick start — Qwen 3.5 fine-tuning on Ampere (RTX 3090 / RTX 3060)

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
# RTX 3060 users: GPU_VRAM_GB=12 python run_loop.py

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

### Ampere VRAM guidance

| GPU | Good starting point | Notes |
|-----|---------------------|-------|
| **RTX 3090 (24 GB)** | `max_seq_length: 2048`, `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 8` | Current default baseline. Enough room to explore larger micro-batches in Phase 2. |
| **RTX 3060 (12 GB)** | `max_seq_length: 1024`, `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8-16` | Start conservative and only scale up after observing stable memory use. |

For the 3090, the constraint `max_micro_tokens ≤ 8192` (seq_len × batch_size) keeps peak memory well within 24 GB. For a 3060, set `GPU_VRAM_GB=12` before running `python run_loop.py` to apply a tighter validator limit **and** a more conservative Phase 1 seed (`max_seq_length: 1024`, `per_device_train_batch_size: 1`).

Ampere-specific rule of thumb:

- **3090**: maximize experiment throughput first; it has enough headroom to explore more aggressive Phase 2 configs.
- **3060**: maximize iteration count and stability first; more small, clean runs usually beat one memory-edge run.

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
# Qwen 3.5 fine-tuning on Ampere (RTX 3090 / RTX 3060)
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

## Taking the self-improving loop to the next level

The current loop already proposes, runs, and scores experiments autonomously. To push it further on Ampere GPUs, the next gains likely come from improving the **research policy**, not just adding more raw search:

1. **GPU-aware search profiles.** Treat RTX 3060 and RTX 3090 as separate operating regimes and compare ideas within each budget instead of mixing all results together.
2. **Learning from trajectories, not single runs.** Have the agent track which mutations repeatedly help or hurt composite score, then bias future proposals toward successful patterns.
3. **Promote stable configs into templates.** When the loop finds a strong region, fork it into a reusable seed config for that GPU tier instead of always mutating from one global baseline.
4. **Add lightweight memory telemetry.** Persist peak VRAM, tokens/sec, and failure causes so the agent can trade off quality against throughput on 3060 vs 3090.
5. **Introduce periodic reflection.** Every N runs, ask the agent to summarize what changed, what failed, and what hypothesis should be tested next. That is the simplest path from "search loop" to "self-improving research system."

## Platform support

| Platform | Mode | Status |
|----------|------|--------|
| NVIDIA RTX 3090 (24 GB) | Qwen 3.5 fine-tuning | ✅ Fully supported |
| NVIDIA RTX 3060 (12 GB) | Qwen 3.5 fine-tuning | ✅ Supported with conservative batch/sequence settings |
| NVIDIA RTX 4090 (24 GB) | Qwen 3.5 fine-tuning | ✅ Should work |
| NVIDIA H100 (80 GB) | From-scratch pretraining | ✅ Fully supported |
| Other NVIDIA ≤ 24 GB | Qwen 3.5 fine-tuning | ⚠️ Start from the RTX 3060 settings |

For even smaller GPUs (< 16 GB), start with the RTX 3060 settings, reduce `max_micro_tokens`, and prefer more gradient accumulation over larger micro-batches.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
