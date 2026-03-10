# Qwen3.5-4B Unsloth Autoresearch (RTX 3090)

A constrained local optimization loop for fine-tuning **Qwen3.5-4B** with **Unsloth bf16 LoRA** on a single RTX 3090.

## What this scaffold does

- Keeps the trainer and evaluator mostly static.
- Mutates only a constrained config/search space.
- Uses phased search so the loop does not thrash structural choices too early.
- Uses exact-row dataset mixing instead of proportional slicing by bucket size.
- Separates train and eval into subprocesses so VRAM is released between phases.
- Uses batched, left-padded eval generation.
- Fails fast on NaN / inf / OOM-ish crashes by writing a failure stub and skipping generation.

## Recommended environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install -r requirements.txt
```

## Important notes

- Qwen3.5 fine-tuning in Unsloth should use **bf16 LoRA**, not 4-bit QLoRA.
- Keep `use_gradient_checkpointing="unsloth"` enabled.
- For eval speed, `FastLanguageModel.for_inference(model)` is called before generation.
- This is a scaffold. You still need to replace the placeholder judge with a real judge for phase 2.

## Project layout

```text
qwen35-autoresearch/
├── README.md
├── requirements.txt
├── program.md
├── search_space.yaml
├── configs/
│   └── base.yaml
├── data/
│   ├── sft_general.jsonl
│   ├── reasoning.jsonl
│   ├── format.jsonl
│   ├── val.jsonl
│   └── judge_prompts.jsonl
├── adapters/
├── train_qwen35_unsloth.py
├── eval.py
├── score_run.py
├── validate_config.py
├── propose_next.py
├── run_loop.py
├── run_many.sh
└── results.jsonl
```

## First run

```bash
python run_loop.py
```

## Repeat loop

```bash
bash run_many.sh 20
```

## Data format

Training buckets (`sft_general.jsonl`, `reasoning.jsonl`, `format.jsonl`, `val.jsonl`) use:

```json
{"system":"optional system prompt","user":"task or question","assistant":"target answer"}
```

Eval prompts (`judge_prompts.jsonl`) use:

```json
{"id":"ex-1","prompt":"<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n","ideal":"reference answer"}
```

## Tuning guidance

Phase 1:
- Keep `target_modules_set=qkv_omlp`
- Keep `packing=true`
- Keep `max_seq_length=2048`
- Search only LR / LoRA / warmup / WD / data mix / reasoning ratio

Phase 2:
- Expose batch size / grad accumulation / max sequence length

Phase 3:
- Optionally explore structural changes like `attn_only`

## Production TODOs

- Replace `mock_judge_score()` with a fixed judge model and frozen rubric
- Add exact-task metrics where possible
- Add top-k survivor selection
- Add persistent hardware telemetry if you want to model watts or runtime drift
