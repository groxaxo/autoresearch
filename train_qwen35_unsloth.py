import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


TARGET_MODULE_MAP = {
    "qkv_omlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_only": ["gate_proj", "up_proj", "down_proj"],
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(example):
    system = example.get("system", "")
    user = example["user"]
    assistant = example["assistant"]

    text = ""
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{user}<|im_end|>\n"
    text += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    return {"text": text}


def sample_dataset(path, n_rows, seed):
    ds = load_dataset("json", data_files=str(path), split="train")
    if len(ds) == 0:
        raise ValueError(f"Empty dataset: {path}")

    if n_rows <= len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_rows))
    else:
        parts = []
        remaining = n_rows
        local_seed = seed
        while remaining > 0:
            chunk = ds.shuffle(seed=local_seed)
            take = min(len(chunk), remaining)
            parts.append(chunk.select(range(take)))
            remaining -= take
            local_seed += 1
        ds = concatenate_datasets(parts)
    return ds


def allocate_counts(weights, total_rows):
    raw = {k: total_rows * v for k, v in weights.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total_rows - sum(counts.values())
    if remainder > 0:
        ranked = sorted(weights.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
        for i in range(remainder):
            counts[ranked[i % len(ranked)]] += 1
    return counts


def load_mixed_dataset(config):
    space = load_yaml("search_space.yaml")
    mix = space["dataset_mix_options"][config["dataset_mix_name"]]
    total_rows = int(config.get("target_train_rows", 12000))
    seed = int(config["seed"])

    counts = allocate_counts(mix, total_rows)
    datasets = []
    for name in mix.keys():
        path = Path("data") / f"{name}.jsonl"
        ds = sample_dataset(path, counts[name], seed)
        datasets.append(ds)

    ds = concatenate_datasets(datasets).shuffle(seed=seed)
    ds = ds.map(build_prompt)
    return ds


def is_finite_number(x):
    if x is None:
        return True
    try:
        x = float(x)
    except Exception:
        return True
    return math.isfinite(x)


def collect_nonfinite_flags(train_result, eval_metrics):
    flags = {
        "training_loss_nonfinite": False,
        "eval_loss_nonfinite": False,
        "metrics_nonfinite": False,
    }

    training_loss = getattr(train_result, "training_loss", None)
    if not is_finite_number(training_loss):
        flags["training_loss_nonfinite"] = True

    eval_loss = eval_metrics.get("eval_loss", None)
    if not is_finite_number(eval_loss):
        flags["eval_loss_nonfinite"] = True

    metrics = getattr(train_result, "metrics", {}) or {}
    for _, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            flags["metrics_nonfinite"] = True
            break

    return flags


def write_failure_eval_stub(output_dir: Path, runtime_sec: float, reason: str):
    stub = {
        "val_loss": 999.0,
        "format_score": 0.0,
        "judge_score": 0.0,
        "runtime_sec": runtime_sec,
        "stability_score": 0.0,
        "failure_penalty": 10.0,
        "nan_detected": True,
        "failure_reason": reason,
    }
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2)
    (output_dir / "SKIP_EVAL_NONFINITE").write_text(reason + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=TARGET_MODULE_MAP[config["lora"]["target_modules_set"]],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    train_ds = load_mixed_dataset(config)
    val_ds = load_dataset("json", data_files="data/val.jsonl", split="train").map(build_prompt)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=max(1, config["per_device_train_batch_size"]),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=-1 if config.get("max_steps") is None else int(config["max_steps"]),
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to=[],
        seed=config["seed"],
        dataset_text_field="text",
        max_length=config["max_seq_length"],
        packing=config["packing"],
        eos_token="<|im_end|>",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    try:
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        runtime_sec = time.time() - t0
    except RuntimeError as exc:
        runtime_sec = time.time() - t0
        reason = f"runtime_error: {exc}"
        with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "runtime_sec": runtime_sec,
                    "train_metrics": {},
                    "eval_metrics": {},
                    "nonfinite_flags": {},
                    "exception": reason,
                    "traceback": traceback.format_exc(),
                },
                f,
                indent=2,
            )
        write_failure_eval_stub(output_dir, runtime_sec, reason)
        print(f"[FAIL-FAST] RuntimeError for {output_dir.name}: {exc}")
        return
    except Exception as exc:
        runtime_sec = time.time() - t0
        reason = f"exception: {exc}"
        with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "runtime_sec": runtime_sec,
                    "train_metrics": {},
                    "eval_metrics": {},
                    "nonfinite_flags": {},
                    "exception": reason,
                    "traceback": traceback.format_exc(),
                },
                f,
                indent=2,
            )
        write_failure_eval_stub(output_dir, runtime_sec, reason)
        print(f"[FAIL-FAST] Exception for {output_dir.name}: {exc}")
        return

    nonfinite = collect_nonfinite_flags(train_result, eval_metrics)
    any_nonfinite = any(nonfinite.values())

    metrics_blob = {
        "runtime_sec": runtime_sec,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "nonfinite_flags": nonfinite,
    }

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_blob, f, indent=2)

    if any_nonfinite:
        reason = json.dumps(nonfinite)
        write_failure_eval_stub(output_dir=output_dir, runtime_sec=runtime_sec, reason=reason)
        print(f"[FAIL-FAST] Non-finite metrics detected for {output_dir.name}: {nonfinite}")
        return

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
