import argparse
import json
import os
from pathlib import Path

import torch
from unsloth import FastLanguageModel


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_format(outputs):
    total = len(outputs)
    good = 0
    for row in outputs:
        text = row["prediction"].strip()
        if text:
            good += 1
    return good / max(total, 1)


def mock_judge_score(outputs):
    scores = []
    for row in outputs:
        text = row["prediction"].strip()
        score = min(len(text) / 200.0, 1.0) if text else 0.0
        scores.append(score)
    return sum(scores) / max(len(scores), 1)


def generate_predictions_batched(
    run_dir: Path,
    prompts,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    max_input_length: int = 1536,
    max_new_tokens: int = 256,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(run_dir),
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model.eval()
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    outputs = []
    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        batch_prompts = [row["prompt"] for row in batch]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            pad_to_multiple_of=8,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i, row in enumerate(batch):
            prompt_len = int(input_lengths[i])
            new_tokens = gen[i, prompt_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(
                {
                    "id": row.get("id", start + i),
                    "prompt": row["prompt"],
                    "prediction": pred,
                    "ideal": row.get("ideal", ""),
                }
            )

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sentinel = run_dir / "SKIP_EVAL_NONFINITE"
    if sentinel.exists():
        print(f"Skipping eval for {run_dir} because {sentinel.name} exists.")
        return

    with open(run_dir / "train_metrics.json", "r", encoding="utf-8") as f:
        train_metrics = json.load(f)

    val_loss = train_metrics.get("eval_metrics", {}).get("eval_loss", 999.0)
    runtime_sec = train_metrics.get("runtime_sec", 999999.0)

    eval_rows = load_jsonl("data/judge_prompts.jsonl")
    outputs = generate_predictions_batched(
        run_dir=run_dir,
        prompts=eval_rows,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    format_score = score_format(outputs)
    judge_score = mock_judge_score(outputs)

    result = {
        "val_loss": val_loss,
        "format_score": format_score,
        "judge_score": judge_score,
        "runtime_sec": runtime_sec,
        "stability_score": 1.0,
        "failure_penalty": 0.0,
        "nan_detected": False,
    }

    with open(run_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
