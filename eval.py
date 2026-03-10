import argparse
import json
import math
import os
import re
import unicodedata
from pathlib import Path

import torch
from unsloth import FastLanguageModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Format scoring (unchanged)
# ---------------------------------------------------------------------------

def score_format(outputs):
    total = len(outputs)
    good = 0
    for row in outputs:
        text = row["prediction"].strip()
        if text:
            good += 1
    return good / max(total, 1)


# ---------------------------------------------------------------------------
# Judge: token-based metrics (lightweight, deterministic)
# ---------------------------------------------------------------------------

def _normalize(text):
    """Lower-case, strip accents, collapse whitespace, remove punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text):
    return _normalize(text).split()


def compute_token_f1(prediction, ideal):
    """Word-level F1 between prediction and ideal."""
    pred_tokens = _tokenize(prediction)
    ideal_tokens = _tokenize(ideal)
    if not ideal_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in set(ideal_tokens))
    precision = common / len(pred_tokens)
    recall = common / len(ideal_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_keyword_recall(prediction, ideal):
    """Fraction of unique content words from ideal present in prediction."""
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "as", "into", "about", "and",
        "or", "but", "not", "no", "if", "it", "its", "that", "this", "than",
    }
    ideal_words = set(_tokenize(ideal)) - stop
    if not ideal_words:
        return 1.0
    pred_words = set(_tokenize(prediction))
    hits = ideal_words & pred_words
    return len(hits) / len(ideal_words)


def compute_length_ratio_score(prediction, ideal):
    """Score based on relative length to ideal (penalise very short/long)."""
    pred_len = max(len(prediction.split()), 1)
    ideal_len = max(len(ideal.split()), 1)
    ratio = pred_len / ideal_len
    if ratio < 0.25:
        return ratio / 0.25 * 0.5
    if ratio > 4.0:
        return max(0.0, 1.0 - (ratio - 4.0) / 4.0)
    if 0.5 <= ratio <= 2.0:
        return 1.0
    if ratio < 0.5:
        return 0.5 + (ratio - 0.25) / 0.25 * 0.5
    return 1.0 - (ratio - 2.0) / 2.0 * 0.3


def token_judge_score(outputs):
    """Deterministic judge using token overlap metrics (no model needed)."""
    scores = []
    for row in outputs:
        prediction = row["prediction"].strip()
        ideal = row.get("ideal", "").strip()
        if not prediction:
            scores.append(0.0)
            continue
        if not ideal:
            scores.append(min(len(prediction) / 200.0, 1.0))
            continue
        f1 = compute_token_f1(prediction, ideal)
        kw = compute_keyword_recall(prediction, ideal)
        lr = compute_length_ratio_score(prediction, ideal)
        score = 0.50 * f1 + 0.30 * kw + 0.20 * lr
        scores.append(score)
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Judge: LLM-as-judge (optional, uses the loaded model)
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a strict evaluator. Rate the answer quality from 0 to 10.\n"
    "Criteria: correctness, completeness, conciseness.\n"
    "Respond with ONLY a single integer between 0 and 10.<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: {question}\n"
    "Reference answer: {ideal}\n"
    "Model answer: {prediction}\n"
    "Score (0-10):<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _parse_judge_rating(text):
    """Extract the first integer 0-10 from model output."""
    for m in re.finditer(r"\b(\d{1,2})\b", text):
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val / 10.0
    return 0.5


def _extract_question(prompt):
    """Pull the user question out of a ChatML prompt string."""
    parts = prompt.split("<|im_start|>user\n")
    if len(parts) < 2:
        return prompt
    return parts[-1].split("<|im_end|>")[0].strip()


def llm_judge_score(model, tokenizer, outputs, device, batch_size=4, max_new_tokens=16):
    """Use the same fine-tuned model to rate each prediction against its ideal."""
    judge_prompts = []
    for row in outputs:
        question = _extract_question(row["prompt"])
        judge_prompts.append(
            _JUDGE_TEMPLATE.format(
                question=question,
                ideal=row.get("ideal", ""),
                prediction=row["prediction"].strip(),
            )
        )

    scores = []
    for start in range(0, len(judge_prompts), batch_size):
        batch = judge_prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
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

        for i in range(len(batch)):
            new_tokens = gen[i, int(input_lengths[i]) :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            scores.append(_parse_judge_rating(response))

    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Combined judge
# ---------------------------------------------------------------------------

def judge_score(outputs, model=None, tokenizer=None, device=None, use_llm_judge=False):
    """Combined judge score.

    Default mode uses fast token-based metrics only (no extra VRAM).
    With ``--use-llm-judge`` the loaded model also rates predictions,
    blended 60 % token / 40 % LLM.
    """
    tok_score = token_judge_score(outputs)

    if use_llm_judge and model is not None and tokenizer is not None:
        llm_score = llm_judge_score(model, tokenizer, outputs, device)
        return 0.60 * tok_score + 0.40 * llm_score

    return tok_score


# ---------------------------------------------------------------------------
# Prediction generation (loads model once, optionally runs LLM judge)
# ---------------------------------------------------------------------------

def generate_predictions_and_judge(
    run_dir: Path,
    prompts,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    max_input_length: int = 1536,
    max_new_tokens: int = 256,
    use_llm_judge: bool = False,
):
    """Load model, generate predictions, run judge, return outputs + score."""
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

    j_score = judge_score(
        outputs,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_llm_judge=use_llm_judge,
    )

    return outputs, j_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=False,
        help="Also run the loaded model as an LLM judge (adds latency).",
    )
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
    outputs, j_score = generate_predictions_and_judge(
        run_dir=run_dir,
        prompts=eval_rows,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        use_llm_judge=args.use_llm_judge,
    )

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    format_sc = score_format(outputs)

    result = {
        "val_loss": val_loss,
        "format_score": format_sc,
        "judge_score": j_score,
        "runtime_sec": runtime_sec,
        "stability_score": 1.0,
        "failure_penalty": 0.0,
        "nan_detected": False,
    }

    with open(run_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
