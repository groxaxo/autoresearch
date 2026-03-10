import argparse
import json
from pathlib import Path


def relative_val_loss_score(val_loss: float, baseline_loss: float) -> float:
    if baseline_loss <= 0:
        return 0.0
    improvement = (baseline_loss - val_loss) / baseline_loss
    return max(0.0, min(1.0, 0.5 + improvement * 5.0))


def latency_penalty(runtime_sec: float) -> float:
    if runtime_sec <= 20 * 60:
        return 0.0
    return min(1.0, (runtime_sec - 20 * 60) / (10 * 60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline_loss", type=float, default=2.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    with open(run_dir / "eval_results.json", "r", encoding="utf-8") as f:
        r = json.load(f)

    weights = {
        "judge_score": 0.45,
        "format_score": 0.25,
        "val_loss_score": 0.20,
        "stability_score": 0.10,
        "latency_penalty": -0.10,
        "failure_penalty": -0.10,
    }

    val_loss_score = relative_val_loss_score(r["val_loss"], args.baseline_loss)
    lat_pen = latency_penalty(r["runtime_sec"])
    fail_pen = min(1.0, float(r.get("failure_penalty", 0.0)))

    composite = (
        weights["judge_score"] * float(r["judge_score"])
        + weights["format_score"] * float(r["format_score"])
        + weights["val_loss_score"] * val_loss_score
        + weights["stability_score"] * float(r["stability_score"])
        + weights["latency_penalty"] * lat_pen
        + weights["failure_penalty"] * fail_pen
    )

    out = {
        **r,
        "val_loss_score": val_loss_score,
        "latency_penalty_value": lat_pen,
        "failure_penalty_clamped": fail_pen,
        "composite_score": composite,
    }

    with open(run_dir / "final_score.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
