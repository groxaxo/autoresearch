import json
import subprocess
from pathlib import Path


def run(cmd):
    print(">", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def append_result(row, path="results.jsonl"):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def infer_baseline_loss(results):
    finite_losses = [x["val_loss"] for x in results if isinstance(x.get("val_loss"), (int, float)) and x["val_loss"] < 900]
    if not finite_losses:
        return 2.0
    return min(finite_losses)


def load_results(path="results.jsonl"):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    run(["python", "propose_next.py"])
    cfgs = sorted(Path("configs").glob("exp_*.yaml"))
    cfg = str(cfgs[-1])

    run(["python", "validate_config.py", cfg])
    run(["python", "train_qwen35_unsloth.py", "--config", cfg])

    run_name = Path(cfg).stem
    run_dir = Path(f"adapters/{run_name}")

    if not (run_dir / "SKIP_EVAL_NONFINITE").exists():
        run([
            "python",
            "eval.py",
            "--run_dir",
            str(run_dir),
            "--batch_size",
            "4",
        ])

    prior_results = load_results()
    baseline_loss = infer_baseline_loss(prior_results)
    run(["python", "score_run.py", "--run_dir", str(run_dir), "--baseline_loss", str(baseline_loss)])

    with open(run_dir / "final_score.json", "r", encoding="utf-8") as f:
        score = json.load(f)

    append_result(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "config_path": cfg,
            "composite_score": score["composite_score"],
            "judge_score": score["judge_score"],
            "format_score": score["format_score"],
            "val_loss": score["val_loss"],
            "runtime_sec": score["runtime_sec"],
            "failure_penalty": score.get("failure_penalty", 0.0),
        }
    )


if __name__ == "__main__":
    main()
