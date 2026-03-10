import copy
import json
import random
from pathlib import Path

import yaml


MUTABLE_PATHS_PHASE1 = [
    ("learning_rate",),
    ("lora", "r"),
    ("lora", "alpha"),
    ("lora", "dropout"),
    ("warmup_ratio",),
    ("weight_decay",),
    ("dataset_mix_name",),
    ("reasoning_ratio",),
]

MUTABLE_PATHS_PHASE2 = MUTABLE_PATHS_PHASE1 + [
    ("max_seq_length",),
    ("per_device_train_batch_size",),
    ("gradient_accumulation_steps",),
]

DATASET_MIX_CHOICES = ["default", "reasoning_heavy", "format_heavy"]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


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


def get_nested(cfg, path):
    cur = cfg
    for p in path:
        cur = cur[p]
    return cur


def set_nested(cfg, path, value):
    cur = cfg
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def choose_neighbor(value, choices, rng):
    idx = choices.index(value)
    candidates = [idx]
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(choices) - 1:
        candidates.append(idx + 1)
    return choices[rng.choice(candidates)]


def choose_parent(results, rng):
    if not results:
        return None
    ranked = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    if rng.random() < 0.75:
        return ranked[0]
    top_k = ranked[: min(3, len(ranked))]
    return rng.choice(top_k)


def mutate_config(cfg, space, run_idx, rng):
    cfg = copy.deepcopy(cfg)
    phase_paths = MUTABLE_PATHS_PHASE1 if run_idx <= 20 else MUTABLE_PATHS_PHASE2
    n_mutations = rng.choice([1, 1, 2, 2, 3])

    chosen = rng.sample(phase_paths, k=min(n_mutations, len(phase_paths)))

    mapping = {
        ("learning_rate",): space["lr"],
        ("lora", "r"): space["lora_r"],
        ("lora", "alpha"): space["lora_alpha"],
        ("lora", "dropout"): space["lora_dropout"],
        ("max_seq_length",): space["max_seq_length"],
        ("per_device_train_batch_size",): space["per_device_train_batch_size"],
        ("gradient_accumulation_steps",): space["gradient_accumulation_steps"],
        ("warmup_ratio",): space["warmup_ratio"],
        ("weight_decay",): space["weight_decay"],
        ("reasoning_ratio",): space["reasoning_ratio"],
        ("dataset_mix_name",): DATASET_MIX_CHOICES,
    }

    for path in chosen:
        current = get_nested(cfg, path)
        choices = mapping[path]
        if current in choices:
            new_value = choose_neighbor(current, choices, rng)
        else:
            new_value = rng.choice(choices)
        set_nested(cfg, path, new_value)

    if run_idx <= 20:
        cfg["search_phase"] = 1
        cfg["lora"]["target_modules_set"] = "qkv_omlp"
        cfg["packing"] = True
        cfg["max_seq_length"] = 2048
    else:
        cfg["search_phase"] = 2

    return cfg


def main():
    rng = random.Random(42)

    base = load_yaml("configs/base.yaml")
    space = load_yaml("search_space.yaml")["search_space"]
    results = load_results()

    parent = choose_parent(results, rng)
    if parent is None:
        cfg = copy.deepcopy(base)
    else:
        cfg = load_yaml(parent["config_path"])

    run_idx = len(results) + 1
    cfg = mutate_config(cfg, space, run_idx, rng)

    run_name = f"exp_{run_idx:04d}"
    cfg["run_name"] = run_name
    cfg["output_dir"] = f"adapters/{run_name}"

    out_path = f"configs/{run_name}.yaml"
    save_yaml(cfg, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
