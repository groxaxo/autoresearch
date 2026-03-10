import sys
import os
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python validate_config.py <config.yaml>")

    cfg = load_yaml(sys.argv[1])
    space = load_yaml("search_space.yaml")
    s = space["search_space"]
    c = space["constraints"]

    assert cfg["learning_rate"] in s["lr"]
    assert cfg["lora"]["r"] in s["lora_r"]
    assert cfg["lora"]["alpha"] in s["lora_alpha"]
    assert cfg["lora"]["dropout"] in s["lora_dropout"]
    assert cfg["max_seq_length"] in s["max_seq_length"]
    assert cfg["per_device_train_batch_size"] in s["per_device_train_batch_size"]
    assert cfg["gradient_accumulation_steps"] in s["gradient_accumulation_steps"]
    assert cfg["packing"] in s["packing"]
    assert cfg["warmup_ratio"] in s["warmup_ratio"]
    assert cfg["weight_decay"] in s["weight_decay"]
    assert cfg["reasoning_ratio"] in s["reasoning_ratio"]
    assert cfg["lora"]["target_modules_set"] in s["target_modules_set"]
    assert cfg["dataset_mix_name"] in load_yaml("search_space.yaml")["dataset_mix_options"]

    micro_tokens = cfg["max_seq_length"] * cfg["per_device_train_batch_size"]
    step_tokens = micro_tokens * cfg["gradient_accumulation_steps"]
    gpu_vram_gb = int(os.environ.get("GPU_VRAM_GB", "0") or "0")
    gpu_micro_token_limits = {
        12: 2048,
        24: c["max_micro_tokens"],
    }
    max_micro_tokens = c["max_micro_tokens"]
    if gpu_vram_gb in gpu_micro_token_limits:
        max_micro_tokens = min(max_micro_tokens, gpu_micro_token_limits[gpu_vram_gb])

    assert micro_tokens <= max_micro_tokens, (
        f"OOM risk too high: micro_tokens={micro_tokens}, limit={max_micro_tokens}"
    )
    assert step_tokens <= c["max_step_tokens"], (
        f"Runtime risk too high: step_tokens={step_tokens}"
    )

    if cfg.get("search_phase", 1) == 1:
        assert cfg["lora"]["target_modules_set"] == "qkv_omlp"
        assert cfg["packing"] is True
        expected_phase1_seq = 1024 if gpu_vram_gb == 12 else 2048
        assert cfg["max_seq_length"] == expected_phase1_seq
        if gpu_vram_gb == 12:
            assert cfg["per_device_train_batch_size"] == 1

    print("OK")


if __name__ == "__main__":
    main()
