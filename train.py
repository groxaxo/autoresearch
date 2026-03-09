"""
Autoresearch pretraining script. Single-file, CUDA-only.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
       AUTORESEARCH_PROFILE=rtx3090x2 torchrun --standalone --nproc_per_node=2 train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict, replace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from kernels import get_kernel
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

fa3 = None


def init_flash_attention():
    global fa3
    if fa3 is None:
        cap = torch.cuda.get_device_capability()
        # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        fa3 = get_kernel(repo).flash_attn_interface
    return fa3

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class DistributedConfig:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main(self):
        return self.rank == 0


@dataclass(frozen=True)
class TrainingPreset:
    name: str = "default"
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"
    total_batch_size: int = 2**19
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple[float, float] = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    depth: int = 8
    device_batch_size: int = 128
    activation_checkpointing: bool = False
    compile_model: bool = True


DEFAULT_PRESET = TrainingPreset()
PRESETS = {
    "default": DEFAULT_PRESET,
    "rtx3090": replace(
        DEFAULT_PRESET,
        name="rtx3090",
        depth=10,
        device_batch_size=24,
        total_batch_size=2**18,
        window_pattern="L",
        activation_checkpointing=True,
        compile_model=False,
    ),
    "rtx3090x2": replace(
        DEFAULT_PRESET,
        name="rtx3090x2",
        depth=12,
        device_batch_size=24,
        total_batch_size=2**19,
        window_pattern="L",
        activation_checkpointing=True,
        compile_model=False,
    ),
}


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedConfig()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedConfig(rank=rank, local_rank=local_rank, world_size=world_size, enabled=True)


def resolve_training_preset(dist_cfg):
    profile = os.environ.get("AUTORESEARCH_PROFILE", "default").lower()
    profile = {
        "3090": "rtx3090",
        "3090x2": "rtx3090x2",
        "dual3090": "rtx3090x2",
    }.get(profile, profile)
    if profile == "rtx3090" and dist_cfg.world_size > 1:
        profile = "rtx3090x2"
    if profile not in PRESETS:
        raise ValueError(f"Unknown AUTORESEARCH_PROFILE={profile!r}. Choose from: {', '.join(sorted(PRESETS))}")
    preset = PRESETS[profile]
    return replace(
        preset,
        aspect_ratio=env_int("AUTORESEARCH_ASPECT_RATIO", preset.aspect_ratio),
        head_dim=env_int("AUTORESEARCH_HEAD_DIM", preset.head_dim),
        window_pattern=os.environ.get("AUTORESEARCH_WINDOW_PATTERN", preset.window_pattern),
        total_batch_size=env_int("AUTORESEARCH_TOTAL_BATCH_SIZE", preset.total_batch_size),
        embedding_lr=env_float("AUTORESEARCH_EMBEDDING_LR", preset.embedding_lr),
        unembedding_lr=env_float("AUTORESEARCH_UNEMBEDDING_LR", preset.unembedding_lr),
        matrix_lr=env_float("AUTORESEARCH_MATRIX_LR", preset.matrix_lr),
        scalar_lr=env_float("AUTORESEARCH_SCALAR_LR", preset.scalar_lr),
        weight_decay=env_float("AUTORESEARCH_WEIGHT_DECAY", preset.weight_decay),
        warmup_ratio=env_float("AUTORESEARCH_WARMUP_RATIO", preset.warmup_ratio),
        warmdown_ratio=env_float("AUTORESEARCH_WARMDOWN_RATIO", preset.warmdown_ratio),
        final_lr_frac=env_float("AUTORESEARCH_FINAL_LR_FRAC", preset.final_lr_frac),
        depth=env_int("AUTORESEARCH_DEPTH", preset.depth),
        device_batch_size=env_int("AUTORESEARCH_DEVICE_BATCH_SIZE", preset.device_batch_size),
        activation_checkpointing=env_flag(
            "AUTORESEARCH_ACTIVATION_CHECKPOINTING",
            preset.activation_checkpointing,
        ),
        compile_model=env_flag("AUTORESEARCH_COMPILE", preset.compile_model),
    )


def iter_rank_batches(loader, rank, world_size):
    for _ in range(rank):
        next(loader)
    while True:
        batch = next(loader)
        for _ in range(world_size - 1):
            next(loader)
        yield batch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_mean(tensor, dist_cfg):
    if not dist_cfg.enabled:
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist_cfg.world_size
    return reduced


def should_use_no_sync(dist_cfg, micro_step, grad_accum_steps):
    return dist_cfg.enabled and micro_step + 1 < grad_accum_steps


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = init_flash_attention().flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_checkpointing = config.activation_checkpointing
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.activation_checkpointing and self.training:
                def block_forward(x_in, cos, sin, block=block, layer_idx=i, window_size=self.window_sizes[i]):
                    ve = self.value_embeds[str(layer_idx)](idx) if str(layer_idx) in self.value_embeds else None
                    return block(x_in, ve, (cos, sin), window_size)
                x = checkpoint(block_forward, x, *cos_sin, use_reentrant=False)
            else:
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = DEFAULT_PRESET.aspect_ratio
HEAD_DIM = DEFAULT_PRESET.head_dim
WINDOW_PATTERN = DEFAULT_PRESET.window_pattern

# Optimization
TOTAL_BATCH_SIZE = DEFAULT_PRESET.total_batch_size
EMBEDDING_LR = DEFAULT_PRESET.embedding_lr
UNEMBEDDING_LR = DEFAULT_PRESET.unembedding_lr
MATRIX_LR = DEFAULT_PRESET.matrix_lr
SCALAR_LR = DEFAULT_PRESET.scalar_lr
WEIGHT_DECAY = DEFAULT_PRESET.weight_decay
ADAM_BETAS = DEFAULT_PRESET.adam_betas
WARMUP_RATIO = DEFAULT_PRESET.warmup_ratio
WARMDOWN_RATIO = DEFAULT_PRESET.warmdown_ratio
FINAL_LR_FRAC = DEFAULT_PRESET.final_lr_frac

# Model size
DEPTH = DEFAULT_PRESET.depth
DEVICE_BATCH_SIZE = DEFAULT_PRESET.device_batch_size


def build_model_config(depth, vocab_size, preset):
    base_dim = depth * preset.aspect_ratio
    model_dim = ((base_dim + preset.head_dim - 1) // preset.head_dim) * preset.head_dim
    num_heads = model_dim // preset.head_dim
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=preset.window_pattern,
        activation_checkpointing=preset.activation_checkpointing,
    )


def get_lr_multiplier(progress, preset):
    if progress < preset.warmup_ratio:
        return progress / preset.warmup_ratio if preset.warmup_ratio > 0 else 1.0
    elif preset.warmdown_ratio == 0 or progress < 1.0 - preset.warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / preset.warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * preset.final_lr_frac

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress, preset):
    return preset.weight_decay * (1 - progress)

def main():
    # -----------------------------------------------------------------------
    # Setup: tokenizer, model, optimizer, dataloader
    # -----------------------------------------------------------------------
    dist_cfg = init_distributed()
    try:
        t_start = time.time()
        seed = 42 + dist_cfg.rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda", dist_cfg.local_rank)
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        H100_BF16_PEAK_FLOPS = 989.5e12
        preset = resolve_training_preset(dist_cfg)

        def log(*args, **kwargs):
            if dist_cfg.is_main:
                print(*args, **kwargs)

        tokenizer = Tokenizer.from_directory()
        vocab_size = tokenizer.get_vocab_size()
        log(f"Vocab size: {vocab_size:,}")
        log(f"Training preset: {preset.name}")
        log(f"Distributed: {dist_cfg.enabled} (world_size={dist_cfg.world_size})")

        config = build_model_config(preset.depth, vocab_size, preset)
        log(f"Model config: {asdict(config)}")

        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()

        param_counts = model.num_scaling_params()
        log("Parameter counts:")
        for key, value in param_counts.items():
            log(f"  {key:24s}: {value:,}")
        num_params = param_counts['total']
        num_flops_per_token = model.estimate_flops()
        log(f"Estimated FLOPs per token: {num_flops_per_token:e}")

        tokens_per_fwdbwd = preset.device_batch_size * MAX_SEQ_LEN * dist_cfg.world_size
        assert preset.total_batch_size % tokens_per_fwdbwd == 0
        grad_accum_steps = preset.total_batch_size // tokens_per_fwdbwd

        optimizer = model.setup_optimizer(
            unembedding_lr=preset.unembedding_lr,
            embedding_lr=preset.embedding_lr,
            scalar_lr=preset.scalar_lr,
            adam_betas=preset.adam_betas,
            matrix_lr=preset.matrix_lr,
            weight_decay=preset.weight_decay,
        )

        if preset.compile_model:
            model = torch.compile(model, dynamic=False)
        if dist_cfg.enabled:
            model = DDP(model, device_ids=[dist_cfg.local_rank], output_device=dist_cfg.local_rank, broadcast_buffers=False)

        train_loader = iter_rank_batches(
            make_dataloader(tokenizer, preset.device_batch_size, MAX_SEQ_LEN, "train"),
            dist_cfg.rank,
            dist_cfg.world_size,
        )
        x, y, epoch = next(train_loader)

        log(f"Time budget: {TIME_BUDGET}s")
        log(f"Gradient accumulation steps: {grad_accum_steps}")

        # -------------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------------
        t_start_training = time.time()
        smooth_train_loss = 0
        total_training_time = 0
        step = 0

        while True:
            torch.cuda.synchronize(device)
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                sync_context = model.no_sync() if should_use_no_sync(dist_cfg, micro_step, grad_accum_steps) else nullcontext()
                with sync_context:
                    with autocast_ctx:
                        loss = model(x, y)
                    train_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                x, y, epoch = next(train_loader)

            # Progress and schedules
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lrm = get_lr_multiplier(progress, preset)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(progress, preset)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group['kind'] == 'muon':
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()
            model.zero_grad(set_to_none=True)

            train_loss_f = reduce_mean(train_loss, dist_cfg).item()

            # Fast fail: abort if loss is exploding
            if train_loss_f > 100:
                log("FAIL")
                raise SystemExit(1)

            torch.cuda.synchronize(device)
            t1 = time.time()
            dt = t1 - t0

            if step > 10:
                total_training_time += dt

            # Logging
            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
            pct_done = 100 * progress
            tok_per_sec = int(preset.total_batch_size / dt)
            mfu = 100 * num_flops_per_token * preset.total_batch_size / dt / (H100_BF16_PEAK_FLOPS * dist_cfg.world_size)
            remaining = max(0, TIME_BUDGET - total_training_time)

            log(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

            # GC management (Python's GC causes ~500ms stalls)
            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()

            step += 1

            # Time's up — but only stop after warmup steps so we don't count compilation
            if step > 10 and total_training_time >= TIME_BUDGET:
                break

        log()  # newline after \r training log

        total_tokens = step * preset.total_batch_size

        if dist_cfg.enabled:
            dist.barrier()

        # Final eval
        val_bpb = float("nan")
        if dist_cfg.is_main:
            model.eval()
            with autocast_ctx:
                val_bpb = evaluate_bpb(unwrap_model(model), tokenizer, preset.device_batch_size)

        # Final summary
        t_end = time.time()
        steady_state_mfu = 100 * num_flops_per_token * preset.total_batch_size * max(step - 10, 0) / total_training_time / (H100_BF16_PEAK_FLOPS * dist_cfg.world_size) if total_training_time > 0 else 0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        log("---")
        log(f"val_bpb:          {val_bpb:.6f}")
        log(f"training_seconds: {total_training_time:.1f}")
        log(f"total_seconds:    {t_end - t_start:.1f}")
        log(f"peak_vram_mb:     {peak_vram_mb:.1f}")
        log(f"mfu_percent:      {steady_state_mfu:.2f}")
        log(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
        log(f"num_steps:        {step}")
        log(f"num_params_M:     {num_params / 1e6:.1f}")
        log(f"depth:            {preset.depth}")
    finally:
        if dist_cfg.enabled and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
