"""
config.py — Single source of truth for all hyperparameters.
Every value maps directly to project_plan2.md.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size: int = 32000           # Llama-2 tokenizer
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12                 # query heads
    n_kv_heads: int = 4              # key/value heads  (GQA 3:1 ratio)
    head_dim: int = 64               # d_model / n_heads = 768 / 12 = 64 ✓
    ffn_hidden_dim: int = 2560       # SwiGLU hidden dim
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5       # matches LlamaConfig(rms_norm_eps=1e-5) used in
                                     # the validation suite — critical for torch.allclose
    tie_word_embeddings: bool = True
    max_seq_len: int = 2048

    # ── Initialization ────────────────────────────────────────────────────────
    init_std: float = 0.02
    # out_proj (attn) and down_proj (FFN) use scaled residual init:
    #   std = init_std / sqrt(2 * n_layers)
    # Applied in Transformer._init_weights after module construction.

    # ── Sequence / Batching ───────────────────────────────────────────────────
    seq_len: int = 2048
    micro_batch_size: int = 16
    grad_accum_steps: int = 16
    # effective_batch_tokens = 16 * 16 * 2048 = 524,288 ✓

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 1337                 # fixes both data shuffle + weight init

    # ── Training schedule ─────────────────────────────────────────────────────
    total_steps: int = 9537
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 350
    lr_schedule: str = "cosine"

    # ── Optimizer ─────────────────────────────────────────────────────────────
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    # weight_decay applied to dim>=2 params (incl. embeddings); NOT to dim<2
    # (biases, RMSNorm scales).  Matches nanoGPT convention.
    grad_clip_max_norm: float = 1.0

    # ── Precision / Compile ───────────────────────────────────────────────────
    compile_model: bool = True       # torch.compile — measured 47% speedup on H100
    allow_tf32: bool = True          # matmul + cudnn TF32; free throughput on Ampere+
    mixed_precision: str = "bf16"    # forward/backward dtype
    grad_accum_dtype: str = "fp32"   # accumulated gradients stay in fp32

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_every_steps: int = 500
    keep_last_n: int = 3             # rolling window of recent checkpoints
    keep_best_val: bool = True       # additionally keep best-val checkpoint
    save_optimizer_state: bool = True  # required for correct Adam resume
    checkpoint_every_minutes: int = 30  # time-based safety trigger (mid-interval crash)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_steps: int = 10        # train loss, grad norm, LR
    eval_every_steps: int = 500      # val loss + perplexity on held-out split
    log_backend: str = "csv"         # "csv" | "wandb"

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: str = "data/shards"
    val_dir: str = "data/shards"      # glob val_*.bin here at eval time
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tokenizer_id: str = "meta-llama/Llama-2-7b-hf"  # HF token required; set in .env
