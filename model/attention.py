"""
model/attention.py — Grouped Query Attention (GQA) with RoPE.

Configuration (from TrainConfig):
  n_heads    = 12  (query heads)
  n_kv_heads = 4   (key/value heads, 3:1 GQA ratio)
  head_dim   = 64
  d_model    = 768

SDPA call uses:
  F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

Shape contract for enable_gqa=True (PyTorch ≥ 2.5):
  q:  (B, n_heads,    T, head_dim)
  k:  (B, n_kv_heads, T, head_dim)
  v:  (B, n_kv_heads, T, head_dim)
  n_heads % n_kv_heads == 0  (ratio = 3)

RoPE is NOT owned by this module.  A single RotaryEmbedding lives on the
Transformer and passes (cos, sin) into each layer's forward() call — avoids
16 redundant identical cache copies.

Scaled residual init on out_proj.weight is applied by Transformer._init_weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig
from model.rope import apply_rotary_emb

# ── Version guard ─────────────────────────────────────────────────────────────
# enable_gqa=True in F.scaled_dot_product_attention was added in PyTorch 2.5.
# Use tuple comparison — string comparison fails for versions like 2.10 vs 2.5.
def _parse_torch_minor(version_str: str) -> tuple[int, int]:
    """Parse 'major.minor' from a torch version string, stripping suffixes."""
    clean = version_str.split("+")[0]   # drop local build tag (e.g. "+cu118")
    for sep in ("a", "b", "rc", "dev"):
        clean = clean.split(sep)[0]     # drop pre-release suffix
    parts = clean.split(".")
    return int(parts[0]), int(parts[1])

_TORCH_VERSION = _parse_torch_minor(torch.__version__)
if _TORCH_VERSION < (2, 5):
    raise RuntimeError(
        f"torch >= 2.5.0 is required for F.scaled_dot_product_attention"
        f"(enable_gqa=True), but found torch=={torch.__version__}.\n"
        f"Upgrade with: pip install 'torch>=2.5.0'"
    )
# ─────────────────────────────────────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.head_dim

        assert config.n_heads % config.n_kv_heads == 0, (
            f"n_heads ({config.n_heads}) must be divisible by "
            f"n_kv_heads ({config.n_kv_heads}) for GQA"
        )

        self.q_proj   = nn.Linear(config.d_model, config.n_heads    * config.head_dim, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads  * config.head_dim, config.d_model,   bias=False)
        # No self.rope here — one shared RotaryEmbedding lives on the Transformer.

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, T, d_model)
            cos: (1, 1, T, head_dim) — from Transformer-level RotaryEmbedding
            sin: (1, 1, T, head_dim)
        """
        B, T, _ = x.shape

        # Project → reshape to (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k (not v); cos/sin shared from Transformer level
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # GQA attention — PyTorch handles the KV head broadcasting internally
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # Merge heads and project out
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)
