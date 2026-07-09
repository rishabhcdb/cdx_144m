"""
model/rope.py — Rotary Position Embeddings (RoPE), θ=10000.

Implementation notes:
- cos/sin tables are precomputed at __init__ time and registered as non-persistent
  buffers so they move with the module to the correct device but are NOT saved in
  state_dict (they're deterministically recomputable).
- rotate_half splits the last dimension in two halves and applies the rotation:
    rotated = [−x2, x1] (standard formulation used by HF Llama)
- Both q and k are rotated; only q and k, not v.
- The cache shape is (1, 1, max_seq_len, head_dim) to broadcast cleanly over
  (batch, n_heads, seq_len, head_dim).
- get_cos_sin() retrieves precomputed tables for external application.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # inv_freq: (head_dim // 2,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device = None):
        """Precompute cos/sin tables for all positions up to seq_len."""
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # outer product → (seq_len, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # concat so last dim = head_dim  (matches rotate_half split)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        # Shape: (1, 1, seq_len, head_dim) for broadcasting over (B, H, T, D)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (cos, sin) sliced to seq_len.
        Call once per Transformer forward pass; pass the result down to each
        attention layer so the tables are computed only once, not per-layer.
        """
        return (
            self.cos_cached[:, :, :seq_len, :],  # (1, 1, T, head_dim)
            self.sin_cached[:, :, :seq_len, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: [x1, x2] → [−x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a query or key tensor (public API)."""
    return (x * cos) + (_rotate_half(x) * sin)
