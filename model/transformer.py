"""
model/transformer.py — Full 144M-parameter decoder-only transformer.

Architecture (all from TrainConfig / project_plan2.md):
  token embedding (wte)
  → N × TransformerBlock
      pre-norm (RMSNorm) → GQA → residual
      pre-norm (RMSNorm) → SwiGLU FFN → residual
  → final RMSNorm
  → tied LM head (shares weight with wte)

Initialization:
  - All Linear / Embedding weights: N(0, init_std=0.02)
  - Scaled residual projections (attn out_proj, FFN down_proj):
      std = init_std / sqrt(2 * n_layers)
    This prevents the residual stream from growing with depth.
  - Biases: n/a (all Linear layers are bias=False)
  - RMSNorm weights: initialised to 1 by nn.Parameter(torch.ones(...)) default

Weight-decay split (configure_optimizers):
  - decay group:    dim >= 2  → includes all weight matrices AND embeddings
  - no_decay group: dim <  2  → RMSNorm scales (1D), nothing else (no biases)
  Matches nanoGPT convention; embeddings ARE decayed (they are 2D).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import TrainConfig
from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.ffn import SwiGLUFFN
from model.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn      = GroupedQueryAttention(config)
        self.ffn_norm  = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn       = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual; cos/sin forwarded from Transformer level
        x = x + self.attn(self.attn_norm(x), cos, sin)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.wte    = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm   = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Single shared RotaryEmbedding — cos/sin computed once per forward pass
        # and passed into every block.  Avoids 16 redundant identical cache copies.
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Tied LM head: shares weight tensor with wte — no extra parameters.
        # nn.Linear stores weight as (out_features, in_features) = (vocab_size, d_model)
        # nn.Embedding stores weight as (num_embeddings, embedding_dim) = (vocab_size, d_model)
        # Same shape → direct tie works.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie

        # Apply weight initialisation
        self.apply(self._init_weights)

        # Scaled residual init: override std on out_proj and down_proj AFTER
        # apply() so the base init doesn't overwrite it.
        residual_std = config.init_std / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:     (B, T) token indices
            targets: (B, T) next-token targets for loss computation (optional)
        Returns:
            logits:  (B, T, vocab_size)
            loss:    scalar cross-entropy loss if targets provided, else None
        """
        x = self.wte(idx)  # (B, T, d_model)

        # Compute RoPE tables once for this sequence length, share across all layers
        cos, sin = self.rope.get_cos_sin(idx.shape[1])

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy in fp32 for numerical stability (softmax precision)
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_param_count(self) -> int:
        """Total trainable parameters. The tied lm_head.weight is counted once."""
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, config: TrainConfig) -> AdamW:
        """
        Split parameters into two AdamW groups:
          - decay    (dim >= 2): all weight matrices AND embeddings
          - no_decay (dim <  2): RMSNorm scales (1D vectors)

        Uses fused AdamW for lower kernel-launch overhead on CUDA.
        """
        decay_params    = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() <  2]

        # Sanity check: lm_head.weight == wte.weight (tied), so it appears once
        # in parameters() — confirm we're not double-counting.
        assert self.lm_head.weight.data_ptr() == self.wte.weight.data_ptr(), \
            "Embedding tie broken — lm_head.weight and wte.weight must share storage."

        optim_groups = [
            {"params": decay_params,    "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # fused=True: single fused CUDA kernel per step, faster than the default
        # elementwise loop.  Requires CUDA; falls back gracefully if unavailable.
        use_fused = torch.cuda.is_available()
        optimizer = AdamW(
            optim_groups,
            lr=config.peak_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            fused=use_fused,
        )

        n_decay    = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(
            f"Optimizer: {len(decay_params)} decay tensors ({n_decay:,} params), "
            f"{len(no_decay_params)} no-decay tensors ({n_no_decay:,} params). "
            f"fused={use_fused}"
        )

        return optimizer
