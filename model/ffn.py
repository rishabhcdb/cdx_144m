"""
model/ffn.py — SwiGLU Feed-Forward Network.

Architecture (identical to LLaMA):
    hidden = SiLU(gate_proj(x)) * up_proj(x)
    output = down_proj(hidden)

Where:
  gate_proj: d_model → ffn_hidden_dim
  up_proj:   d_model → ffn_hidden_dim
  down_proj: ffn_hidden_dim → d_model

Scaled residual init is applied to down_proj.weight by Transformer._init_weights
after construction:  std = init_std / sqrt(2 * n_layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


class SwiGLUFFN(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) ⊗ up → down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
