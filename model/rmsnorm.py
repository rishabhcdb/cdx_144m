"""
model/rmsnorm.py — RMSNorm with fp32 variance computation.

Key implementation notes:
- Variance is computed in fp32 regardless of input dtype to prevent underflow
  in bf16 training (squared small values can flush to zero in bf16).
- eps=1e-5 matches the LlamaConfig default used in the validation suite, so
  torch.allclose failures are real implementation divergences, not epsilon drift.
- No mean subtraction (RMSNorm, not LayerNorm).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 for numerically stable variance computation
        x_fp32 = x.float()
        # rms = 1 / sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Cast normalised result back to the input dtype, then scale
        return (x_fp32 * rms).to(x.dtype) * self.weight
