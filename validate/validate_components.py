"""
validate/validate_components.py — Per-component allclose checks vs HF Llama reference.

Strategy (no gated-license download):
  - Instantiate LlamaConfig at our exact small dimensions (hidden_size=768, etc.)
  - Build a randomly-initialized LlamaModel from that config
  - Copy matching weights from HF reference into our implementation
  - Both sides run in fp32 for clean, unambiguous allclose comparison
    (avoids dtype-mismatch false failures from bf16 vs fp32 near-misses)
  - Tolerance: atol=1e-5, rtol=1e-4 (strict fp32 defaults — much tighter than
    the bf16-appropriate atol=1e-3 that would be needed if one side were bf16)

Components tested:
  1. RMSNorm
  2. SwiGLU FFN (gate, up, down projections)
  3. GQA Attention (q, k, v projections + out_proj + RoPE)
     Note: RoPE is validated implicitly as part of the attention test since
     HF's LlamaAttention applies it internally during forward().

Usage:
    cd <project_root>
    python validate/validate_components.py
    # Expected output: PASS for all 3 components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel

from config import TrainConfig
from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.ffn import SwiGLUFFN

# ── Tolerance ─────────────────────────────────────────────────────────────────
# Both sides run in fp32.  Use standard fp32 tolerance; this is intentionally
# tighter than the bf16-appropriate atol=1e-3 to catch real bugs.
ATOL = 1e-5
RTOL = 1e-4

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def build_hf_reference(our_config: TrainConfig) -> tuple[LlamaConfig, LlamaModel]:
    """
    Build a randomly-initialized HF LlamaModel at our exact dimensions.
    No pretrained weights downloaded — just the architecture code.
    """
    hf_cfg = LlamaConfig(
        hidden_size=our_config.d_model,
        num_hidden_layers=our_config.n_layers,
        num_attention_heads=our_config.n_heads,
        num_key_value_heads=our_config.n_kv_heads,
        intermediate_size=our_config.ffn_hidden_dim,
        vocab_size=our_config.vocab_size,
        rope_theta=our_config.rope_theta,
        rms_norm_eps=our_config.rms_norm_eps,
        hidden_act="silu",
        max_position_embeddings=our_config.max_seq_len,
        attention_bias=False,
        mlp_bias=False,
    )
    # Random init (no download) — we only need the architecture, not pretrained weights
    torch.manual_seed(42)
    hf_model = LlamaModel(hf_cfg).float().eval()
    return hf_cfg, hf_model


def check_allclose(name: str, ours: torch.Tensor, ref: torch.Tensor) -> bool:
    max_diff = (ours - ref).abs().max().item()
    ok = torch.allclose(ours, ref, atol=ATOL, rtol=RTOL)
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name:<30s}  max_diff={max_diff:.2e}  (atol={ATOL}, rtol={RTOL})")
    return ok


# ── Test 1: RMSNorm ───────────────────────────────────────────────────────────

def test_rmsnorm(our_cfg: TrainConfig, hf_model: LlamaModel) -> bool:
    print("\n--- RMSNorm ---")
    # Use the first layer's input_layernorm from the HF model as reference
    hf_norm = hf_model.layers[0].input_layernorm.float()

    our_norm = RMSNorm(our_cfg.d_model, eps=our_cfg.rms_norm_eps).float()
    # Copy weight
    our_norm.weight.data.copy_(hf_norm.weight.data)

    torch.manual_seed(42)
    x = torch.randn(2, 128, our_cfg.d_model)  # (B, T, d_model)

    with torch.no_grad():
        ref_out  = hf_norm(x)
        our_out  = our_norm(x)

    return check_allclose("RMSNorm forward", our_out, ref_out)


# ── Test 2: SwiGLU FFN ────────────────────────────────────────────────────────

def test_ffn(our_cfg: TrainConfig, hf_model: LlamaModel) -> bool:
    print("\n--- SwiGLU FFN ---")
    hf_mlp = hf_model.layers[0].mlp.float()

    our_ffn = SwiGLUFFN(our_cfg).float()
    # HF: gate_proj, up_proj, down_proj — same names
    our_ffn.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
    our_ffn.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
    our_ffn.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)

    torch.manual_seed(42)
    x = torch.randn(2, 128, our_cfg.d_model)

    with torch.no_grad():
        ref_out = hf_mlp(x)
        our_out = our_ffn(x)

    return check_allclose("SwiGLU FFN forward", our_out, ref_out)


# ── Test 3: GQA Attention (includes RoPE implicitly) ─────────────────────────

def test_attention(our_cfg: TrainConfig, hf_model: LlamaModel) -> bool:
    """
    Tests GQA attention + RoPE together.
    HF's LlamaAttention applies RoPE internally, so matching its output validates
    both the projection weights and the rotary embedding implementation.

    Newer transformers API (≥ 4.43):
      - RoPE is pre-computed at the model level via model.rotary_emb(x, position_ids)
      - Passed into attention as position_embeddings=(cos, sin)
      - Return is a 2-tuple (output, attn_weights) — not the old 3-tuple
    """
    print("\n--- GQA Attention (incl. RoPE) ---")
    hf_attn = hf_model.layers[0].self_attn.float()

    our_attn = GroupedQueryAttention(our_cfg).float()
    # Weight mapping: HF uses q_proj, k_proj, v_proj, o_proj
    our_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
    our_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
    our_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
    our_attn.out_proj.weight.data.copy_(hf_attn.o_proj.weight.data)

    T = 64
    torch.manual_seed(42)
    x = torch.randn(2, T, our_cfg.d_model)

    with torch.no_grad():
        position_ids = torch.arange(T).unsqueeze(0).expand(2, -1)
        # Compute cos/sin using HF's own rotary module — same source of truth
        # for both sides of the comparison.
        # HF rotary_emb returns (B, T, head_dim); our apply_rotary_emb expects
        # (1, 1, T, head_dim) to broadcast over (B, n_heads, T, head_dim).
        # Reshape here for the test only — in production our rope.get_cos_sin()
        # already returns the correct (1, 1, T, head_dim) shape.
        cos, sin = hf_model.rotary_emb(x, position_ids)  # (B, T, head_dim)
        cos = cos[:1].unsqueeze(1)   # (1, 1, T, head_dim)
        sin = sin[:1].unsqueeze(1)
        # New API: pass as position_embeddings; returns (output, attn_weights)
        ref_out, _ = hf_attn(x, attention_mask=None, position_ids=position_ids,
                              position_embeddings=(cos.squeeze(0), sin.squeeze(0)))
        # Our model receives the correctly-shaped cos/sin directly
        our_out = our_attn(x, cos, sin)

    return check_allclose("GQA Attention + RoPE forward", our_out, ref_out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Component validation vs HF LlamaModel (randomly initialized)")
    print(f"Both sides: fp32  |  atol={ATOL}  rtol={RTOL}")
    print("=" * 60)

    our_cfg = TrainConfig()
    torch.manual_seed(42)
    hf_cfg, hf_model = build_hf_reference(our_cfg)
    print(f"\nHF LlamaConfig: hidden={hf_cfg.hidden_size}, layers={hf_cfg.num_hidden_layers}, "
          f"heads={hf_cfg.num_attention_heads}, kv_heads={hf_cfg.num_key_value_heads}")

    results = {
        "RMSNorm":         test_rmsnorm(our_cfg, hf_model),
        "SwiGLU FFN":      test_ffn(our_cfg, hf_model),
        "GQA Attn + RoPE": test_attention(our_cfg, hf_model),
    }

    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name}")
    print("=" * 60)

    if not all_pass:
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed components: {failed}")
        print("Investigate before starting pretraining.")
        sys.exit(1)
    else:
        print("\nAll components match HF reference. Safe to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()
