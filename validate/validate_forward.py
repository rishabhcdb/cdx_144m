"""
validate/validate_forward.py — Full forward pass validation vs HF LlamaModel.

After the per-component tests in validate_components.py, this test validates
the full assembled model: token embedding → N blocks → final norm → logits.

Strategy:
  - Randomly initialize both our Transformer and a HF LlamaForCausalLM
    at the same dimensions.
  - Copy all weights layer by layer.
  - Run the same token sequence through both.
  - Assert torch.allclose on the output logits (fp32, atol=1e-5).

This catches integration bugs that individual component tests won't catch,
e.g. wrong block ordering, missing final norm, tied embedding not applied.

Usage:
    cd <project_root>
    python validate/validate_forward.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from config import TrainConfig
from model.transformer import Transformer

ATOL = 1e-5
RTOL = 1e-4

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def build_hf_causal(our_cfg: TrainConfig) -> LlamaForCausalLM:
    hf_cfg = LlamaConfig(
        hidden_size=our_cfg.d_model,
        num_hidden_layers=our_cfg.n_layers,
        num_attention_heads=our_cfg.n_heads,
        num_key_value_heads=our_cfg.n_kv_heads,
        intermediate_size=our_cfg.ffn_hidden_dim,
        vocab_size=our_cfg.vocab_size,
        rope_theta=our_cfg.rope_theta,
        rms_norm_eps=our_cfg.rms_norm_eps,
        hidden_act="silu",
        max_position_embeddings=our_cfg.max_seq_len,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(99)
    return LlamaForCausalLM(hf_cfg).float().eval()


def copy_weights(our_model: Transformer, hf_model: LlamaForCausalLM):
    """
    Copy weights from HF model into our model layer by layer.
    Name mapping:
      HF                                  Ours
      model.embed_tokens.weight        →  wte.weight  (also lm_head via tie)
      model.layers[i].input_layernorm  →  blocks[i].attn_norm
      model.layers[i].self_attn.q_proj →  blocks[i].attn.q_proj
      model.layers[i].self_attn.k_proj →  blocks[i].attn.k_proj
      model.layers[i].self_attn.v_proj →  blocks[i].attn.v_proj
      model.layers[i].self_attn.o_proj →  blocks[i].attn.out_proj
      model.layers[i].post_attn_norm   →  blocks[i].ffn_norm
      model.layers[i].mlp.gate_proj    →  blocks[i].ffn.gate_proj
      model.layers[i].mlp.up_proj      →  blocks[i].ffn.up_proj
      model.layers[i].mlp.down_proj    →  blocks[i].ffn.down_proj
      model.norm.weight                →  norm.weight
      lm_head.weight (tied to embed)   →  lm_head.weight (tied to wte)
    """
    hf = hf_model.model  # LlamaModel (strips the lm_head wrapper)

    our_model.wte.weight.data.copy_(hf.embed_tokens.weight.data)

    for i, (our_block, hf_layer) in enumerate(zip(our_model.blocks, hf.layers)):
        our_block.attn_norm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        our_block.attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        our_block.attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        our_block.attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        our_block.attn.out_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
        our_block.ffn_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
        our_block.ffn.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
        our_block.ffn.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
        our_block.ffn.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)

    our_model.norm.weight.data.copy_(hf.norm.weight.data)
    # lm_head is tied to wte — already updated above


def main():
    print("=" * 60)
    print("Full forward pass validation vs HF LlamaForCausalLM")
    print(f"Both sides: fp32  |  atol={ATOL}  rtol={RTOL}")
    print("=" * 60)

    our_cfg   = TrainConfig()
    hf_model  = build_hf_causal(our_cfg)

    # Build our model with same random seed so init matches, then overwrite weights
    torch.manual_seed(99)
    our_model = Transformer(our_cfg).float().eval()

    print(f"\nOur param count:  {our_model.get_param_count():,}")
    print(f"HF param count:   {sum(p.numel() for p in hf_model.parameters()):,}")

    copy_weights(our_model, hf_model)

    # Run same token sequence through both
    T = 64
    torch.manual_seed(42)
    idx = torch.randint(0, our_cfg.vocab_size, (2, T))

    with torch.no_grad():
        our_logits, _ = our_model(idx)
        hf_logits     = hf_model(idx).logits

    max_diff = (our_logits - hf_logits).abs().max().item()
    ok       = torch.allclose(our_logits, hf_logits, atol=ATOL, rtol=RTOL)
    tag      = PASS if ok else FAIL

    print(f"\n  [{tag}] Full forward pass logits  max_diff={max_diff:.2e}")
    print("\n" + "=" * 60)

    if ok:
        print("Full forward pass matches HF reference. Architecture is correct.")
        sys.exit(0)
    else:
        print("Mismatch detected. Investigate per-component tests first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
