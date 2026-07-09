"""
train.py — Main pretraining script for the 144M-parameter transformer.

Execution order within this file:
  1. TF32 flags  (before any CUDA ops)
  2. Seed everything
  3. Model init
  4. Parameter count assertion (~144M)
  5. torch.compile(model)   ← single biggest lever: measured 47% speedup on H100
  6. Optimizer
  7. DataLoaders
  8. Training loop
     - bf16 autocast (forward + backward)
     - fp32 grad accumulation
     - Cosine LR schedule
     - Grad clipping
     - Eval every 500 steps  → val loss + perplexity
     - Step-based checkpoint every 500 steps
     - Time-based checkpoint safety (every 30 min between step saves)
     - Log every 10 steps

Usage:
    python train.py                         # full run (9,537 steps)
    python train.py --resume checkpoints/step_005000.pt
    python train.py --max_steps 100 --smoke_test    # smoke test

Smoke test pass criterion:
    Final loss must drop below ln(vocab_size) = ln(32000) ≈ 10.37.
    If it doesn't, training is diverging (bad init, LR, or data bug) — stop.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv

from checkpoint import CheckpointManager
from config import TrainConfig
from data.dataloader import ShardedDataLoader
from logger import Logger
from lr_schedule import get_lr
from model.transformer import Transformer

# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="144M-param LLM pretraining")
    p.add_argument("--resume",     type=str,  default=None,  help="Path to checkpoint to resume from")
    p.add_argument("--max_steps",  type=int,  default=None,  help="Override total_steps (e.g. for smoke test)")
    p.add_argument("--smoke_test", action="store_true",       help="Assert loss < ln(32000) at end and exit 0/1")
    return p.parse_args()


def eval_val_loss(
    model: torch.nn.Module,
    val_dir: str,
    config: TrainConfig,
    device: torch.device,
    n_batches: int = 20,
) -> tuple[float, float]:
    """
    Evaluate on the combined validation set.

    Globs all val_*.bin files in val_dir (val_fineweb.bin + val_stories.bin),
    concatenates their token arrays, and evaluates n_batches contiguous batches
    from the combined sequence.

    Returns (val_loss, perplexity).
    """
    import glob as _glob
    model.eval()

    # Discover and sort all val shards
    val_paths = sorted(_glob.glob(os.path.join(val_dir, "val_*.bin")))
    if not val_paths:
        raise FileNotFoundError(
            f"No val_*.bin files found in '{val_dir}'. "
            "Run prepare_fineweb.py and prepare_stories.py first."
        )

    # Concatenate all val shards into one array
    arrays = [
        np.memmap(p, dtype=np.uint16, mode="r").astype(np.int64)
        for p in val_paths
    ]
    val_tokens = np.concatenate(arrays)
    print(
        f"  [val] {len(val_paths)} shard(s): "
        + ", ".join(os.path.basename(p) for p in val_paths)
        + f"  total={len(val_tokens):,} tokens"
    )

    B, T = config.micro_batch_size, config.seq_len
    total_loss = 0.0
    n_evaluated = 0
    with torch.no_grad():
        for i in range(n_batches):
            offset = i * B * T
            if offset + B * T + 1 > len(val_tokens):
                break
            buf = val_tokens[offset : offset + B * T + 1]
            x   = torch.from_numpy(buf[:-1].reshape(B, T)).to(device)
            y   = torch.from_numpy(buf[1: ].reshape(B, T)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            total_loss  += loss.item()
            n_evaluated += 1

    model.train()
    val_loss = total_loss / max(n_evaluated, 1)
    val_ppl  = math.exp(val_loss)
    return val_loss, val_ppl


# ─────────────────────────────────────────────────────────────────────────────


def main():
    load_dotenv()
    args   = parse_args()
    config = TrainConfig()

    total_steps = args.max_steps if args.max_steps is not None else config.total_steps

    # ── 1. TF32 flags (must be set before any CUDA matmul) ────────────────────
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        print("TF32 enabled (matmul + cudnn)")

    # ── 2. Seed ────────────────────────────────────────────────────────────────
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"Seed: {config.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 3. Model ───────────────────────────────────────────────────────────────
    model = Transformer(config).to(device)
    n_params = model.get_param_count()
    print(f"Parameters: {n_params:,}  ({n_params/1e6:.2f}M)")

    # ── 4. Param count assertion ───────────────────────────────────────────────
    assert 143_000_000 <= n_params <= 146_000_000, (
        f"Unexpected parameter count {n_params:,}. "
        "Check architecture config before wasting GPU time."
    )

    # ── 5. torch.compile ──────────────────────────────────────────────────────
    if config.compile_model:
        print("Compiling model (torch.compile)… first few steps will be slower.")
        model = torch.compile(model)

    # ── 6. Optimizer ──────────────────────────────────────────────────────────
    # configure_optimizers handles the decay/no-decay param split and fused AdamW.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(config)

    # ── 7. DataLoaders ────────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(
        shard_dir=config.data_dir,
        split="train",
        batch_size=config.micro_batch_size,
        seq_len=config.seq_len,
        seed=config.seed,
    )
    print(f"Train data: {train_loader}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = CheckpointManager.load(
            args.resume, model, optimizer,
            train_loader=train_loader,
            device=str(device),
        )
        start_step += 1  # resume from the next step

    ckpt_mgr = CheckpointManager(config)
    logger   = Logger(config)

    # ── 8. Training loop ──────────────────────────────────────────────────────
    model.train()
    t_start = time.time()

    for step in range(start_step, total_steps):
        t_step = time.time()
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)

            # Scale loss so gradients average (not sum) across accumulation steps
            loss = loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # accum_loss is now the mean loss over all micro batches
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % config.log_every_steps == 0:
            tok_per_sec = (
                config.micro_batch_size * config.seq_len * config.grad_accum_steps
                / (time.time() - t_step)
            )
            elapsed_h = (time.time() - t_start) / 3600
            print(
                f"step {step:>6d}/{total_steps} | "
                f"loss {accum_loss:.4f} | "
                f"grad_norm {grad_norm:.3f} | "
                f"lr {lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"elapsed {elapsed_h:.2f}h"
            )
            logger.log_train(step, accum_loss, float(grad_norm), lr)

        # ── Validation + checkpoint ───────────────────────────────────────────
        do_eval   = (step % config.eval_every_steps == 0 and step > 0)
        do_ckpt   = (step % config.checkpoint_every_steps == 0 and step > 0)
        do_ckpt  |= ckpt_mgr.should_checkpoint_time()  # time-based safety trigger

        val_loss = None
        if do_eval:
            val_loss, val_ppl = eval_val_loss(model, config.val_dir, config, device)
            print(f"  [val] step {step}: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
            logger.log_val(step, val_loss, val_ppl)

        if do_ckpt:
            ckpt_mgr.save(step, model, optimizer, train_loader, val_loss)

    # ── Final save ────────────────────────────────────────────────────────────
    final_ckpt = ckpt_mgr.save(total_steps - 1, model, optimizer, train_loader, val_loss=None)
    logger.close()

    total_h = (time.time() - t_start) / 3600
    print(f"\nTraining complete. Total time: {total_h:.2f}h")
    print(f"Final checkpoint: {final_ckpt}")

    # ── Smoke test assertion ───────────────────────────────────────────────────
    if args.smoke_test:
        import math
        RANDOM_BASELINE = math.log(config.vocab_size)  # ln(32000) ≈ 10.37
        if accum_loss < RANDOM_BASELINE:
            print(f"\nSMOKE TEST PASS: final loss {accum_loss:.4f} < {RANDOM_BASELINE:.4f}")
            sys.exit(0)
        else:
            print(
                f"\nSMOKE TEST FAIL: final loss {accum_loss:.4f} >= {RANDOM_BASELINE:.4f}. "
                "Check init, LR, or data loading."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
