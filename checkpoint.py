"""
checkpoint.py — Save / load training checkpoints.

Two checkpoint types are maintained in parallel:
  1. Rolling window: last `keep_last_n` step-based checkpoints
     (step_00500.pt, step_01000.pt, …) — older ones deleted automatically.
  2. Best-val: the single checkpoint with the lowest validation loss seen
     (best_val.pt) — kept indefinitely, overwritten when a new best is found.

Checkpoint contents (all in a single .pt dict):
    {
        "step":       int,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),  # includes Adam moments
        "dataloader": train_loader.state_dict(),  # shard_idx, token_idx, shard order, RNG
        "val_loss":   float | None,
        "config":     dataclasses.asdict(TrainConfig),
    }

Saving the optimizer state is mandatory for correct resume — without it, Adam
momentum and variance restart from zero, causing the LR schedule to feel like
a cold start even though we're mid-training.

Saving the dataloader state is equally mandatory — without it every resume
restarts from token 0 of the (re-shuffled) shard order, silently re-reading
early data and skipping whatever was ahead. The 9,537 × 524,288 = 5B unique
token guarantee depends on exact position restore.

Time-based safety:
    checkpoint.py doesn't track time itself.  train.py calls should_checkpoint_time()
    each step, which returns True if more than `checkpoint_every_minutes` have
    passed since the last save.  This provides a safety net against mid-interval
    session timeouts independent of the step-based cadence.
"""

import dataclasses
import glob
import os
import time
from typing import Optional

import torch

from config import TrainConfig


class CheckpointManager:
    def __init__(self, config: TrainConfig):
        self.config         = config
        self.out_dir        = config.checkpoint_dir
        self.keep_last_n    = config.keep_last_n
        self.keep_best_val  = config.keep_best_val
        self._best_val_loss = float("inf")
        self._last_save_t   = time.time()

        os.makedirs(self.out_dir, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,                        # ShardedDataLoader (avoid circular import)
        val_loss: Optional[float] = None,
    ) -> str:
        """
        Saves a checkpoint and manages the rolling window + best-val tracking.
        Returns the path of the saved step checkpoint.
        """
        # Unwrap torch.compile wrapper if present (it wraps the module)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        ckpt = {
            "step":       step,
            "model":      raw_model.state_dict(),
            "optimizer":  optimizer.state_dict() if self.config.save_optimizer_state else None,
            "dataloader": train_loader.state_dict(),
            "val_loss":   val_loss,
            "config":     dataclasses.asdict(self.config),
        }

        # Step checkpoint
        path = os.path.join(self.out_dir, f"step_{step:06d}.pt")
        torch.save(ckpt, path)
        self._last_save_t = time.time()
        print(f"  [ckpt] Saved {path}" + (f"  val_loss={val_loss:.4f}" if val_loss else ""))

        # Prune rolling window
        self._prune_old_checkpoints()

        # Best-val checkpoint
        if self.keep_best_val and val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            best_path = os.path.join(self.out_dir, "best_val.pt")
            torch.save(ckpt, best_path)
            print(f"  [ckpt] New best val → {best_path}  (loss={val_loss:.4f})")

        return path

    def _prune_old_checkpoints(self):
        """Delete oldest step checkpoints beyond the rolling window."""
        pattern  = os.path.join(self.out_dir, "step_*.pt")
        existing = sorted(glob.glob(pattern))  # sorted by step number (lexicographic = numeric)
        while len(existing) > self.keep_last_n:
            oldest = existing.pop(0)
            os.remove(oldest)
            print(f"  [ckpt] Pruned {oldest}")

    # ── Load ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_loader=None,                   # ShardedDataLoader | None
        device: str = "cuda",
    ) -> int:
        """
        Load checkpoint into model, optimizer, and dataloader.
        Returns the step number to resume from.

        Restores dataloader position (shard_idx, token_idx, shard order, RNG)
        so the next next_batch() call continues exactly where training left off.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if train_loader is not None and ckpt.get("dataloader") is not None:
            train_loader.load_state_dict(ckpt["dataloader"])

        step = ckpt["step"]
        print(
            f"  [ckpt] Resumed from {path}  "
            f"(step={step}, val_loss={ckpt.get('val_loss')}, "
            f"shard={ckpt.get('dataloader', {}).get('shard_idx', '?')}, "
            f"tok={ckpt.get('dataloader', {}).get('token_idx', '?')})"
        )
        return step

    # ── Time-based safety trigger ─────────────────────────────────────────────

    def should_checkpoint_time(self) -> bool:
        """
        Returns True if more than `checkpoint_every_minutes` have elapsed since
        the last save.  Called every step in the train loop with negligible cost.
        """
        elapsed_min = (time.time() - self._last_save_t) / 60.0
        return elapsed_min >= self.config.checkpoint_every_minutes
