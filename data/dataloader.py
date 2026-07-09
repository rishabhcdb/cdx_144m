"""
data/dataloader.py — Sharded np.memmap DataLoader for .bin token files.

Each .bin shard is a flat uint16 array of token IDs written by the data-prep
scripts.  This loader streams (x, y) next-token pairs from them with zero
copies — np.memmap reads are backed directly by the OS file buffer.

Usage:
    loader = ShardedDataLoader(
        shard_dir="data/shards",
        split="train",          # reads train_*.bin
        batch_size=16,
        seq_len=2048,
        seed=1337,
    )
    x, y = loader.next_batch()  # tensors on CPU; move to device in train.py

Checkpoint persistence:
    state = loader.state_dict()          # call before saving checkpoint
    loader.load_state_dict(state)        # call after resuming checkpoint

    Saves: current shard index, token index within that shard, the full
    (post-shuffle) shard path order, and the RNG internal state.  All four
    are required — the RNG state diverges from the seed-derived initial value
    after the first epoch-boundary reshuffle, so re-seeding from scratch would
    give a different shard order than where we actually left off.

Shard naming convention:
    data/shards/train_00000.bin
    data/shards/train_00001.bin
    ...
    data/shards/val.bin          (single validation shard, read separately)
"""

import glob
import os
import random

import numpy as np
import torch


class ShardedDataLoader:
    def __init__(
        self,
        shard_dir: str,
        split: str,          # "train" → looks for train_*.bin
        batch_size: int,
        seq_len: int,
        seed: int = 1337,
    ):
        self.batch_size = batch_size
        self.seq_len    = seq_len

        pattern = os.path.join(shard_dir, f"{split}_*.bin")
        self.shard_paths = sorted(glob.glob(pattern))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shards found matching '{pattern}'. "
                "Run data/prepare_fineweb.py and data/prepare_stories.py first."
            )

        self._rng = random.Random(seed)
        self._rng.shuffle(self.shard_paths)  # reproducible shard order

        self._shard_idx  = 0
        self._token_idx  = 0
        self._tokens     = self._load_shard(self.shard_paths[0])

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_shard(path: str) -> np.ndarray:
        """Memory-map a uint16 .bin shard, cast to int64 for correct tensor dtype."""
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        # Must be int64: torch.from_numpy preserves numpy dtype exactly, and
        # F.cross_entropy requires torch.int64 (Long) targets — int32 raises a
        # dtype error.  uint16 storage is fine; int64 tensors are required.
        return mm.astype(np.int64)

    def _advance_shard(self):
        self._shard_idx = (self._shard_idx + 1) % len(self.shard_paths)
        self._token_idx = 0
        self._tokens    = self._load_shard(self.shard_paths[self._shard_idx])
        if self._shard_idx == 0:
            # Reshuffled shard order at each epoch boundary for variety
            self._rng.shuffle(self.shard_paths)

    # ── Public interface ──────────────────────────────────────────────────────

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (batch_size, seq_len) — input token IDs
            y: (batch_size, seq_len) — next-token target IDs
        """
        B, T = self.batch_size, self.seq_len
        needed = B * T + 1  # +1 so y = x shifted by 1

        # If the current shard doesn't have enough tokens left, advance
        while self._token_idx + needed > len(self._tokens):
            self._advance_shard()

        buf = self._tokens[self._token_idx : self._token_idx + needed]
        x   = torch.from_numpy(buf[:-1].reshape(B, T))
        y   = torch.from_numpy(buf[1: ].reshape(B, T))

        self._token_idx += B * T
        return x, y

    def state_dict(self) -> dict:
        """
        Capture full loader position for checkpoint persistence.

        Four fields are required for an exact resume:
          - shard_paths:  current shuffled order (may differ from init order
                          after epoch-boundary reshuffles)
          - shard_idx:    which shard we're currently reading
          - token_idx:    offset within that shard
          - rng_state:    internal RNG state — diverges from the seed-derived
                          initial value after the first epoch wrap, so simply
                          re-seeding from config.seed would give the wrong order
        """
        return {
            "shard_paths": list(self.shard_paths),
            "shard_idx":   self._shard_idx,
            "token_idx":   self._token_idx,
            "rng_state":   self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore loader position from a saved state_dict.
        Must be called after __init__ (so shard_paths is already populated)
        but before the first next_batch() call on the resumed run.
        """
        self.shard_paths  = state["shard_paths"]
        self._shard_idx   = state["shard_idx"]
        self._token_idx   = state["token_idx"]
        self._rng.setstate(state["rng_state"])
        # Re-load the shard we were mid-way through
        self._tokens = self._load_shard(self.shard_paths[self._shard_idx])

    def __repr__(self) -> str:
        total_tokens = sum(
            os.path.getsize(p) // 2  # uint16 = 2 bytes/token
            for p in self.shard_paths
        )
        return (
            f"ShardedDataLoader(n_shards={len(self.shard_paths)}, "
            f"~{total_tokens/1e9:.2f}B tokens, "
            f"batch={self.batch_size}, seq_len={self.seq_len})"
        )
