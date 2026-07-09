"""
data/prepare_fineweb.py — Tokenise FineWeb → uint16 .bin shards.

Designed to run on the GPU pod, NOT locally.

Strategy:
  - Stream documents via datatrove.ParquetReader from HuggingFace.
  - Stop at a clean document boundary once TARGET_TRAIN_TOKENS are written.
  - Carve out the first ~VAL_TOKENS documents BEFORE any shuffling/packing →
    written to val.bin.  val.bin is never seen during training.
  - Within each shard: accumulate documents, shuffle their order (document-level,
    not token-level), then pack and write.  This gives true document-level
    shuffling within shards; shard order is shuffled by the dataloader.
  - EOS token (id=2 for Llama-2) appended between packed documents so the
    model learns to treat them as hard boundaries.

Shard layout (written to OUT_DIR):
    train_00000.bin, train_00001.bin, ...   (TARGET_TRAIN_TOKENS total)
    val_fineweb.bin                         (whole-document val split, ~VAL_TOKENS)

Prerequisites on GPU pod:
    pip install datatrove[io] transformers sentencepiece python-dotenv
    # Set HF_TOKEN in .env or export it
    python data/prepare_fineweb.py

Fixed seed (1337) for all shuffling — matches the global reproducibility seed.
"""

import os
import random
import sys
import time

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_PATH     = "hf://datasets/HuggingFaceFW/fineweb/sample/10BT"
OUT_DIR          = "data/shards"
TOKENIZER_ID     = "meta-llama/Llama-2-7b-hf"

TARGET_TRAIN_TOKENS = 4_700_000_000   # ~4.7B
VAL_TOKENS          =     8_000_000   # ~8M held-out

SHARD_SIZE       = 100_000_000        # ~100M tokens per shard (uint16 → ~200MB each)
SEED             = 1337
EOS_ID           = 2                  # Llama-2 <eos> token id

# ─────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not set. Add it to .env or export it.")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=hf_token)

    # ── Datatrove reader ──────────────────────────────────────────────────────
    from datatrove.pipeline.readers import ParquetReader
    data_reader = ParquetReader(DATASET_PATH, limit=None)

    rng = random.Random(SEED)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def tokenize_doc(text: str) -> list[int]:
        """Tokenise a document, strip special tokens, append EOS."""
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(EOS_ID)
        return ids

    def write_shard(tokens: list[int], path: str):
        """Write a flat list of token IDs as uint16 to a .bin file."""
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(path)
        return len(arr)

    # ── Phase 1: collect val_fineweb.bin ─────────────────────────────────────
    # Read documents until accumulated token count reaches VAL_TOKENS, then
    # flush the FULL accumulated list (no truncation) so the last document is
    # never cut off mid-sentence.
    # These documents are excluded from training shards entirely.

    print(f"\nCollecting validation split (~{VAL_TOKENS/1e6:.0f}M tokens)…")
    val_tokens: list[int] = []
    val_doc_count = 0

    doc_iter = data_reader()
    for doc in doc_iter:
        toks = tokenize_doc(doc.text)
        val_tokens.extend(toks)
        val_doc_count += 1
        if len(val_tokens) >= VAL_TOKENS:
            break

    val_path = os.path.join(OUT_DIR, "val_fineweb.bin")
    n_val = write_shard(val_tokens, val_path)   # full list — no [:VAL_TOKENS] truncation
    print(f"  Wrote val_fineweb.bin: {n_val:,} tokens ({val_doc_count} docs)")
    del val_tokens

    # ── Phase 2: training shards ───────────────────────────────────────────────
    print(f"\nTokenising training data (target: {TARGET_TRAIN_TOKENS/1e9:.1f}B tokens)…")

    shard_idx          = 0
    total_train_tokens = 0
    shard_doc_buf: list[list[int]] = []   # accumulate whole documents
    shard_tok_count    = 0

    def flush_shard(docs: list[list[int]], idx: int) -> int:
        """Shuffle docs within shard, pack, write. Returns token count."""
        rng.shuffle(docs)
        flat = [tok for doc in docs for tok in doc]
        path = os.path.join(OUT_DIR, f"train_{idx:05d}.bin")
        n    = write_shard(flat, path)
        print(f"  Shard {idx:05d}: {n:,} tokens → {path}")
        return n

    pbar = tqdm(unit=" tok", unit_scale=True, total=TARGET_TRAIN_TOKENS)

    # doc_iter is already advanced past the val documents — continue from there
    for doc in doc_iter:
        toks = tokenize_doc(doc.text)
        shard_doc_buf.append(toks)
        shard_tok_count += len(toks)

        if shard_tok_count >= SHARD_SIZE:
            n = flush_shard(shard_doc_buf, shard_idx)
            total_train_tokens += n
            pbar.update(n)
            shard_idx       += 1
            shard_doc_buf    = []
            shard_tok_count  = 0

        if total_train_tokens >= TARGET_TRAIN_TOKENS:
            break   # stop at clean document boundary

    # Flush any remaining docs into a final partial shard
    if shard_doc_buf and total_train_tokens < TARGET_TRAIN_TOKENS:
        n = flush_shard(shard_doc_buf, shard_idx)
        total_train_tokens += n
        pbar.update(n)

    pbar.close()
    print(
        f"\nDone. {shard_idx + 1} training shards, "
        f"{total_train_tokens:,} tokens ({total_train_tokens/1e9:.3f}B). "
        f"val.bin: {n_val:,} tokens."
    )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
