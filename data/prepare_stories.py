"""
data/prepare_stories.py — Download and tokenise the stories corpus → uint16 .bin shards.

Designed to run on the GPU pod, NOT locally.

Dataset: roneneldan/TinyStories (V2, GPT-4 generated)
  - Downloaded directly via wget using the HF resolve URL.
  - Stories are separated by '<|endoftext|>' in the raw .txt file.
  - We tokenise up to TARGET_TOKENS at a clean story boundary and write
    SHARD_SIZE-token shards.

Shard layout (written to OUT_DIR):
    stories_00000.bin, stories_00001.bin, ...   (TARGET_TOKENS train shards)
    val_stories.bin                              (whole-story val split, ~VAL_TOKENS)

The stories shards are mixed with FineWeb shards by the dataloader at training
time (it reads all train_*.bin + stories_*.bin together).

Prerequisites on GPU pod:
    pip install transformers sentencepiece python-dotenv
    # Set HF_TOKEN in .env (needed for tokenizer, not for this dataset which is public)
    python data/prepare_stories.py

Fixed seed (1337) for story-level shuffle — same global seed.
"""

import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

# Public HF resolve URLs for TinyStoriesV2 (no auth required)
STORIES_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
]
DOWNLOAD_DIR     = "data/raw"
OUT_DIR          = "data/shards"
TOKENIZER_ID     = "meta-llama/Llama-2-7b-hf"

TARGET_TOKENS    = 300_000_000   # ~300M tokens from stories corpus
VAL_TOKENS       =     500_000   # ~0.5M held-out (proportional: 300M / 4700M * 8M ≈ 0.51M)
SHARD_SIZE       = 100_000_000   # ~100M tokens per shard
STORY_SEP        = "<|endoftext|>"
SEED             = 1337
EOS_ID           = 2             # Llama-2 <eos> token id

# ─────────────────────────────────────────────────────────────────────────────

def wget_download(url: str, dest_dir: str) -> str:
    """Download url into dest_dir with wget, return local file path."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    dest     = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  Already exists, skipping download: {dest}")
        return dest
    print(f"  wget {url}")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", dest, url],
        check=True,
    )
    return dest


def tokenize_story(text: str, tokenizer) -> list[int]:
    """Tokenise a single story, append EOS."""
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    if ids:
        ids.append(EOS_ID)
    return ids


def write_shard(tokens: list[int], path: str) -> int:
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories token shards.")
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Override token targets to tiny values for quick pipeline testing.",
    )
    args = parser.parse_args()

    # Runtime-only overrides — module-level constants are never mutated
    target_tokens    = 5_000_000 if args.smoke_test else TARGET_TOKENS
    val_tokens_target = 200_000  if args.smoke_test else VAL_TOKENS
    if args.smoke_test:
        print(f"[smoke_test] Token targets overridden: train={target_tokens:,}  val={val_tokens_target:,}")

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not set. Add it to .env or export it. "
                 "(Needed for the tokenizer, not for the stories dataset itself.)")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=hf_token)

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"\nDownloading TinyStories files…")
    local_files = [wget_download(url, DOWNLOAD_DIR) for url in STORIES_URLS]

    # ── Collect all stories ───────────────────────────────────────────────────
    print(f"\nParsing stories…")
    all_stories: list[str] = []
    for path in local_files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        stories = [s.strip() for s in raw.split(STORY_SEP) if s.strip()]
        all_stories.extend(stories)
        print(f"  {path}: {len(stories):,} stories")

    print(f"  Total stories: {len(all_stories):,}")

    # ── Document-level shuffle ────────────────────────────────────────────────
    rng = random.Random(SEED)
    rng.shuffle(all_stories)
    print(f"  Shuffled (seed={SEED})")

    # ── Phase 1: carve out val_stories.bin ──────────────────────────────────
    # Take whole stories from the front of the shuffled list until VAL_TOKENS
    # is reached. Write the full accumulated list (no truncation) so the last
    # story is never cut off. Remove these stories from the training pool.

    print(f"\nCarving val split (~{val_tokens_target/1e3:.0f}K tokens)…")
    val_buf: list[int] = []
    val_story_count = 0
    while all_stories and len(val_buf) < val_tokens_target:
        story = all_stories.pop(0)
        val_buf.extend(tokenize_story(story, tokenizer))
        val_story_count += 1

    val_path = os.path.join(OUT_DIR, "val_stories.bin")
    n_val = write_shard(val_buf, val_path)   # full list — no truncation
    print(f"  Wrote val_stories.bin: {n_val:,} tokens ({val_story_count} stories)")
    del val_buf

    # ── Phase 2: training shards ──────────────────────────────────────────────
    print(f"\nTokenising training data (target: {target_tokens/1e6:.0f}M tokens)…")

    shard_idx    = 0
    total_tokens = 0
    shard_buf: list[int] = []

    pbar = tqdm(all_stories, unit=" story")
    for story in pbar:
        toks = tokenize_story(story, tokenizer)
        shard_buf.extend(toks)

        if len(shard_buf) >= SHARD_SIZE:
            path = os.path.join(OUT_DIR, f"stories_{shard_idx:05d}.bin")
            n    = write_shard(shard_buf[:SHARD_SIZE], path)
            total_tokens += n
            print(f"\n  Shard {shard_idx:05d}: {n:,} tokens → {path}")
            shard_idx += 1
            shard_buf  = shard_buf[SHARD_SIZE:]

        if total_tokens >= target_tokens:
            break

    # Flush remainder into a final partial shard
    if shard_buf and total_tokens < target_tokens:
        path = os.path.join(OUT_DIR, f"stories_{shard_idx:05d}.bin")
        n    = write_shard(shard_buf, path)
        total_tokens += n
        print(f"  Shard {shard_idx:05d} (partial): {n:,} tokens → {path}")

    pbar.close()
    print(
        f"\nDone. {shard_idx + 1} train shards, "
        f"{total_tokens:,} tokens ({total_tokens/1e6:.1f}M). "
        f"val_stories.bin: {n_val:,} tokens."
    )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
