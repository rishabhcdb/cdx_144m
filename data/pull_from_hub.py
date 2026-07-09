"""
data/pull_from_hub.py — Download shards from a private HF dataset repo into data/shards/.

Run once at the start of any new pod session, before train.py.

Usage:
    python data/pull_from_hub.py
    python data/pull_from_hub.py --repo rishabhcdb/cdx_144m
    python data/pull_from_hub.py --dest data/shards
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Pull data shards from HF dataset repo.")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--dest", default="data/shards")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not set.")

    repo_id = args.repo or os.environ.get("HF_SHARD_REPO")
    if not repo_id:
        sys.exit("HF_SHARD_REPO not set. Add it to .env or pass --repo <owner/name>.")

    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading {repo_id}/shards/ → {dest} …")

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
        allow_patterns=["shards/*"],
        local_dir=os.path.dirname(dest),
        local_dir_use_symlinks=False,
    )

    files = [f for f in os.listdir(dest) if f.endswith(".bin")]
    total_bytes = sum(os.path.getsize(os.path.join(dest, f)) for f in files)
    print(f"Done. {len(files)} shards ({total_bytes/1e9:.2f} GB) in {dest!r}")
    print(f"  train: {sum(1 for f in files if f.startswith('train_'))}  "
          f"stories: {sum(1 for f in files if f.startswith('stories_'))}  "
          f"val: {sum(1 for f in files if f.startswith('val_'))}")


if __name__ == "__main__":
    main()
