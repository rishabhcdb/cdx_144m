"""
data/push_to_hub.py — Upload data/shards/ to a private HF dataset repo.

Usage:
    python data/push_to_hub.py
    python data/push_to_hub.py --shard_dir data/shards
    python data/push_to_hub.py --repo rishabhcdb/cdx_144m
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Push data shards to HF dataset repo.")
    parser.add_argument("--shard_dir", default="data/shards")
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not set.")

    repo_id = args.repo or os.environ.get("HF_SHARD_REPO")
    if not repo_id:
        sys.exit("HF_SHARD_REPO not set. Add it to .env or pass --repo <owner/name>.")

    shard_dir = args.shard_dir
    if not os.path.isdir(shard_dir):
        sys.exit(f"Shard directory not found: {shard_dir!r}")

    files = [f for f in os.listdir(shard_dir) if f.endswith(".bin")]
    total_bytes = sum(os.path.getsize(os.path.join(shard_dir, f)) for f in files)
    print(f"Uploading {len(files)} files ({total_bytes/1e9:.2f} GB) → {repo_id}/shards/")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=shard_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="shards",
        ignore_patterns=["*.py", "__pycache__"],
    )

    print("Done.")


if __name__ == "__main__":
    main()
