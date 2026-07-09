"""
data/push_to_hub.py — Upload data/shards/ to a private HF dataset repo.

Run this after prepare_fineweb.py and prepare_stories.py have finished writing
all shards.  The entire data/shards/ folder is uploaded under a "shards/"
prefix in the repo, preserving the flat file layout that ShardedDataLoader
expects on download.

Repo layout after upload:
    <HF_SHARD_REPO>/shards/train_00000.bin
    <HF_SHARD_REPO>/shards/train_00001.bin
    ...
    <HF_SHARD_REPO>/shards/stories_00000.bin
    <HF_SHARD_REPO>/shards/val_fineweb.bin
    <HF_SHARD_REPO>/shards/val_stories.bin

Prerequisites on GPU pod:
    pip install huggingface_hub python-dotenv
    HF_TOKEN and HF_SHARD_REPO set in .env (or exported)

Usage:
    python data/push_to_hub.py
    python data/push_to_hub.py --shard_dir data/shards   # override shard dir
    python data/push_to_hub.py --repo myuser/cdx144m-data  # override repo id
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Push data shards to HF Hub.")
    parser.add_argument("--shard_dir", default="data/shards",
                        help="Local shard directory to upload (default: data/shards)")
    parser.add_argument("--repo", default=None,
                        help="HF dataset repo ID (overrides HF_SHARD_REPO env var)")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN not set. Add it to .env or export it.")

    repo_id = args.repo or os.environ.get("HF_SHARD_REPO")
    if not repo_id:
        sys.exit(
            "HF_SHARD_REPO not set. Add it to .env or pass --repo <owner/name>.\n"
            "Example: HF_SHARD_REPO=myuser/cdx144m-data"
        )

    shard_dir = args.shard_dir
    if not os.path.isdir(shard_dir):
        sys.exit(f"Shard directory not found: {shard_dir!r}. Run data prep scripts first.")

    # Count files to give a useful summary before the potentially long upload
    files = [f for f in os.listdir(shard_dir) if f.endswith(".bin")]
    total_bytes = sum(os.path.getsize(os.path.join(shard_dir, f)) for f in files)
    print(f"Uploading {len(files)} .bin files ({total_bytes/1e9:.2f} GB) from '{shard_dir}' …")
    print(f"  → HF repo: {repo_id}  (subfolder: shards/)")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    # Create the repo if it doesn't exist yet (private by default)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    api.upload_folder(
        folder_path=shard_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="shards",   # all files land under shards/ in the repo
        ignore_patterns=["*.py", "__pycache__"],
    )

    print(f"\nDone. Shards are now at: https://huggingface.co/datasets/{repo_id}/tree/main/shards")


if __name__ == "__main__":
    main()
