"""
data/pull_from_hub.py — Download shards from HF Hub into data/shards/.

Run this ONCE at the start of any new pod session, before train.py.
Downloads only the shards/ subfolder from the HF dataset repo, so you
don't pull any unrelated repo contents.

After this script completes, ShardedDataLoader can find all shards at
data/shards/train_*.bin, data/shards/stories_*.bin, data/shards/val_*.bin
— exactly the paths it expects.

Prerequisites on GPU pod:
    pip install huggingface_hub python-dotenv
    HF_TOKEN and HF_SHARD_REPO set in .env (or exported)

Usage:
    python data/pull_from_hub.py
    python data/pull_from_hub.py --repo myuser/cdx144m-data
    python data/pull_from_hub.py --dest data/shards   # override local dir
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Pull data shards from HF Hub.")
    parser.add_argument("--repo", default=None,
                        help="HF dataset repo ID (overrides HF_SHARD_REPO env var)")
    parser.add_argument("--dest", default="data/shards",
                        help="Local destination directory (default: data/shards)")
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

    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading shards from {repo_id}/shards → {dest} …")

    from huggingface_hub import snapshot_download

    # snapshot_download with allow_patterns restricts to the shards/ subfolder.
    # Files land at: <local_dir>/shards/*.bin
    # We then move/link them to dest if needed, or point dest to local_dir/shards.
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
        allow_patterns=["shards/*"],
        local_dir=os.path.dirname(dest),    # download into parent of dest
        local_dir_use_symlinks=False,        # actual copies, not symlinks
    )

    # snapshot_download places files at local_root/shards/*.bin
    # which is exactly <parent_of_dest>/shards/ = dest
    downloaded = os.path.join(local_root, "shards")
    if os.path.abspath(downloaded) != os.path.abspath(dest):
        # Edge case: user passed a non-standard --dest; show where files actually are
        print(f"  Note: files downloaded to {downloaded!r}  (not {dest!r})")
        print(f"  Either move them manually or set --dest accordingly.")
    else:
        files = [f for f in os.listdir(dest) if f.endswith(".bin")]
        total_bytes = sum(os.path.getsize(os.path.join(dest, f)) for f in files)
        print(f"\nDone. {len(files)} shards ({total_bytes/1e9:.2f} GB) in {dest!r}")
        print(f"  train_*.bin: {sum(1 for f in files if f.startswith('train_'))}")
        print(f"  stories_*.bin: {sum(1 for f in files if f.startswith('stories_'))}")
        print(f"  val_*.bin: {sum(1 for f in files if f.startswith('val_'))}")


if __name__ == "__main__":
    main()
