#!/usr/bin/env python3
"""
One-off, local, CPU-only tool: split the already-processed Khmer dataset
(data/fish/khmer_base + data/fish/khmer_base_val) into N Kaggle-sized shards
and upload them to the HF dataset repo, alongside a manifest and a
round-robin cursor.

Why this exists: Kaggle's /kaggle/working is hard-capped at 20GB, but the
full processed dataset is ~22GB -- it doesn't fit as a single download even
before the base checkpoint, VQ tokens, or training output. Splitting the
TRAIN audio into shards lets kaggle/train_khmer_base.ipynb pull one shard at
a time instead. The validation set stays a single unsharded archive since
it's small (~400MB) and cheap to re-pull every session.

Run this from the machine that already has data/fish/khmer_base populated
(from running scripts/01-09) -- it does NOT rebuild the processed data, it
just repacks what's already on disk. Needs HF_TOKEN in the environment.

Usage:
    python scripts/reshard_processed_data.py --repo Panhapich/khmer-tts-processed \\
        --key khmer_base_v1 --n-shards 6
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF dataset repo, e.g. Panhapich/khmer-tts-processed")
    parser.add_argument("--key", required=True, help="cache key, e.g. khmer_base_v1 (must match the notebook's DATA_CACHE_KEY)")
    parser.add_argument("--n-shards", type=int, default=6)
    parser.add_argument("--train-dir", default="data/fish/khmer_base")
    parser.add_argument("--val-dir", default="data/fish/khmer_base_val")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in the environment.", file=sys.stderr)
        sys.exit(1)

    full_train = os.path.join(ROOT, args.train_dir)
    if not os.path.isdir(full_train):
        print(f"ERROR: {full_train} does not exist -- run scripts/01-09 first.", file=sys.stderr)
        sys.exit(1)

    from khmer_tts.collab.data_cache import pack_and_upload_sharded

    print(f"Sharding {args.train_dir} into {args.n_shards} shard(s), "
          f"uploading to {args.repo} under key {args.key!r} ...")
    pack_and_upload_sharded(
        args.repo, args.key, ROOT,
        train_dir=args.train_dir, val_dir=args.val_dir,
        n_shards=args.n_shards, token=token, private=args.private,
    )
    print("Done. The training notebook will pick this up automatically on "
          "Kaggle (round-robin cursor reset to shard 0).")


if __name__ == "__main__":
    main()
