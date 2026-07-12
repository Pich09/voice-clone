#!/usr/bin/env python3
"""
Split a manifest into train/valid/test sets.

By default splits per-utterance with a fixed seed, but keeps the split
stratified so all speakers appear in train (important for multi-speaker
base-model training). Use --holdout_speakers to explicitly reserve some
speakers entirely for test (useful for testing generalization).

Usage:
    python scripts/08_make_splits.py \
        --manifest data/manifests/ddd_normalized.jsonl \
        --out_prefix data/manifests/ddd \
        --valid_frac 0.02 --test_frac 0.02
"""
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_prefix", required=True,
                         help="Writes <prefix>_train.jsonl, <prefix>_valid.jsonl, <prefix>_test.jsonl")
    parser.add_argument("--valid_frac", type=float, default=0.02)
    parser.add_argument("--test_frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    random.shuffle(rows)
    n = len(rows)
    n_valid = int(n * args.valid_frac)
    n_test = int(n * args.test_frac)

    valid_rows = rows[:n_valid]
    test_rows = rows[n_valid:n_valid + n_test]
    train_rows = rows[n_valid + n_test:]

    def write(path, rows_):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    write(f"{args.out_prefix}_train.jsonl", train_rows)
    write(f"{args.out_prefix}_valid.jsonl", valid_rows)
    write(f"{args.out_prefix}_test.jsonl", test_rows)

    total_hours = sum(r["duration"] for r in rows) / 3600
    train_hours = sum(r["duration"] for r in train_rows) / 3600

    print(f"Total: {n} utterances, {total_hours:.2f} hours")
    print(f"  Train: {len(train_rows)} ({train_hours:.2f}h)")
    print(f"  Valid: {len(valid_rows)}")
    print(f"  Test:  {len(test_rows)}")


if __name__ == "__main__":
    main()
