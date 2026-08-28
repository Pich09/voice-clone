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
import collections
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
    parser.add_argument("--holdout_speakers", nargs="*", default=[],
                         help="Speaker id(s) to reserve entirely for test "
                              "(e.g. to check generalization to an unseen "
                              "voice), instead of being split like everyone "
                              "else.")
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    holdout = set(args.holdout_speakers)
    by_speaker = collections.defaultdict(list)
    for r in rows:
        by_speaker[r.get("speaker_id", "unknown")].append(r)

    # Split per speaker rather than globally shuffling all rows together --
    # a global shuffle can put a low-utterance-count speaker entirely into
    # valid/test with zero train exposure, which defeats multi-speaker base
    # training for that speaker. Splitting within each speaker's own rows
    # guarantees every non-holdout speaker with >=1 utterance keeps at least
    # one utterance in train.
    valid_rows, test_rows, train_rows = [], [], []
    for speaker, group in by_speaker.items():
        if speaker in holdout:
            test_rows.extend(group)
            continue

        group = list(group)
        random.shuffle(group)
        n_group = len(group)
        n_valid = int(n_group * args.valid_frac)
        n_test = int(n_group * args.test_frac)
        # Never take every utterance from a speaker into valid/test -- keep
        # at least one in train whenever the speaker has any at all.
        if n_valid + n_test >= n_group:
            n_valid = min(n_valid, max(n_group - 1, 0))
            n_test = min(n_test, max(n_group - 1 - n_valid, 0))

        valid_rows.extend(group[:n_valid])
        test_rows.extend(group[n_valid:n_valid + n_test])
        train_rows.extend(group[n_valid + n_test:])

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

    print(f"Total: {len(rows)} utterances, {total_hours:.2f} hours ({len(by_speaker)} speakers)")
    print(f"  Train: {len(train_rows)} ({train_hours:.2f}h)")
    print(f"  Valid: {len(valid_rows)}")
    print(f"  Test:  {len(test_rows)}")


if __name__ == "__main__":
    main()
