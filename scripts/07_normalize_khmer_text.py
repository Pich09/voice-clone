#!/usr/bin/env python3
"""
Run the Khmer text normalizer over every transcript in a manifest and
write out a manifest with `text_raw` (original) and `text` (normalized)
fields. This must run before splitting into train/valid/test.

Usage:
    python scripts/07_normalize_khmer_text.py \
        --manifest data/manifests/ddd_clean.jsonl \
        --output data/manifests/ddd_normalized.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khmer_tts.text.normalize import normalize_khmer_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_ok, n_empty = 0, 0
    with open(args.manifest, encoding="utf-8") as in_f, \
         open(args.output, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_text = row["text"]
            normalized = normalize_khmer_text(raw_text)

            if not normalized.strip():
                n_empty += 1
                continue

            row["text_raw"] = raw_text
            row["text"] = normalized
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Normalized {n_ok} transcripts. Skipped {n_empty} empty results.")


if __name__ == "__main__":
    main()
