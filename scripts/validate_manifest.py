#!/usr/bin/env python3
"""
Validate a JSONL manifest file.

Checks:
- Every line is valid JSON
- Required fields are present: audio_path, text, speaker_id, duration
- audio_path exists on disk
- duration is a positive number
- text is non-empty after stripping

Usage:
    python scripts/validate_manifest.py data/manifests/ddd_clean.jsonl
"""
import argparse
import json
import os
import sys

REQUIRED_FIELDS = ["audio_path", "text", "speaker_id", "duration"]


def validate(manifest_path: str) -> int:
    n_ok, n_bad = 0, 0
    errors = []

    with open(manifest_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON ({e})")
                n_bad += 1
                continue

            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                errors.append(f"line {line_no}: missing fields {missing}")
                n_bad += 1
                continue

            if not row["text"] or not row["text"].strip():
                errors.append(f"line {line_no}: empty text")
                n_bad += 1
                continue

            if not isinstance(row["duration"], (int, float)) or row["duration"] <= 0:
                errors.append(f"line {line_no}: bad duration {row['duration']!r}")
                n_bad += 1
                continue

            if not os.path.exists(row["audio_path"]):
                errors.append(f"line {line_no}: audio_path not found: {row['audio_path']}")
                n_bad += 1
                continue

            n_ok += 1

    print(f"Manifest: {manifest_path}")
    print(f"  OK rows:    {n_ok}")
    print(f"  Bad rows:   {n_bad}")
    if errors:
        print("  First 20 errors:")
        for e in errors[:20]:
            print("   -", e)

    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Path to JSONL manifest")
    args = parser.parse_args()
    sys.exit(validate(args.manifest))
