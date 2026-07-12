#!/usr/bin/env python3
"""
Download DDD Khmer dataset(s) from Hugging Face and export raw audio +
metadata into data/raw/ddd/.

Usage:
    python scripts/01_download_ddd.py \
        --dataset DDD-Cambodia/khm-asr-cultural \
        --split train \
        --out_dir data/raw/ddd

Requires: `datasets`, `soundfile`, and a HuggingFace account/token if the
dataset is gated (set HF_TOKEN env var, or run `huggingface-cli login`).
"""
import argparse
import os

import soundfile as sf
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="data/raw/ddd")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Optional cap for quick smoke tests")
    args = parser.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join("data", "manifests", "ddd_raw.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    print(f"Loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=False)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    n_written = 0
    with open(manifest_path, "a", encoding="utf-8") as manifest_f:
        for i, row in enumerate(ds):
            # Field names vary across DDD releases -- adjust these keys to
            # match the actual dataset schema (check ds.features first).
            audio = row.get("audio")
            text = row.get("transcript") or row.get("text") or ""
            speaker_id = row.get("speaker_id") or row.get("speaker") or "unknown"

            if audio is None or not text.strip():
                continue

            array = audio["array"]
            sr = audio["sampling_rate"]
            fname = f"{speaker_id}_{i:07d}.wav"
            out_path = os.path.join(audio_dir, fname)
            sf.write(out_path, array, sr)

            duration = len(array) / sr
            record = {
                "audio_path": out_path,
                "text": text.strip(),
                "speaker_id": speaker_id,
                "duration": round(duration, 3),
                "source": args.dataset,
            }
            manifest_f.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 500 == 0:
                print(f"  ... {n_written} samples exported")

    print(f"Done. Wrote {n_written} records to {manifest_path}")


if __name__ == "__main__":
    main()
