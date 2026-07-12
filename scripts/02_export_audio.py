#!/usr/bin/env python3
"""
Merge one or more raw manifests (e.g. from multiple DDD dataset repos)
into a single consolidated manifest, re-checking that audio files exist
and durations are computed directly from the audio file (not trusted
blindly from upstream metadata).

Usage:
    python scripts/02_export_audio.py \
        --inputs data/manifests/ddd_raw.jsonl \
        --output data/manifests/ddd_raw_merged.jsonl
"""
import argparse
import json
import os

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_in, n_out = 0, 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for path in args.inputs:
            with open(path, encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    row = json.loads(line)

                    if not os.path.exists(row["audio_path"]):
                        continue

                    try:
                        info = sf.info(row["audio_path"])
                        row["duration"] = round(info.frames / info.samplerate, 3)
                        row["sample_rate"] = info.samplerate
                        row["channels"] = info.channels
                    except Exception as e:
                        print(f"skip (unreadable audio): {row['audio_path']} ({e})")
                        continue

                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_out += 1

    print(f"Read {n_in} rows across {len(args.inputs)} manifest(s).")
    print(f"Wrote {n_out} valid rows to {args.output}")


if __name__ == "__main__":
    main()
