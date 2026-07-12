#!/usr/bin/env python3
"""
Loudness-normalize and resample audio to the final training format
(Section 6.5): 24kHz mono 16-bit PCM WAV, -23 to -18 LUFS, true peak < -1.5dB.

Uses ffmpeg's loudnorm filter directly (single-pass, good enough for a
training corpus; use two-pass loudnorm if you need broadcast-grade
precision).

Usage:
    python scripts/06_loudness_normalize.py \
        --manifest data/manifests/ddd_vad.jsonl \
        --output_dir data/processed/ddd_24k \
        --output_manifest data/manifests/ddd_clean.jsonl
"""
import argparse
import json
import os
import subprocess


def normalize_file(input_path: str, output_path: str, target_lufs: float = -20.0,
                    true_peak: float = -1.5, lra: float = 11.0) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}",
        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--target_lufs", type=float, default=-20.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(args.manifest, encoding="utf-8") as in_f, \
         open(args.output_manifest, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fname = os.path.splitext(os.path.basename(row["audio_path"]))[0] + ".wav"
            out_path = os.path.join(args.output_dir, fname)

            ok = normalize_file(row["audio_path"], out_path, args.target_lufs)
            if not ok:
                print(f"ffmpeg failed on {row['audio_path']}")
                n_fail += 1
                continue

            row["audio_path"] = out_path
            row["sample_rate"] = 24000
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1

            if n_ok % 500 == 0:
                print(f"  ... normalized {n_ok} files")

    print(f"Done. OK: {n_ok}  Failed: {n_fail}")
    print(f"This is your clean manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
