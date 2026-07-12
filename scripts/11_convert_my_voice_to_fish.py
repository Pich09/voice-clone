#!/usr/bin/env python3
"""
Preprocess and convert your own voice recordings into Fish Speech format
for voice adaptation (Section 9.2).

Expects a manifest of your recordings (audio_path, text) -- you can hand
-write this JSONL, or record with a script that logs prompts + audio
paths as you go. This script:
  1. Runs the same loudness normalization as the DDD pipeline
  2. Runs Khmer text normalization on your transcripts
  3. Writes wav/lab pairs into data/fish/my_voice/<speaker>/

Usage:
    python scripts/11_convert_my_voice_to_fish.py \
        --manifest data/manifests/my_voice_raw.jsonl \
        --speaker_id my_voice \
        --out_dir data/fish/my_voice
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khmer_tts.text.normalize import normalize_khmer_text


def normalize_audio(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="JSONL with audio_path and text for your recordings")
    parser.add_argument("--speaker_id", default="my_voice")
    parser.add_argument("--out_dir", default="data/fish/my_voice")
    parser.add_argument("--processed_dir", default="data/processed/my_voice_24k")
    args = parser.parse_args()

    speaker_dir = os.path.join(args.out_dir, args.speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    manifest_out = []
    idx = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx += 1

            processed_path = os.path.join(args.processed_dir, f"{idx:08d}.wav")
            ok = normalize_audio(row["audio_path"], processed_path)
            if not ok:
                print(f"skip (ffmpeg failed): {row['audio_path']}")
                continue

            normalized_text = normalize_khmer_text(row["text"])
            if not normalized_text.strip():
                print(f"skip (empty after normalization): {row['audio_path']}")
                continue

            stem = f"{idx:08d}"
            wav_out = os.path.join(speaker_dir, f"{stem}.wav")
            lab_out = os.path.join(speaker_dir, f"{stem}.lab")

            os.replace(processed_path, wav_out)
            with open(lab_out, "w", encoding="utf-8") as lab_f:
                lab_f.write(normalized_text + "\n")

            manifest_out.append({
                "audio_path": wav_out, "text": normalized_text,
                "text_raw": row["text"], "speaker_id": args.speaker_id,
            })

    out_manifest_path = f"data/manifests/{args.speaker_id}_train.jsonl"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for r in manifest_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Converted {len(manifest_out)} recordings for speaker '{args.speaker_id}'")
    print(f"Fish dataset: {speaker_dir}")
    print(f"Manifest:     {out_manifest_path}")

    total_minutes = 0
    # quick duration count via ffprobe would go here in a fuller version
    print("\nReminder (Section 9.2): aim for at least 30 minutes, ideally "
          "1-3+ hours, recorded in the same quiet room with the same mic.")


if __name__ == "__main__":
    main()
