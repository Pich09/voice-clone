#!/usr/bin/env python3
"""
Convert a clean, normalized manifest into Fish Speech's expected dataset
layout (Section 8):

    data/fish/khmer_base/
      speaker_001/
        00000001.wav
        00000001.lab
        00000002.wav
        00000002.lab

Each .lab file contains ONLY the normalized transcript, matching
its same-numbered .wav.

Usage:
    python scripts/09_convert_to_fish_format.py \
        --manifest data/manifests/ddd_train.jsonl \
        --out_dir data/fish/khmer_base
"""
import argparse
import json
import os
import shutil
from collections import defaultdict


def sanitize_speaker_id(raw_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(raw_id))
    return safe or "unknown_speaker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--copy", action="store_true",
                         help="Copy audio files (default: symlink to save disk space)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    speaker_counters = defaultdict(int)
    n_written = 0

    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            speaker = sanitize_speaker_id(row["speaker_id"])
            speaker_dir = os.path.join(args.out_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)

            speaker_counters[speaker] += 1
            idx = speaker_counters[speaker]
            stem = f"{idx:08d}"

            wav_out = os.path.join(speaker_dir, f"{stem}.wav")
            lab_out = os.path.join(speaker_dir, f"{stem}.lab")

            src_wav = os.path.abspath(row["audio_path"])
            if args.copy:
                shutil.copyfile(src_wav, wav_out)
            else:
                if os.path.lexists(wav_out):
                    os.remove(wav_out)
                os.symlink(src_wav, wav_out)

            with open(lab_out, "w", encoding="utf-8") as lab_f:
                lab_f.write(row["text"].strip() + "\n")

            n_written += 1

    print(f"Wrote {n_written} wav/lab pairs across {len(speaker_counters)} speakers.")
    print(f"Speakers: {dict(speaker_counters)}")
    print(f"\nDataset ready at: {args.out_dir}")
    print("Next: run Fish Speech's own VQ token extraction + protobuf build "
          "(see vendor/fish-speech/tools/, or scripts/10_train_fish_khmer_base.sh).")


if __name__ == "__main__":
    main()
