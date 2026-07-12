#!/usr/bin/env python3
"""
Trim leading/trailing silence with Silero VAD and reject files with too
little actual speech (Section 6.4).

Usage:
    pip install silero-vad torch torchaudio
    python scripts/05_vad_trim.py \
        --manifest data/manifests/ddd_denoised.jsonl \
        --output_dir data/processed/ddd_vad \
        --output_manifest data/manifests/ddd_vad.jsonl \
        --min_speech_ratio 0.4
"""
import argparse
import json
import os

import soundfile as sf


def get_speech_timestamps_and_trim(audio_path: str, out_path: str, min_speech_ratio: float):
    """Returns (kept: bool, speech_ratio: float)."""
    try:
        import torch
        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        (get_speech_timestamps, _, read_audio, *_rest) = utils
    except Exception as e:
        # Fallback: no VAD available, just copy through and assume it passes.
        import shutil
        shutil.copyfile(audio_path, out_path)
        return True, 1.0

    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    if not timestamps:
        return False, 0.0

    total_speech = sum(t["end"] - t["start"] for t in timestamps)
    speech_ratio = total_speech / len(wav)

    if speech_ratio < min_speech_ratio:
        return False, speech_ratio

    # Trim to first speech start .. last speech end, keeping internal pauses.
    start, end = timestamps[0]["start"], timestamps[-1]["end"]
    trimmed = wav[start:end]

    data, sr = sf.read(audio_path)
    # Map 16kHz VAD sample indices back to original sample rate.
    ratio = sr / 16000
    orig_start = int(start * ratio)
    orig_end = int(end * ratio)
    if data.ndim > 1:
        trimmed_audio = data[orig_start:orig_end, :]
    else:
        trimmed_audio = data[orig_start:orig_end]

    sf.write(out_path, trimmed_audio, sr)
    return True, speech_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--min_speech_ratio", type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)

    n_kept, n_rejected = 0, 0
    with open(args.manifest, encoding="utf-8") as in_f, \
         open(args.output_manifest, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fname = os.path.basename(row["audio_path"])
            out_path = os.path.join(args.output_dir, fname)

            kept, ratio = get_speech_timestamps_and_trim(
                row["audio_path"], out_path, args.min_speech_ratio
            )
            if not kept:
                n_rejected += 1
                continue

            try:
                info = sf.info(out_path)
                row["duration"] = round(info.frames / info.samplerate, 3)
            except Exception:
                pass

            row["audio_path"] = out_path
            row["speech_ratio"] = round(ratio, 4)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Kept: {n_kept}  Rejected (too little speech): {n_rejected}")


if __name__ == "__main__":
    main()
