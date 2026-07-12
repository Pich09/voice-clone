#!/usr/bin/env python3
"""
Audio quality control: score every clip in a manifest and assign a
quality grade (A/B/C/D), per Section 6.2 of the project plan.

Metrics computed per file:
    duration, rms_db, peak_db, clipping_ratio, silence_ratio, sample_rate

Grading heuristic (tune thresholds to your data):
    A: clean, 16kHz+, no clipping, low silence ratio, RMS in healthy range
    B: usable but noisier / slightly more silence
    C: weak -- borderline, avoid for TTS training
    D: reject -- too short, too much clipping, or too quiet/loud

Usage:
    python scripts/03_audio_qc.py \
        --manifest data/manifests/ddd_raw_merged.jsonl \
        --output data/manifests/ddd_qc.jsonl
"""
import argparse
import json

import numpy as np
import soundfile as sf


def compute_metrics(audio_path: str) -> dict:
    data, sr = sf.read(audio_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono for measurement

    duration = len(data) / sr
    peak = np.max(np.abs(data)) if len(data) else 0.0
    rms = np.sqrt(np.mean(data ** 2)) if len(data) else 0.0

    peak_db = 20 * np.log10(max(peak, 1e-9))
    rms_db = 20 * np.log10(max(rms, 1e-9))

    clipping_ratio = float(np.mean(np.abs(data) >= 0.999)) if len(data) else 1.0

    # crude silence detection: fraction of 20ms frames below -40dB RMS
    frame_len = max(int(0.02 * sr), 1)
    n_frames = len(data) // frame_len
    if n_frames == 0:
        silence_ratio = 1.0
    else:
        frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-9))
        silence_ratio = float(np.mean(frame_db < -40))

    return {
        "duration": round(duration, 3),
        "sample_rate": sr,
        "peak_db": round(float(peak_db), 2),
        "rms_db": round(float(rms_db), 2),
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
    }


def grade(metrics: dict, text: str) -> str:
    dur = metrics["duration"]
    sr = metrics["sample_rate"]
    clip = metrics["clipping_ratio"]
    sil = metrics["silence_ratio"]
    rms_db = metrics["rms_db"]

    if dur < 0.5 or dur > 20:
        return "D"
    if not text or len(text.strip()) < 2:
        return "D"
    if clip > 0.01:
        return "D"
    if sr < 16000:
        return "D"
    if sil > 0.6:
        return "D"

    if sr >= 22050 and clip == 0 and sil < 0.2 and -30 <= rms_db <= -12:
        return "A"

    if sil < 0.4 and rms_db > -35:
        return "B"

    return "C"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    with open(args.manifest, encoding="utf-8") as in_f, \
         open(args.output, "w", encoding="utf-8") as out_f:
        for line_no, line in enumerate(in_f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                metrics = compute_metrics(row["audio_path"])
            except Exception as e:
                print(f"line {line_no}: failed to read audio ({e}), grading D")
                metrics = {"duration": row.get("duration", 0), "sample_rate": 0,
                           "peak_db": -99, "rms_db": -99, "clipping_ratio": 1.0,
                           "silence_ratio": 1.0}

            row.update(metrics)
            row["quality_grade"] = grade(metrics, row.get("text", ""))
            grade_counts[row["quality_grade"]] += 1

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if line_no % 1000 == 0:
                print(f"  ... scored {line_no} files")

    total = sum(grade_counts.values())
    print(f"\nScored {total} files.")
    for g in "ABCD":
        pct = 100 * grade_counts[g] / total if total else 0
        print(f"  {g}: {grade_counts[g]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
