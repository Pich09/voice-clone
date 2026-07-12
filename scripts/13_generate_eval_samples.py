#!/usr/bin/env python3
"""
Generate audio for every sentence in eval/test_sentences_km.txt using a
given model/speaker, for human scoring (Section 12).

Usage:
    python scripts/13_generate_eval_samples.py \
        --model_dir models/khmer_base \
        --speaker default \
        --sentences eval/test_sentences_km.txt \
        --out_dir outputs/eval/khmer_base_v1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khmer_tts.inference.fish_backend import FishSpeechBackend
from khmer_tts.text.normalize import normalize_khmer_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--speaker", default="default")
    parser.add_argument("--sentences", default="eval/test_sentences_km.txt")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.sentences, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    backend = FishSpeechBackend(model_dir=args.model_dir)

    scorecard_path = os.path.join(args.out_dir, "SCORECARD.md")
    with open(scorecard_path, "w", encoding="utf-8") as sc:
        sc.write("# Evaluation Scorecard\n\n")
        sc.write(f"Model: `{args.model_dir}`  \nSpeaker: `{args.speaker}`\n\n")
        sc.write("| # | Sentence | Audio | Pronunciation (1-5) | Naturalness (1-5) "
                  "| Voice Similarity (1-5) | Artifacts (1-5) | Long-text stable? |\n")
        sc.write("|---|---|---|---|---|---|---|---|\n")

        for i, sentence in enumerate(sentences, 1):
            normalized = normalize_khmer_text(sentence)
            out_path = os.path.join(args.out_dir, f"sample_{i:03d}.wav")
            backend.synthesize(text=normalized, output_path=out_path, speaker=args.speaker)
            sc.write(f"| {i} | {sentence} | `{os.path.basename(out_path)}` | | | | | |\n")
            print(f"[{i}/{len(sentences)}] {sentence} -> {out_path}")

    print(f"\nGenerated {len(sentences)} eval samples in {args.out_dir}")
    print(f"Fill in scores in {scorecard_path}")
    print("\nRelease gates (Section 12.2):")
    print("  Pronunciation >= 4.0, Naturalness >= 3.8, Voice similarity >= 4.0,")
    print("  no severe artifacts, no repeated/skipped words.")


if __name__ == "__main__":
    main()
