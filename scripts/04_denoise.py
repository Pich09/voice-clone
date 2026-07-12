#!/usr/bin/env python3
"""
Selective denoising, per Section 6.3:
    A-grade -> keep raw (no denoise)
    B-grade -> light denoise with DeepFilterNet
    C/D     -> excluded upstream, but if present here, skip

Usage:
    pip install deepfilternet
    python scripts/04_denoise.py \
        --manifest data/manifests/ddd_qc.jsonl \
        --output_dir data/processed/ddd_denoised \
        --output_manifest data/manifests/ddd_denoised.jsonl
"""
import argparse
import json
import os
import shutil


def denoise_file(input_path: str, output_path: str):
    """Light denoise using DeepFilterNet. Falls back to a plain copy if
    the library isn't installed, so the pipeline still runs end-to-end
    for smoke testing without the heavy dependency."""
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        shutil.copyfile(input_path, output_path)
        return False

    if not hasattr(denoise_file, "_model"):
        denoise_file._model, denoise_file._df_state, _ = init_df()

    audio, _ = load_audio(input_path, sr=denoise_file._df_state.sr())
    enhanced = enhance(denoise_file._model, denoise_file._df_state, audio)
    save_audio(output_path, enhanced, denoise_file._df_state.sr())
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--min_grade_to_keep", default="B", choices=["A", "B", "C"],
                         help="Reject anything worse than this grade")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    threshold = grade_order[args.min_grade_to_keep]

    n_kept, n_denoised, n_rejected = 0, 0, 0

    with open(args.manifest, encoding="utf-8") as in_f, \
         open(args.output_manifest, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            g = row.get("quality_grade", "D")

            if grade_order.get(g, 3) > threshold:
                n_rejected += 1
                continue

            fname = os.path.basename(row["audio_path"])
            out_path = os.path.join(args.output_dir, fname)

            if g == "A":
                shutil.copyfile(row["audio_path"], out_path)
            else:  # B-grade: light denoise
                was_denoised = denoise_file(row["audio_path"], out_path)
                if was_denoised:
                    n_denoised += 1

            row["audio_path"] = out_path
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Kept: {n_kept}  (denoised: {n_denoised})  Rejected: {n_rejected}")
    if n_denoised == 0:
        print("Note: DeepFilterNet not installed -- B-grade files were copied "
              "as-is. Run `pip install deepfilternet` for real denoising.")


if __name__ == "__main__":
    main()
