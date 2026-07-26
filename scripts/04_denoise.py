#!/usr/bin/env python3
"""
Selective denoising, per Section 6.3:
    A-grade -> keep raw (no denoise)
    B-grade -> light denoise
    C/D     -> excluded upstream, but if present here, skip

Denoiser backends, best available wins:
    1. DeepFilterNet (`pip install deepfilternet`) -- best quality, but its
       Rust native extension has no prebuilt wheel on some platforms
       (fails to build on Kaggle, for example).
    2. noisereduce (`pip install noisereduce`) -- pure-python spectral
       gating, installs anywhere. Lighter touch than DeepFilterNet but a
       real denoise, not a no-op.
    3. plain copy-through -- smoke tests only.

Usage:
    python scripts/04_denoise.py \
        --manifest data/manifests/ddd_qc.jsonl \
        --output_dir data/processed/ddd_denoised \
        --output_manifest data/manifests/ddd_denoised.jsonl
"""
import argparse
import json
import os
import shutil


def _load_denoiser():
    """Return (backend_name, denoise_fn) for the best available backend;
    denoise_fn is None for the copy-through fallback."""
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio

        model, df_state, _ = init_df()

        def run_df(input_path, output_path):
            audio, _ = load_audio(input_path, sr=df_state.sr())
            save_audio(output_path, enhance(model, df_state, audio), df_state.sr())

        return "deepfilternet", run_df
    except ImportError:
        pass

    try:
        import noisereduce as nr
        import soundfile as sf

        def run_nr(input_path, output_path):
            data, sr = sf.read(input_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Light touch for B-grade clips: leave some noise floor rather
            # than risk the musical-noise artifacts of full suppression.
            cleaned = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.9)
            sf.write(output_path, cleaned, sr)

        return "noisereduce", run_nr
    except ImportError:
        pass

    return "copy", None


def denoise_file(input_path: str, output_path: str):
    """Denoise with the best available backend; plain copy if none."""
    if not hasattr(denoise_file, "_backend"):
        denoise_file._backend = _load_denoiser()
        print(f"Denoise backend: {denoise_file._backend[0]}")
    name, fn = denoise_file._backend
    if fn is None:
        shutil.copyfile(input_path, output_path)
        return False
    fn(input_path, output_path)
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

    n_kept, n_denoised, n_rejected, n_b_grade = 0, 0, 0, 0

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
                n_b_grade += 1
                was_denoised = denoise_file(row["audio_path"], out_path)
                if was_denoised:
                    n_denoised += 1

            row["audio_path"] = out_path
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"Kept: {n_kept}  (B-grade: {n_b_grade}, denoised: {n_denoised})  "
          f"Rejected: {n_rejected}")
    if n_b_grade == 0:
        print("Note: no B-grade clips in this batch -- A-grade audio is kept "
              "raw by design, so nothing needed denoising.")
    elif n_denoised == 0:
        print("Note: no denoiser installed -- B-grade files were copied as-is. "
              "Run `pip install noisereduce` (works everywhere) or "
              "`pip install deepfilternet` (better, needs a Rust-buildable "
              "platform) for real denoising.")
    if n_kept == 0:
        raise SystemExit(
            f"ERROR: all {n_rejected} clips were rejected by the "
            f"--min_grade_to_keep={args.min_grade_to_keep} filter -- nothing "
            "left to train on. Check scripts/03_audio_qc.py's grade "
            "distribution above."
        )


if __name__ == "__main__":
    main()
