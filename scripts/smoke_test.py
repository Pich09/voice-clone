#!/usr/bin/env python3
"""
Local, GPU-free smoke test of the whole DATA pipeline.

Generates a handful of tiny synthetic WAV clips + Khmer transcripts, then runs
the real numbered scripts (02 -> 09) on them, plus text normalization and the
relay registry logic. Proves the plumbing works before you spend any Kaggle GPU
time. Does NOT test Fish Speech training/inference (that needs a GPU + the engine).

Usage:
    python scripts/smoke_test.py
Exit code 0 = all steps passed.

Only needs: numpy, soundfile. (ffmpeg optional; step 06 is skipped without it.)
"""
import json
import os
import shutil
import subprocess
import sys
import wave

import numpy as np

# Windows consoles default to cp1252 and can't print Khmer; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WS = os.path.join(ROOT, "data", "_smoke")           # isolated workspace
AUDIO = os.path.join(WS, "raw_audio")
MAN = os.path.join(WS, "manifests")

# Khmer lines chosen to exercise the normalizer (numbers/currency/date/percent).
LINES = [
    "សួស្តី ខ្ញុំជាជំនួយការសំឡេងភាសាខ្មែរ។",
    "ខ្ញុំមាន $10 នៅឆ្នាំ 2026។",
    "តម្លៃ ៛5000 សម្រាប់ម្ហូបនេះ។",
    "ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026 គឺជាថ្ងៃចាប់ផ្តើម។",
    "តម្លៃបញ្ចុះ 25% សម្រាប់អតិថិជនថ្មី។",
    "ថ្ងៃនេះ អាកាសធាតុល្អណាស់។",
]

PASS, FAIL = "PASS", "FAIL"
results = []


def step(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def gen_audio():
    """Write clean 24kHz mono sine clips (grade-A-eligible) + a raw manifest."""
    if os.path.exists(WS):
        shutil.rmtree(WS)
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    sr = 24000
    raw_manifest = os.path.join(MAN, "smoke_raw.jsonl")
    with open(raw_manifest, "w", encoding="utf-8") as f:
        for i, text in enumerate(LINES):
            dur = 1.0 + 0.4 * i                       # 1.0 .. 3.0 s
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            freq = 180 + 40 * i
            wave_data = 0.2 * np.sin(2 * np.pi * freq * t)  # amp 0.2 -> ~-17 dBFS, no clip
            pcm = (wave_data * 32767).astype("<i2")
            speaker = f"spk_{i % 2}"                   # 2 speakers -> tests sharding/splits
            path = os.path.join(AUDIO, f"{speaker}_{i:04d}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            f.write(json.dumps({
                "audio_path": path, "text": text,
                "speaker_id": speaker, "duration": round(dur, 3),
                "source": "smoke",
            }, ensure_ascii=False) + "\n")
    return raw_manifest


def run(script, *args):
    """Run a pipeline script; return (ok, stdout+stderr tail)."""
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, "\n".join(tail[-3:])


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def main():
    print("=== Khmer TTS pipeline smoke test ===\n")

    # 0. text normalization (pure, no audio) -------------------------------
    try:
        from khmer_tts.text.normalize import normalize_khmer_text, split_sentences
        out = normalize_khmer_text("ខ្ញុំមាន $10 នៅឆ្នាំ 2026។")
        ok = bool(out) and "$" not in out and "10" not in out  # digits verbalized away
        step("text normalization verbalizes numbers/currency", ok, out[:40])
        step("sentence splitter", len(split_sentences(normalize_khmer_text(
            "សួស្តី។ លាហើយ។"))) == 2)
    except Exception as e:
        step("text normalization", False, repr(e))

    # 1. synthetic audio ---------------------------------------------------
    raw = gen_audio()
    step("generate synthetic audio + raw manifest", count_lines(raw) == len(LINES),
         f"{count_lines(raw)} clips")

    # 2. export/merge ------------------------------------------------------
    merged = os.path.join(MAN, "merged.jsonl")
    ok, tail = run("02_export_audio.py", "--inputs", raw, "--output", merged)
    step("02 export/merge", ok and count_lines(merged) == len(LINES), tail)

    # 3. QC grading --------------------------------------------------------
    qc = os.path.join(MAN, "qc.jsonl")
    ok, tail = run("03_audio_qc.py", "--manifest", merged, "--output", qc)
    grades = [json.loads(l)["quality_grade"] for l in open(qc, encoding="utf-8")] if os.path.exists(qc) else []
    step("03 audio QC grading", ok and len(grades) == len(LINES),
         "grades=" + "".join(grades))

    # 4. denoise (copy-through fallback, no deepfilternet) -----------------
    den_dir = os.path.join(WS, "denoised"); den = os.path.join(MAN, "denoised.jsonl")
    ok, tail = run("04_denoise.py", "--manifest", qc,
                   "--output_dir", den_dir, "--output_manifest", den)
    step("04 denoise (graceful fallback)", ok and count_lines(den) > 0, tail)

    # 5. VAD trim (copy-through fallback, no silero) -----------------------
    vad_dir = os.path.join(WS, "vad"); vad = os.path.join(MAN, "vad.jsonl")
    ok, tail = run("05_vad_trim.py", "--manifest", den,
                   "--output_dir", vad_dir, "--output_manifest", vad)
    step("05 VAD trim (graceful fallback)", ok and count_lines(vad) > 0, tail)

    # 6. loudness (needs ffmpeg) — skip cleanly if absent ------------------
    clean = os.path.join(MAN, "clean.jsonl")
    if have_ffmpeg():
        cl_dir = os.path.join(WS, "clean")
        ok, tail = run("06_loudness_normalize.py", "--manifest", vad,
                       "--output_dir", cl_dir, "--output_manifest", clean)
        step("06 loudness normalize", ok and count_lines(clean) > 0, tail)
    else:
        shutil.copyfile(vad, clean)
        step("06 loudness normalize", True, "SKIPPED (no ffmpeg) — routed 05 output onward")

    # 7. Khmer text normalization over the manifest ------------------------
    norm = os.path.join(MAN, "normalized.jsonl")
    ok, tail = run("07_normalize_khmer_text.py", "--manifest", clean, "--output", norm)
    step("07 normalize manifest text", ok and count_lines(norm) > 0, tail)

    # 8. splits ------------------------------------------------------------
    ok, tail = run("08_make_splits.py", "--manifest", norm,
                   "--out_prefix", os.path.join(MAN, "smoke"))
    train = os.path.join(MAN, "smoke_train.jsonl")
    step("08 train/valid/test split", ok and count_lines(train) > 0,
         f"train={count_lines(train)}")

    # 9. validate ----------------------------------------------------------
    ok, tail = run("validate_manifest.py", train)
    step("validate_manifest", ok, tail)

    # 10. convert to Fish speaker-folder format (--copy: Windows-safe) -----
    fish_dir = os.path.join(WS, "fish")
    ok, tail = run("09_convert_to_fish_format.py", "--manifest", train,
                   "--out_dir", fish_dir, "--copy")
    wavs = []
    labs = []
    for dp, _, fns in os.walk(fish_dir):
        wavs += [f for f in fns if f.endswith(".wav")]
        labs += [f for f in fns if f.endswith(".lab")]
    step("09 convert to Fish format (.wav/.lab pairs)",
         ok and len(wavs) > 0 and len(wavs) == len(labs),
         f"{len(wavs)} wav / {len(labs)} lab")

    # 11. relay registry logic (pure) --------------------------------------
    try:
        from khmer_tts.collab import select_best, bucket_for_key
        e = [{"step": 1000, "val_loss": 2.5, "path": "a"},
             {"step": 2000, "val_loss": 2.1, "path": "b"}]
        buckets = {bucket_for_key(f"spk_{i}", 2) for i in range(20)}
        step("relay: best-checkpoint + sharding logic",
             select_best(e)["path"] == "b" and buckets == {0, 1})
    except Exception as ex:
        step("relay logic", False, repr(ex))

    # summary --------------------------------------------------------------
    n_fail = sum(1 for r in results if r[0] == FAIL)
    print("\n" + "=" * 48)
    print(f"{len(results) - n_fail}/{len(results)} steps passed.")
    if n_fail:
        print("FAILED:", ", ".join(r[1] for r in results if r[0] == FAIL))
    else:
        print("All good — the data pipeline works end-to-end. "
              "Safe to move to Kaggle for the GPU training steps.")
    print(f"(workspace left at {WS} for inspection; delete it anytime)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
