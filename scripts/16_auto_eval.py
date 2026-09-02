#!/usr/bin/env python3
"""
Automated proxy scoring for TTS checkpoints, no human listening required
for fast iteration between checkpoints (Section 12 still wants a human
scorecard -- see 13_generate_eval_samples.py -- before actually shipping;
this is for narrowing down which checkpoints are even worth that human
pass).

Proxies used (all approximate -- see AUTO_EVAL caveats in README):
  - Pronunciation  -> CER between input text and an ASR transcript of the
                      generated audio (facebook/mms-1b-all, Khmer target).
  - Voice similarity -> cosine similarity between speaker embeddings
                      (speechbrain ECAPA-TDNN) of the generated clip and
                      the speaker's reference clip.
  - Naturalness    -> UTMOS, a non-intrusive MOS predictor (no reference
                      audio needed).
  - Artifacts      -> clipping ratio + long-silence-gap heuristics on the
                      raw waveform.
  - Long-text stability -> flagged from the same CER: a very high CER
                      usually means the ASR caught repeated or dropped
                      words, not just mispronunciation.

None of these are Khmer-specific tools of the same maturity as English
equivalents (there is no strong dedicated Khmer ASR/MOS model at the time
of writing) -- treat scores as directionally useful for comparing
checkpoints against each other, not as a substitute for the human
scorecard's release gates.

Usage:
    python scripts/16_auto_eval.py \
        --model_dir models/khmer_base \
        --speaker default \
        --sentences eval/test_sentences_km.txt \
        --out_dir outputs/eval/khmer_base_v1

Add --skip_generation to score .wav files that 13_generate_eval_samples.py
(or a previous run of this script) already produced in --out_dir, instead
of re-synthesizing them.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Rough, adjustable proxy thresholds -- NOT the same scale as the human
# 1-5 scorecard. These just flag "worth a human listen" vs "probably fine".
CER_OK = 0.15          # character error rate vs. input text
CER_UNSTABLE = 0.35    # above this, suspect repeated/dropped words
SPEAKER_SIM_OK = 0.70  # cosine similarity, ECAPA embeddings, range [-1, 1]
UTMOS_OK = 3.3         # predicted MOS, range roughly [1, 5]
CLIPPING_OK = 0.001    # fraction of samples at/near full scale
MAX_SILENCE_GAP_S = 2.0  # a single silent gap longer than this is suspect


def _load_asr():
    from transformers import pipeline
    print("Loading ASR model (facebook/mms-1b-all, Khmer)...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/mms-1b-all",
        device=0 if _has_cuda() else -1,
    )
    asr.tokenizer.set_target_lang("khm")
    asr.model.load_adapter("khm")
    return asr


def _load_speaker_embedder():
    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading speaker embedding model (speechbrain ECAPA-TDNN)...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/spkrec-ecapa-voxceleb",
    )


def _load_utmos():
    from speechmos import utmos
    return utmos


def _has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein CER, no external dep needed -- more meaningful than WER
    for Khmer since the script has no spaces between words."""
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(ref)


def _speaker_similarity(embedder, wav_a: str, wav_b: str) -> float:
    import torch
    import torchaudio
    import torch.nn.functional as F

    def embed(path):
        signal, sr = torchaudio.load(path)
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        return embedder.encode_batch(signal).squeeze()

    a, b = embed(wav_a), embed(wav_b)
    return F.cosine_similarity(a, b, dim=0).item()


def _artifact_flags(wav_path: str) -> tuple[float, float]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))

    silent = np.abs(audio) < 1e-4
    max_gap = 0
    run = 0
    for is_silent in silent:
        run = run + 1 if is_silent else 0
        max_gap = max(max_gap, run)
    max_silence_s = max_gap / sr
    return clipping_ratio, max_silence_s


def _reference_wav_for(speaker: str, speaker_refs_dir: str) -> str | None:
    p = os.path.join(speaker_refs_dir, speaker, "reference.wav")
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--speaker", default="default")
    parser.add_argument("--sentences", default="eval/test_sentences_km.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--speaker_refs_dir", default="data/speaker_refs")
    parser.add_argument("--skip_generation", action="store_true",
                         help="Score existing sample_NNN.wav files in --out_dir "
                              "instead of synthesizing fresh ones.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.sentences, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    from khmer_tts.text.normalize import normalize_khmer_text

    wav_paths = []
    if args.skip_generation:
        for i in range(1, len(sentences) + 1):
            wav_paths.append(os.path.join(args.out_dir, f"sample_{i:03d}.wav"))
            if not os.path.exists(wav_paths[-1]):
                raise SystemExit(
                    f"--skip_generation given but {wav_paths[-1]} is missing -- "
                    "run without it, or run 13_generate_eval_samples.py first."
                )
    else:
        from khmer_tts.inference.fish_backend import FishSpeechBackend
        backend = FishSpeechBackend(model_dir=args.model_dir)
        for i, sentence in enumerate(sentences, 1):
            normalized = normalize_khmer_text(sentence)
            out_path = os.path.join(args.out_dir, f"sample_{i:03d}.wav")
            backend.synthesize(text=normalized, output_path=out_path, speaker=args.speaker)
            print(f"[{i}/{len(sentences)}] generated -> {out_path}")
            wav_paths.append(out_path)

    ref_wav = _reference_wav_for(args.speaker, args.speaker_refs_dir)
    if ref_wav is None:
        print(f"Warning: no reference.wav for speaker {args.speaker!r} under "
              f"{args.speaker_refs_dir} -- voice similarity will be skipped.")

    asr = _load_asr()
    embedder = _load_speaker_embedder() if ref_wav else None
    utmos = _load_utmos()

    rows = []
    for i, (sentence, wav_path) in enumerate(zip(sentences, wav_paths), 1):
        normalized = normalize_khmer_text(sentence)

        asr_text = asr(wav_path)["text"].strip()
        cer = _char_error_rate(normalized, asr_text)

        sim = _speaker_similarity(embedder, wav_path, ref_wav) if embedder else None

        mos = utmos.predict(wav_path)  # naturalness, no reference needed

        clipping_ratio, max_silence_s = _artifact_flags(wav_path)

        unstable = cer >= CER_UNSTABLE
        artifacts_ok = clipping_ratio < CLIPPING_OK and max_silence_s < MAX_SILENCE_GAP_S
        pronunciation_ok = cer < CER_OK
        similarity_ok = sim is None or sim >= SPEAKER_SIM_OK
        naturalness_ok = mos >= UTMOS_OK
        overall_ok = pronunciation_ok and similarity_ok and naturalness_ok and artifacts_ok and not unstable

        rows.append({
            "sample": os.path.basename(wav_path),
            "sentence": sentence,
            "asr_transcript": asr_text,
            "cer": round(cer, 4),
            "speaker_similarity": round(sim, 4) if sim is not None else "",
            "utmos": round(float(mos), 3),
            "clipping_ratio": round(clipping_ratio, 5),
            "max_silence_s": round(max_silence_s, 2),
            "long_text_unstable": unstable,
            "auto_pass": overall_ok,
        })
        print(f"[{i}/{len(sentences)}] CER={cer:.3f} sim={sim} utmos={mos:.2f} "
              f"pass={overall_ok}")

    csv_path = os.path.join(args.out_dir, "AUTO_SCORECARD.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(r["auto_pass"] for r in rows)
    md_path = os.path.join(args.out_dir, "AUTO_SCORECARD.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Automated Evaluation (proxy scores, not a human review)\n\n")
        f.write(f"Model: `{args.model_dir}`  \nSpeaker: `{args.speaker}`  \n")
        f.write(f"Auto-pass: {n_pass}/{len(rows)}\n\n")
        f.write("Thresholds: CER < {:.2f} (unstable >= {:.2f}), speaker sim >= {:.2f}, "
                "UTMOS >= {:.2f}, clipping < {:.3f}%, max silence gap < {:.1f}s\n\n"
                .format(CER_OK, CER_UNSTABLE, SPEAKER_SIM_OK, UTMOS_OK,
                        CLIPPING_OK * 100, MAX_SILENCE_GAP_S))
        f.write("| # | Sample | CER | Speaker sim | UTMOS | Clipping% | Max silence(s) | Unstable | Pass |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | {r['sample']} | {r['cer']:.3f} | {r['speaker_similarity']} | "
                     f"{r['utmos']:.2f} | {r['clipping_ratio']*100:.3f} | {r['max_silence_s']:.2f} | "
                     f"{'yes' if r['long_text_unstable'] else ''} | "
                     f"{'PASS' if r['auto_pass'] else 'FAIL'} |\n")
        f.write("\nSamples that FAIL are worth a human listen first. Samples that PASS "
                "are still eligible for the human scorecard before shipping (Section "
                "12.2 release gates) -- this is a pre-filter, not a replacement.\n")

    print(f"\n{n_pass}/{len(rows)} samples auto-passed.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
