#!/usr/bin/env python3
"""
One-off local inference smoke test: load the base openaudio-s1-mini
checkpoint directly via fish_speech's TTSInferenceEngine (bypassing the
tools/run_inference.py CLI, which doesn't exist in this repo -- see
khmer_tts/inference/fish_backend.py's assumption) and synthesize a single
Khmer sentence to a wav file.

Usage:
    python scripts/14_run_local_inference_smoketest.py \
        --checkpoint-dir checkpoints/openaudio-s1-mini \
        --text "..." --out outputs/eval/smoketest.wav
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fish-dir", default="vendor/fish-speech")
    parser.add_argument("--checkpoint-dir", default="checkpoints/openaudio-s1-mini")
    parser.add_argument("--text", default="សួស្តី​ពិភពលោក")
    parser.add_argument("--out", default="outputs/eval/smoketest.wav")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, args.fish_dir)

    from tools.server.model_manager import ModelManager
    from fish_speech.utils.schema import ServeTTSRequest

    manager = ModelManager(
        mode="tts",
        device=args.device,
        half=False,
        compile=False,
        llama_checkpoint_path=args.checkpoint_dir,
        decoder_checkpoint_path=os.path.join(args.checkpoint_dir, "codec.pth"),
        decoder_config_name="modded_dac_vq",
    )

    req = ServeTTSRequest(text=args.text, references=[], reference_id=None)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    audio_chunks = []
    sample_rate = None
    for result in manager.tts_inference_engine.inference(req):
        if result.code == "error":
            raise result.error
        if result.code in ("segment", "final") and result.audio is not None:
            sample_rate, chunk = result.audio
            audio_chunks.append(chunk)

    if not audio_chunks:
        raise SystemExit("No audio produced.")

    import numpy as np
    import soundfile as sf

    full_audio = np.concatenate(audio_chunks)
    sf.write(args.out, full_audio, sample_rate)
    print(f"Wrote {len(full_audio) / sample_rate:.2f}s of audio to {args.out}")


if __name__ == "__main__":
    main()
