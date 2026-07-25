"""
Fish Speech backend. Loads a fine-tuned checkpoint directory (the merged/
output of scripts/10 and 12) through fish-speech's own TTSInferenceEngine
and conforms it to the shared TTSBackend interface (Section 10).

Earlier versions shelled out to vendor/fish-speech/tools/run_inference.py,
which does not exist in the current fish-speech repo (verified against the
actual checkout -- see scripts/14_run_local_inference_smoketest.py, which
established this in-process ModelManager path as the one that works). The
in-process engine is also loaded once and reused, instead of re-loading the
model per sentence.

Speaker identity comes from reference-audio prompting: put
data/speaker_refs/<speaker>/reference.wav plus its transcript in
reference.txt (or reference.lab) next to it. Without a reference the model
falls back to whatever voice the LoRA fine-tune baked in.
"""

import os
import sys
import time

import soundfile as sf

from .base import TTSBackend, SynthesisResult


class FishSpeechBackend(TTSBackend):
    def __init__(self, model_dir: str, fish_speech_dir: str = "vendor/fish-speech",
                 speaker_refs_dir: str = "data/speaker_refs", device: str = "cuda",
                 codec_checkpoint: str | None = None):
        self.model_dir = model_dir
        self.fish_speech_dir = fish_speech_dir
        self.speaker_refs_dir = speaker_refs_dir
        self.device = device
        # The codec (vocoder) is frozen and not part of a fine-tuned/merged
        # checkpoint dir, so it normally comes from the base checkpoint.
        self.codec_checkpoint = codec_checkpoint
        self.model_version = os.path.basename(model_dir.rstrip("/"))
        self._manager = None

    # ---- engine ----------------------------------------------------------
    def _engine(self):
        if self._manager is None:
            fish_dir = os.path.abspath(self.fish_speech_dir)
            if fish_dir not in sys.path:
                sys.path.insert(0, fish_dir)
            from tools.server.model_manager import ModelManager

            codec = self.codec_checkpoint
            if codec is None:
                local = os.path.join(self.model_dir, "codec.pth")
                codec = local if os.path.exists(local) else \
                    os.path.join("checkpoints", "openaudio-s1-mini", "codec.pth")
            if not os.path.exists(codec):
                raise FileNotFoundError(
                    f"codec checkpoint not found at {codec!r} -- pass "
                    "codec_checkpoint= explicitly (it ships with the base "
                    "checkpoint, e.g. checkpoints/openaudio-s1-mini/codec.pth)."
                )

            self._manager = ModelManager(
                mode="tts",
                device=self.device,
                half=False,
                compile=False,
                llama_checkpoint_path=self.model_dir,
                decoder_checkpoint_path=codec,
                decoder_config_name="modded_dac_vq",
            )
        return self._manager.tts_inference_engine

    # ---- speakers ---------------------------------------------------------
    def list_speakers(self) -> list[str]:
        if not os.path.isdir(self.speaker_refs_dir):
            return ["default"]
        return sorted(
            name for name in os.listdir(self.speaker_refs_dir)
            if os.path.isdir(os.path.join(self.speaker_refs_dir, name))
        )

    def _reference_for(self, speaker: str):
        """Return a ServeReferenceAudio for this speaker, or None. Requires
        both reference.wav AND its transcript (reference.txt / reference.lab)
        -- fish-speech's in-context prompting needs the text too."""
        ref_dir = os.path.join(self.speaker_refs_dir, speaker)
        wav = os.path.join(ref_dir, "reference.wav")
        if not os.path.exists(wav):
            return None
        text = None
        for name in ("reference.txt", "reference.lab"):
            p = os.path.join(ref_dir, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    text = f.read().strip()
                break
        if not text:
            return None
        from fish_speech.utils.schema import ServeReferenceAudio
        with open(wav, "rb") as f:
            return ServeReferenceAudio(audio=f.read(), text=text)

    # ---- synthesis ----------------------------------------------------------
    def synthesize(self, text: str, output_path: str, speaker: str = "default",
                    **kwargs) -> SynthesisResult:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        from fish_speech.utils.schema import ServeTTSRequest

        ref = self._reference_for(speaker)
        req = ServeTTSRequest(
            text=text,
            references=[ref] if ref else [],
            reference_id=None,
            # Khmer normalization already happened upstream
            # (khmer_tts/text/normalize.py); fish's normalizer is en/zh-only.
            normalize=False,
        )

        start = time.time()
        chunks = []
        sample_rate = None
        for result in self._engine().inference(req):
            if result.code == "error":
                raise RuntimeError(
                    f"Fish Speech inference failed after {time.time() - start:.1f}s"
                ) from result.error
            if result.code in ("segment", "final") and result.audio is not None:
                sample_rate, chunk = result.audio
                chunks.append(chunk)

        if not chunks:
            raise RuntimeError(f"Fish Speech produced no audio for text: {text!r}")

        import numpy as np
        audio = np.concatenate(chunks)
        sf.write(output_path, audio, sample_rate)

        return SynthesisResult(
            output_path=output_path,
            duration_seconds=len(audio) / sample_rate,
            sample_rate=sample_rate,
            speaker=speaker,
            model_version=self.model_version,
        )
