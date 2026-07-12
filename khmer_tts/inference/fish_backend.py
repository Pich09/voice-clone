"""
Fish Speech backend. Wraps the Fish Speech inference CLI/API so it
conforms to the shared TTSBackend interface (Section 10).

This is the primary backend recommended in Section 3. It expects a
fine-tuned checkpoint directory produced by scripts/10 and 12, plus
reference audio for each speaker (Fish Speech uses reference-audio
prompting for speaker identity rather than discrete speaker-ID tables,
in addition to any LoRA speaker adaptation).
"""

import os
import subprocess
import time

import soundfile as sf

from .base import TTSBackend, SynthesisResult


class FishSpeechBackend(TTSBackend):
    def __init__(self, model_dir: str, fish_speech_dir: str = "vendor/fish-speech",
                 speaker_refs_dir: str = "data/speaker_refs", device: str = "cuda"):
        self.model_dir = model_dir
        self.fish_speech_dir = fish_speech_dir
        self.speaker_refs_dir = speaker_refs_dir
        self.device = device
        self.model_version = os.path.basename(model_dir.rstrip("/"))

    def list_speakers(self) -> list[str]:
        if not os.path.isdir(self.speaker_refs_dir):
            return ["default"]
        return sorted(
            name for name in os.listdir(self.speaker_refs_dir)
            if os.path.isdir(os.path.join(self.speaker_refs_dir, name))
        )

    def _reference_audio_for(self, speaker: str) -> str | None:
        candidate = os.path.join(self.speaker_refs_dir, speaker, "reference.wav")
        return candidate if os.path.exists(candidate) else None

    def synthesize(self, text: str, output_path: str, speaker: str = "default",
                    **kwargs) -> SynthesisResult:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ref_audio = self._reference_audio_for(speaker)

        cmd = [
            "python", os.path.join(self.fish_speech_dir, "tools", "run_inference.py"),
            "--checkpoint-path", self.model_dir,
            "--text", text,
            "--output", output_path,
            "--device", self.device,
        ]
        if ref_audio:
            cmd += ["--reference-audio", ref_audio]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(
                f"Fish Speech inference failed (took {elapsed:.1f}s):\n{result.stderr}"
            )

        info = sf.info(output_path)
        return SynthesisResult(
            output_path=output_path,
            duration_seconds=info.frames / info.samplerate,
            sample_rate=info.samplerate,
            speaker=speaker,
            model_version=self.model_version,
        )
