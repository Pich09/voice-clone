"""
Abstract backend interface so the API/CLI is never locked to one TTS
model (Section 10). Every concrete backend (FishSpeechBackend,
CosyVoiceBackend, F5TTSBackend) implements this same contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SynthesisResult:
    output_path: str
    duration_seconds: float
    sample_rate: int
    speaker: str
    model_version: str


class TTSBackend(ABC):
    """Common interface for all TTS backends."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str, speaker: str = "default",
                    **kwargs) -> SynthesisResult:
        """
        Synthesize `text` (already Khmer-normalized) into a WAV file at
        `output_path`, using the given speaker/voice identity.
        """
        raise NotImplementedError

    @abstractmethod
    def list_speakers(self) -> list[str]:
        """Return the list of available speaker/voice identities."""
        raise NotImplementedError

    def synthesize_long_text(self, text: str, output_path: str, speaker: str = "default",
                              **kwargs) -> SynthesisResult:
        """
        Default long-text strategy (Section 16, 'Long text unstable' fix):
        split into sentences, synthesize each separately, concatenate
        with short pauses. Backends may override this with something
        smarter (native streaming, chunk-level context, etc).
        """
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        from khmer_tts.text.normalize import split_sentences

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return self.synthesize(text, output_path, speaker, **kwargs)

        pieces = []
        sr = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, sentence in enumerate(sentences):
                tmp_path = os.path.join(tmp_dir, f"part_{i:03d}.wav")
                result = self.synthesize(sentence, tmp_path, speaker, **kwargs)
                data, file_sr = sf.read(tmp_path)
                sr = sr or file_sr
                pieces.append(data)

        pause = np.zeros(int(0.35 * sr))  # 350ms pause between sentences
        full_audio = np.concatenate(
            [seg for piece in pieces for seg in (piece, pause)][:-1]
        )
        sf.write(output_path, full_audio, sr)

        return SynthesisResult(
            output_path=output_path,
            duration_seconds=len(full_audio) / sr,
            sample_rate=sr,
            speaker=speaker,
            model_version=getattr(self, "model_version", "unknown"),
        )
