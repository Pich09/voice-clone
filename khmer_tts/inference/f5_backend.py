"""
F5-TTS backend (backup, Section 3). Only implement/use this if Fish
Speech's Khmer pronunciation quality doesn't meet the release gates in
Section 12.2.

F5-TTS: https://github.com/SWivid/F5-TTS
"""

import os
import time

from .base import TTSBackend, SynthesisResult


class F5TTSBackend(TTSBackend):
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model_version = os.path.basename(model_dir.rstrip("/"))
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            # from f5_tts.api import F5TTS
            # self._model = F5TTS(model_path=self.model_dir, device=self.device)
            raise NotImplementedError(
                "Install F5-TTS (pip install per its repo) and uncomment "
                "the import above before using this backend."
            )
        return self._model

    def list_speakers(self) -> list[str]:
        return ["default"]

    def synthesize(self, text: str, output_path: str, speaker: str = "default",
                    **kwargs) -> SynthesisResult:
        model = self._lazy_load()
        start = time.time()
        # wav, sr, _ = model.infer(ref_file=..., ref_text=..., gen_text=text)
        raise NotImplementedError("Wire up F5-TTS inference call here.")
