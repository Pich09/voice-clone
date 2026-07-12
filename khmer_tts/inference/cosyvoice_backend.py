"""
CosyVoice2 backend (backup, Section 3). Only implement/use this if Fish
Speech's Khmer pronunciation quality doesn't meet the release gates in
Section 12.2.

CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
"""

import os
import time

from .base import TTSBackend, SynthesisResult


class CosyVoiceBackend(TTSBackend):
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model_version = os.path.basename(model_dir.rstrip("/"))
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            # from cosyvoice.cli.cosyvoice import CosyVoice2
            # self._model = CosyVoice2(self.model_dir, load_jit=True, load_trt=False)
            raise NotImplementedError(
                "Install CosyVoice2 (pip install per its repo) and uncomment "
                "the import above before using this backend."
            )
        return self._model

    def list_speakers(self) -> list[str]:
        return ["default"]

    def synthesize(self, text: str, output_path: str, speaker: str = "default",
                    **kwargs) -> SynthesisResult:
        model = self._lazy_load()
        start = time.time()
        # Pseudocode following CosyVoice2's zero-shot/cross-lingual API:
        # for chunk in model.inference_zero_shot(text, prompt_text, prompt_speech):
        #     save chunk to output_path
        raise NotImplementedError("Wire up CosyVoice2 inference call here.")
