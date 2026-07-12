"""
FastAPI server exposing the Khmer TTS pipeline (Section 11).

V1 endpoints:
    GET  /health
    GET  /voices
    POST /tts

Run:
    uvicorn khmer_tts.api:app --host 0.0.0.0 --port 8000
"""

import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from khmer_tts.inference.fish_backend import FishSpeechBackend
from khmer_tts.text.normalize import normalize_khmer_text

MODEL_DIR = os.environ.get("KHMER_TTS_MODEL_DIR", "models/my_voice")
SPEAKER_REFS_DIR = os.environ.get("KHMER_TTS_SPEAKER_REFS_DIR", "data/speaker_refs")
OUTPUT_DIR = os.environ.get("KHMER_TTS_OUTPUT_DIR", "outputs/api")
MAX_TEXT_LENGTH = int(os.environ.get("KHMER_TTS_MAX_TEXT_LENGTH", "2000"))
DEVICE = os.environ.get("KHMER_TTS_DEVICE", "cuda")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Khmer TTS API", version="0.1.0")

_backend = None


def get_backend() -> FishSpeechBackend:
    global _backend
    if _backend is None:
        _backend = FishSpeechBackend(
            model_dir=MODEL_DIR,
            speaker_refs_dir=SPEAKER_REFS_DIR,
            device=DEVICE,
        )
    return _backend


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: str = "default"


class TTSResponse(BaseModel):
    audio_url: str
    duration_seconds: float
    sample_rate: int
    speaker: str
    model_version: str
    normalized_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def list_voices():
    backend = get_backend()
    return {"voices": backend.list_speakers()}


@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    backend = get_backend()

    available = backend.list_speakers()
    if req.speaker not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker '{req.speaker}'. Available: {available}",
        )

    normalized = normalize_khmer_text(req.text)
    if not normalized.strip():
        raise HTTPException(status_code=400, detail="Text normalized to empty string.")

    job_id = uuid.uuid4().hex[:12]
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")

    try:
        result = backend.synthesize_long_text(
            text=normalized, output_path=output_path, speaker=req.speaker
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return TTSResponse(
        audio_url=f"/tts/audio/{job_id}",
        duration_seconds=result.duration_seconds,
        sample_rate=result.sample_rate,
        speaker=result.speaker,
        model_version=result.model_version,
        normalized_text=normalized,
    )


@app.get("/tts/audio/{job_id}")
def get_audio(job_id: str):
    # job_id is uuid-hex only -- reject anything else to prevent path traversal
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job id")

    path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")
