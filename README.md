# Khmer TTS & Voice Cloning

Build a Khmer text-to-speech system, then adapt it to your own voice.

**Stage 1:** DDD Khmer dataset → Khmer base TTS model
**Stage 2:** Khmer base model + your voice recordings → your Khmer voice clone

This repo implements the plan end-to-end: data cleaning → text
normalization → Fish Speech dataset conversion → training → inference →
FastAPI serving → Docker deployment.

## 0. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU training/inference also needs torch matching your CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional but recommended for real (non-smoke-test) runs:
pip install deepfilternet silero-vad

# Fish Speech itself (the actual TTS engine)
git clone https://github.com/fishaudio/fish-speech vendor/fish-speech
pip install -e vendor/fish-speech
# Download a base checkpoint, e.g. openaudio-s1-mini, into checkpoints/
```

## 1. Build the Khmer base model

```bash
# 1. Pull DDD Khmer data
python scripts/01_download_ddd.py --dataset DDD-Cambodia/khm-asr-cultural --out_dir data/raw/ddd

# 2. Consolidate + verify
python scripts/02_export_audio.py --inputs data/manifests/ddd_raw.jsonl --output data/manifests/ddd_raw_merged.jsonl

# 3. Quality control grading (A/B/C/D)
python scripts/03_audio_qc.py --manifest data/manifests/ddd_raw_merged.jsonl --output data/manifests/ddd_qc.jsonl

# 4. Selective denoise (B-grade only)
python scripts/04_denoise.py --manifest data/manifests/ddd_qc.jsonl \
  --output_dir data/processed/ddd_denoised --output_manifest data/manifests/ddd_denoised.jsonl

# 5. VAD trim silence
python scripts/05_vad_trim.py --manifest data/manifests/ddd_denoised.jsonl \
  --output_dir data/processed/ddd_vad --output_manifest data/manifests/ddd_vad.jsonl

# 6. Loudness normalize + resample to 24kHz
python scripts/06_loudness_normalize.py --manifest data/manifests/ddd_vad.jsonl \
  --output_dir data/processed/ddd_24k --output_manifest data/manifests/ddd_clean.jsonl

# 7. Khmer text normalization
python scripts/07_normalize_khmer_text.py --manifest data/manifests/ddd_clean.jsonl \
  --output data/manifests/ddd_normalized.jsonl

# 8. Train/valid/test split
python scripts/08_make_splits.py --manifest data/manifests/ddd_normalized.jsonl --out_prefix data/manifests/ddd

# validate before spending GPU hours on it
python scripts/validate_manifest.py data/manifests/ddd_train.jsonl

# 9. Convert to Fish Speech speaker-folder format
python scripts/09_convert_to_fish_format.py --manifest data/manifests/ddd_train.jsonl --out_dir data/fish/khmer_base

# 10. Fine-tune the Khmer base model
bash scripts/10_train_fish_khmer_base.sh
```

## 2. Clone your own voice

Record 30 min–3+ hours of clean Khmer speech (same room, same mic — see
Section 9.2 of the plan). Log each clip + its transcript as JSONL:

```json
{"audio_path": "data/raw/my_voice/clip_0001.wav", "text": "សួស្តី..."}
```

```bash
python scripts/11_convert_my_voice_to_fish.py \
  --manifest data/manifests/my_voice_raw.jsonl --speaker_id my_voice

bash scripts/12_train_fish_my_voice.sh
```

## 3. Evaluate

```bash
python scripts/13_generate_eval_samples.py \
  --model_dir models/my_voice --speaker my_voice \
  --sentences eval/test_sentences_km.txt --out_dir outputs/eval/my_voice_v1
```

Score each sample per Section 12.2 and check against the release gates
before shipping (Pronunciation ≥ 4.0, Naturalness ≥ 3.8, Voice similarity
≥ 4.0, no severe artifacts).

## 4. Serve

```bash
export KHMER_TTS_MODEL_DIR=models/my_voice
uvicorn khmer_tts.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "សួស្តី ខ្ញុំអាចជួយអ្នកបាន។", "speaker": "my_voice"}'
```

## 5. Deploy

```bash
docker compose up --build
```

## Repo layout

See `khmer_tts/` for the text normalization + inference backend library,
`scripts/` for the numbered pipeline steps above, and `configs/` for
training/preprocessing config files.

## Text normalization

`khmer_tts/text/normalize.py` is the single entry point run identically
before training, validation, and inference (Section 7). It composes:

- `numbers.py` — Arabic/Khmer digit → spoken Khmer word conversion
- `currency.py` — `$10`, `៛5000`, `20000 រៀល` → spoken amounts
- `dates.py` — `ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026`, `05/01/2026` → spoken dates

```python
from khmer_tts.text.normalize import normalize_khmer_text
normalize_khmer_text("ខ្ញុំមាន $10 នៅឆ្នាំ 2026។")
# -> "ខ្ញុំមាន ដប់ដុល្លារ នៅឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ។"
```

## Notes / known gaps

- `scripts/10` and `scripts/12` are shell wrappers around Fish Speech's
  own CLI tools — the exact flag names may drift as Fish Speech evolves,
  check `vendor/fish-speech/tools/` after cloning.
- `CosyVoiceBackend` / `F5TTSBackend` are stubs (Section 3 backups) —
  only implement fully if Fish Speech underperforms on Khmer.
- VAD/denoise scripts fall back gracefully (copy-through) if
  `silero-vad`/`deepfilternet` aren't installed, so the pipeline is
  smoke-testable without the heavy ML deps.
