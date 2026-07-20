# Architecture & Flow — Khmer TTS & Voice Cloning

_Written 2026-07-13. This is the "how does this all fit together" doc — for status/history see `CHECKPOINT.md`._

## 1. The big picture

This project turns Fish Speech (a pretrained multilingual codec-LM TTS engine) into a
**Khmer** TTS engine, then further personalizes it into **your voice**. It does this in
two fine-tuning stages on top of one frozen pretrained checkpoint:

```
                    ┌────────────────────────────┐
                    │  openaudio-s1-mini          │   <- pretrained, frozen,
                    │  (Fish Speech base model)   │      downloaded once from HF
                    └──────────────┬──────────────┘
                                   │
                    Stage 1: LoRA fine-tune on Khmer speech (DDD dataset)
                                   │
                    ┌──────────────▼──────────────┐
                    │  models/khmer_base           │  <- "speaks Khmer" model
                    └──────────────┬──────────────┘
                                   │
                    Stage 2: LoRA fine-tune on YOUR voice (1-3h recordings)
                                   │
                    ┌──────────────▼──────────────┐
                    │  models/my_voice              │  <- "speaks Khmer AS YOU" model
                    └──────────────┬──────────────┘
                                   │
                              Inference (TTS)
                                   │
                            outputs/*.wav
```

Nothing here trains a model from scratch. Every stage is a **LoRA adapter** stacked on
the same frozen backbone — small, cheap, fast to train, and easy to swap out.

## 2. Why LoRA rank differs per stage (important design decision)

- **Stage 1 (learn Khmer)**: needs to teach the model a whole new language's phonetics
  → higher-rank LoRA, `r_32_alpha_16_fast` preset.
- **Stage 2 (learn your voice)**: only needs to nudge timbre/prosody, not relearn
  language → low-rank LoRA, `r_8_alpha_16` preset, so it doesn't overwrite what Stage 1
  just learned ("catastrophic forgetting" risk otherwise).

## 3. Three decoupled boxes (engine-level architecture)

Fish Speech itself is architected as:

```
 text (Khmer graphemes)
        │
        ▼
 ┌─────────────────┐
 │ Text frontend    │   khmer_tts/text/*.py — normalize.py, numbers.py,
 │ (grapheme-based, │   dates.py, currency.py. NO phonemizer/G2P — Fish's
 │  no G2P)         │   tokenizer works directly on Khmer script.
 └────────┬─────────┘
          │ token ids
          ▼
 ┌─────────────────┐
 │ Acoustic codec-LM│   The actual "TTS brain" — an autoregressive
 │ (Fish's LLaMA-   │   transformer (an LLM, but over audio tokens
 │  style model)    │   instead of text tokens). This is what gets
 │                  │   LoRA fine-tuned in Stage 1 & 2.
 └────────┬─────────┘
          │ VQ audio tokens (discrete codes)
          ▼
 ┌─────────────────┐
 │ Vocoder / codec  │   Fish's built-in DAC-style decoder (modded_dac_vq).
 │ decoder          │   Frozen, converts tokens -> waveform. Not fine-tuned.
 └────────┬─────────┘
          │
          ▼
        .wav
```

This decoupling is why fine-tuning is cheap: only the middle box (the LM) gets LoRA
adapters; the text frontend and vocoder are untouched.

## 4. End-to-end data & training pipeline

This is the actual sequence of scripts (`scripts/01`–`14`), run either locally, on
Kaggle, or on Colab via `kaggle/khmer_tts_kaggle.ipynb`:

```
01_download_ddd.py            Download DDD-Cambodia/khmer-speech-dataset from HF
                               (parquet shards, capped at ~100 shards / max_samples)
        │  data/raw/ddd/*.parquet
        ▼
02_export_audio.py            Decode audio out of parquet -> individual wav files
        │  data/processed/ddd_24k/*.wav + data/manifests/ddd_raw.jsonl
        ▼
03_audio_qc.py                 Score each clip (SNR/clipping/etc), drop bad ones
        │  data/manifests/ddd_qc.jsonl
        ▼
04_denoise.py                  Denoise audio
        │  data/processed/ddd_denoised/*.wav + ddd_denoised.jsonl
        ▼
05_vad_trim.py                 Trim silence via VAD (voice activity detection)
        │  data/processed/ddd_vad/*.wav + ddd_vad.jsonl
        ▼
06_loudness_normalize.py       Normalize loudness across clips
        │  ddd_normalized.jsonl
        ▼
07_normalize_khmer_text.py     Normalize Khmer text transcripts (numbers, dates, etc
        │                      via khmer_tts/text/*.py + khmer-nltk word segmentation)
        ▼
08_make_splits.py              Split into train/valid/test
        │  ddd_train.jsonl / ddd_valid.jsonl / ddd_test.jsonl
        ▼
09_convert_to_fish_format.py   Convert into Fish Speech's expected layout:
        │                      speaker-folders of matching .wav + .lab pairs
        │  data/fish/khmer_base/<speaker>/*.wav,*.lab
        ▼
10_train_fish_khmer_base.sh    STAGE 1 TRAINING:
        │                        1. patch_fish_speech_tokenizer.py (upstream bug workaround)
        │                        2. extract_vq.py       -> VQ token extraction (GPU)
        │                        3. build_dataset.py     -> pack into protobuf shards
        │                        4. fish_speech/train.py -> LoRA fine-tune (r_32_alpha_16_fast)
        │  models/khmer_base/  (Khmer-speaking checkpoint)
        ▼
11_convert_my_voice_to_fish.py  Same wav/lab conversion, but for YOUR recordings
        │  data/fish/my_voice/<you>/*.wav,*.lab
        ▼
12_train_fish_my_voice.sh      STAGE 2 TRAINING: same VQ+protobuf+LoRA flow, but
        │                        starting from models/khmer_base (not the raw base
        │                        checkpoint), low-rank LoRA (r_8_alpha_16)
        │  models/my_voice/  (final voice-clone checkpoint)
        ▼
13_generate_eval_samples.py    Generate sample wavs from eval/test_sentences_km.txt
        │                      for listening/QC (objective metrics not wired yet)
        ▼
14_run_local_inference_smoketest.py   One-off: load a checkpoint via
                               fish_speech.inference_engine.TTSInferenceEngine
                               directly and synthesize a sentence to wav.
```

Every script reads/writes JSONL manifests in `data/manifests/` so each stage is
independently re-runnable and inspectable.

## 5. Where each stage actually runs

- **Local machine (WSL2, RTX 3050 4GB)**: good for *verifying the pipeline works*
  end-to-end (data prep + a handful of real training steps + real inference). Too
  little VRAM / too slow for a real fine-tune (WSL2 pages GPU memory to system RAM
  once you exceed 4GB, causing huge slowdowns rather than a clean OOM).
- **Kaggle (free T4 x2)** and **Colab**: where the *real* Stage 1/2 fine-tunes
  (thousands of steps) are meant to run. `kaggle/khmer_tts_kaggle.ipynb` drives the
  whole thing — same scripts 01–14, wrapped in notebook cells with a `run_step()`
  helper so failures raise loudly instead of being swallowed by `!` shell-magic.
- **Colab-specific wrinkle**: WORKDIR is persisted on Google Drive so a session can be
  resumed; the notebook `git pull`s that WORKDIR on reuse. But the *notebook file
  itself* (the `.ipynb` opened in the browser tab) does NOT auto-refresh from a new
  GitHub push — only re-opening it fresh (File → Open notebook → GitHub) picks up
  code changes to the notebook cells themselves.

## 6. Collaborative training (optional, HF-relay)

Because everyone is using free, session-limited GPUs, `khmer_tts/collab/` lets
multiple people take turns training the *same* shared model:

```
 Person A's Kaggle session          Shared HF repo            Person B's session
 ┌───────────────────┐      pull best checkpoint     ┌───────────────────┐
 │ pull_best()        │◄─────────────────────────────│                     │
 │ train on shard A   │                               │                     │
 │ publish()           │──────────────────────────────►│ pull_best()         │
 └───────────────────┘   push new checkpoint+metrics   │ train on shard B    │
                                                        │ publish()            │
                                                        └───────────────────┘
```

- `registry.py` — pick the best checkpoint (`select_best`/`select_latest`) by lowest
  val_loss, pure/tested logic.
- `hf_relay.py` — `HFCheckpointRelay`: ensure_repo → pull_best → publish, with an
  advisory lock (HF has no atomic compare-and-swap, so it's "take turns," not
  concurrency-safe).
- `sharding.py` — each person is assigned a shard of speakers (hashed, no overlap) and
  streams only `TAKE_PER_SESSION` clips per run instead of the whole dataset.
- Gated behind `ENABLE_RELAY` in the notebook — off by default, this is opt-in.

## 7. Serving / inference layer

```
khmer_tts/inference/base.py        TTSBackend interface (engine-agnostic)
khmer_tts/inference/fish_backend.py    Fish Speech backend  ⚠️ currently broken:
                                        shells out to vendor/fish-speech/tools/
                                        run_inference.py, which doesn't exist in
                                        the current fish-speech repo. Real path is
                                        TTSInferenceEngine (see scripts/14 above).
                                        Known issue, not yet fixed (app-serving
                                        layer, deprioritized vs. training).
khmer_tts/inference/f5_backend.py       F5-TTS backend — stub, not implemented.
khmer_tts/inference/cosyvoice_backend.py  CosyVoice backend — stub.
khmer_tts/api.py                        Presumably a serving API wrapping a backend.
Dockerfile.api / docker-compose.yml     Containerized serving.
```

## 8. Repo map (quick reference)

| Path | What it is |
|---|---|
| `scripts/01`–`14` | The pipeline, in order (see §4) |
| `configs/*.yaml` | Preprocessing + training hyperparameters (Hydra-loaded) |
| `data/raw/`, `data/processed/`, `data/manifests/` | Pipeline intermediate outputs (mostly gitignored, regenerable) |
| `data/fish/` | Fish-Speech-formatted training data (wav/lab pairs + protobuf shards) — gitignored, regenerable |
| `checkpoints/openaudio-s1-mini/` | The frozen pretrained base checkpoint (downloaded, gitignored) |
| `models/khmer_base/`, `models/my_voice/` | Stage 1 / Stage 2 output checkpoints (gitignored) |
| `vendor/fish-speech/` | Cloned upstream Fish Speech repo (gitignored except `.gitkeep`) — gets patched by `scripts/patch_fish_speech_tokenizer.py` |
| `khmer_tts/text/` | Khmer text normalization (numbers, dates, currency) used pre-training and at inference time |
| `khmer_tts/collab/` | Optional multi-person HF-relay training |
| `khmer_tts/inference/` | Serving backends (Fish real but has a known bug; F5/CosyVoice stubs) |
| `kaggle/khmer_tts_kaggle.ipynb` | The actual runner — wraps scripts 01–14 for Kaggle/Colab |
| `eval/test_sentences_km.txt` | Khmer sentences used for eval synthesis |
| `outputs/` | Generated wavs, logs |
| `CHECKPOINT.md` | Session-by-session status/history/decisions log |
| `MODEL_CARD.md` | Model documentation (for publishing) |

## 9. Current state (short version — see `CHECKPOINT.md` for full detail)

- ✅ Whole pipeline (01 → 14) verified end-to-end on real hardware locally: real data
  through real VQ extraction, real LoRA training steps, real inference to a wav file.
- ❌ Not yet run for real (thousands of steps, full dataset) — only smoke-scale so far
  (50 samples, 5 steps). That real run is meant to happen on Colab/Kaggle next.
- ⚠️ Currently blocked in practice by a Colab notebook-tab staleness issue (old cell
  code cached in an already-open tab, not re-syncing from GitHub pushes) — needs a
  fresh notebook re-open, not a code fix.
- ❌ `fish_backend.py`'s inference path is broken (wrong script reference) — not fixed
  yet, deprioritized.
- ❌ Objective eval metrics (SECS, Khmer CER), F5-TTS backend, and the HF-relay live
  test are all not done yet.
