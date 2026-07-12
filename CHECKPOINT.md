# Project Checkpoint — Khmer TTS & Voice Cloning

_Last updated: 2026-07-12_

Handoff notes so work can resume without re-deriving context.

## Goal & setup

- Two-stage plan: **Stage 1** DDD Khmer data → Khmer base TTS; **Stage 2** base +
  ~1–3 h of my own voice → voice clone. Engine: **Fish Speech** (codec-LM).
- **Budget = $0.** Train on **free Kaggle GPU** (T4 x2). No paid APIs/tokens.
- Dev machine is Windows 11 (conda env `khmer-asr`); training runs on Kaggle (Linux).

## Dataset

- `DDD-Cambodia/khmer-speech-dataset` — **~727.94 h**, manually curated, multi-speaker.
- Too big for free Kaggle (~125 GB as 24 kHz wav). **Use a ~150–200 h A/B-grade subset**
  (diminishing returns above that for single-voice TTS). Store processed audio as a
  read-only Kaggle *input dataset*, not in the 20 GB working dir.
- ⚠️ Column names not yet confirmed; `collab/sharding.py:detect_keys` auto-detects
  audio/text/speaker but verify on first run.

## Key decisions (agreed)

1. **Architecture = 3 decoupled boxes:** text frontend → acoustic codec-LM → vocoder.
   Keep **grapheme-based** text frontend (NO Khmer G2P/phonemizer).
2. **Engine:** AR codec-LM (Fish) primary; **promote F5-TTS stub to a real fallback**
   behind the existing `TTSBackend` interface (Fish scripts are flag-fragile).
3. **Capacity split (important):**
   - Stage 1 (learn Khmer) = full or **high-rank LoRA (r=32–64)**.
   - Stage 2 (learn my voice) = **freeze backbone + low-rank LoRA (r=8)** to avoid
     forgetting Khmer.
   - ✅ `scripts/10_train_fish_khmer_base.sh` uses `r_32_alpha_16_fast` for Stage 1
     (2026-07-12) — **verified against the real preset**, by actually running training
     locally. The earlier `r_32_alpha_64` guess doesn't exist in fish-speech's repo;
     real options are `r_8_alpha_16` and `r_32_alpha_16_fast` (see
     `vendor/fish-speech/fish_speech/configs/lora/`).
4. **Evaluation is the weak link.** Add objective, automatable metrics to
   `scripts/13_generate_eval_samples.py`: **SECS** (speaker-similarity cosine) and
   **Khmer CER** using the ASR model in the sibling `d:\Khmer-ASR` repo. Not done yet.
5. **Vocoder:** keep Fish's built-in, frozen. Swap to **BigVGAN** only if Khmer audio
   sounds gritty. Ignore WaveGlow; keep LPCNet only for CPU serving.
6. LoRA/LLM clarified for the user: the Fish acoustic model IS an LLM over audio tokens;
   it runs locally, needs no API key, costs nothing. Keep LoRA — dropping it only hurts.

## What has been built this session

- **`kaggle/khmer_tts_kaggle.ipynb`** (36 cells) — full pipeline runner: setup → data
  (scripts 02–09) → Stage 1 train → Stage 2 train → eval → inference → save. Includes
  the collaborative-relay path (gated by `ENABLE_RELAY`).
- **`kaggle/README.md`** — Kaggle setup + relay usage docs.
- **`khmer_tts/collab/`** — checkpoint-relay package (engine-agnostic):
  - `registry.py` — `select_best` / `select_latest` / `bucket_for_key` (pure, tested).
  - `hf_relay.py` — `HFCheckpointRelay`: ensure_repo → pull_best → publish, + registry
    JSON + advisory lock, all in a shared HF repo.
  - `sharding.py` — `stream_shard_to_disk` (per-session shard streaming, key
    auto-detect) + `read_val_loss` (parses Lightning logs).
- **`scripts/smoke_test.py`** — local GPU-free end-to-end test of the data pipeline.

## Collaborative relay (agreed design)

Friends pool free Kaggle GPUs on ONE shared model via a HF repo. **Take turns** (lock is
advisory — HF has no atomic CAS). Each person: unique `TRAINER_ID` + `SHARD_INDEX`;
whole speakers are hashed to shards (no overlap). Flow per session: pull best checkpoint
→ stream `TAKE_PER_SESSION` clips of your shard → train → publish back. "Best" = lowest
val_loss (falls back to highest step).

## Verified vs. NOT verified

- ✅ **Full pipeline verified end-to-end on real hardware (2026-07-12, local RTX 3050
  4GB, WSL2)**, not just a data-only smoke test: download (bounded, shard-by-shard) →
  QC → denoise → VAD → normalize → text-normalize → split → Fish-format convert → real
  **VQ token extraction on GPU** → protobuf dataset build → real **LoRA training steps**
  (actual loss recorded) → real **inference → wav file out**. 50 DDD samples, 5 training
  steps (smoke scale, not a real fine-tune — see below).
- Fixed several real bugs found only by actually running this, not by reading code:
  - `01_download_ddd.py`: `datasets>=5` needs `torchcodec` to auto-decode audio;
    switched to manual `soundfile` decoding instead (avoids the extra dependency).
  - `10_train_fish_khmer_base.sh`: wrong LoRA preset name, wrong `--checkpoint-path`
    default in the VQ step, wrong Hydra config keys (`ckpt_path` → `pretrained_ckpt_path`,
    `data.train_dataset.proto_files` → `train_dataset.proto_files`).
  - **Real upstream fish-speech bug**: `FishTokenizer` passes a raw `tokenizer.tiktoken`
    *file* to `transformers.AutoTokenizer.from_pretrained()`, which has always required
    a *directory*. `fishaudio/openaudio-s1-mini` never published a `tokenizer_config.json`,
    so this fails on every transformers version (tried 4.44.2, 4.56.1 — fish-speech's own
    `uv.lock` pin —, and 4.57.3, identical failure each time). Worked around with
    `scripts/patch_fish_speech_tokenizer.py`, which loads the vocab directly via
    `tiktoken` instead. Wired into the notebook (runs right after cloning
    `vendor/fish-speech`) and into `scripts/10` directly, so it's automatic everywhere.
  - `transformers` unpinned resolves to a 5.x major that breaks fish-speech's tokenizer
    code differently; `protobuf` needs to be >=3.20 (for the `builder` module fish-speech's
    generated `_pb2.py` needs) and <7 (wandb's cap). Pinned `transformers==4.56.1` +
    `protobuf==4.25.5` in the notebook's deps cell.
  - Installing `vendor/fish-speech` with its full declared deps (`pip install -e .`)
    sends pip's resolver into 40+ minutes of backtracking (its `pyproject.toml` hard-pins
    `torch==2.8.0`/`datasets==2.18.0`/etc, conflicting with what's already installed).
    Fixed by installing `--no-deps` and adding the actual runtime deps explicitly.
- ❌ **Not yet run on Kaggle** specifically (Colab and local both verified this session).
- ❌ **Not yet run live:** HF relay round-trip (publish → pull → registry).
- ⚠️ Real Khmer fine-tuning quality is unverified — 5 steps on 50 samples teaches nothing;
  the mechanics are proven, actual pronunciation needs a real run (thousands of steps,
  full/larger dataset) on Colab, where 4GB-VRAM limits and WSL2's slow GPU/host memory
  paging (observed locally: ~4 min/step) don't apply.

## Next steps (priority order)

1. **Run the real Stage 1 fine-tune on Colab**: raise `MAX_SAMPLES` and `STAGE1_STEPS`
   toward real targets (~20000 steps) now that the whole pipeline is proven correct.
2. Verify `khmer-speech-dataset` column names (schema) hasn't drifted.
3. Wire objective eval metrics (SECS + Khmer CER from `d:\Khmer-ASR`) into `scripts/13`.
4. Promote F5-TTS backend from stub to working.
5. Fix `khmer_tts/inference/fish_backend.py` — it shells out to
   `vendor/fish-speech/tools/run_inference.py`, which doesn't exist in the current
   fish-speech repo (real inference goes through `fish_speech.inference_engine
   .TTSInferenceEngine`, see `scripts/14_run_local_inference_smoketest.py` for a working
   example). Not fixed yet — out of scope this session (app-serving layer, not training).
6. (Optional) Add a `TARGET_HOURS` knob (hour-based subsetting) + live-test the relay
   round-trip against a scratch HF repo.

## Reference papers reviewed (for ideas)

Rapid speaker adaptation w/ synthetic data + transfer learning (2312.01107) · Tacotron2
adaptation (2107.12051) · WaveGlow (1811.00002) · End-to-end LPCNet (2202.11301) ·
BigVGAN (2206.04658) · TTS SOTA 2025 survey · Voice Cloning survey (2505.00579).
