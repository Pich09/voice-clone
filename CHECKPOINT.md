# Project Checkpoint — Khmer TTS & Voice Cloning

_Last updated: 2026-07-07_

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
   - ✅ `scripts/10_train_fish_khmer_base.sh` bumped to r=32/alpha=64 for Stage 1
     (2026-07-12). Unverified against actual Fish Speech LoRA presets — confirm
     `r_32_alpha_64` exists under `vendor/fish-speech/` config once cloned; fall back
     to whatever preset name Fish Speech ships if it doesn't.
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

- ✅ **Local smoke test: 13/13 passed.** Data pipeline 02–09 + text normalization +
  splits + Fish-format conversion + relay logic all work. `silero-vad` IS installed
  locally; `ffmpeg` is NOT (step 06 skipped locally — Kaggle has it).
- ❌ **Not yet run on Kaggle/GPU:** Fish install, Stage 1/2 training, inference.
- ❌ **Not yet run live:** HF relay round-trip (publish → pull → registry).
- ⚠️ **Highest risk:** Fish Speech CLI flag drift in `scripts/10` & `scripts/12`
  (their `tools/*` paths/flags change between versions).

## Next steps (priority order)

1. **Minimal first Kaggle run:** `MAX_SAMPLES=200`, `ENABLE_RELAY=False`, tiny
   `STAGE1_STEPS` — just confirm Fish installs and the training command runs. Paste any
   error back to reconcile flags.
2. Verify `khmer-speech-dataset` column names (schema) on Kaggle.
3. Wire objective eval metrics (SECS + Khmer CER from `d:\Khmer-ASR`) into `scripts/13`.
4. Fix Stage-1 LoRA capacity (r=32–64 or full) in `scripts/10`.
5. Promote F5-TTS backend from stub to working.
6. (Optional) Add a `TARGET_HOURS` knob (hour-based subsetting) + a smoke-mode cell in
   the notebook; live-test the relay round-trip against a scratch HF repo.

## Reference papers reviewed (for ideas)

Rapid speaker adaptation w/ synthetic data + transfer learning (2312.01107) · Tacotron2
adaptation (2107.12051) · WaveGlow (1811.00002) · End-to-end LPCNet (2202.11301) ·
BigVGAN (2206.04658) · TTS SOTA 2025 survey · Voice Cloning survey (2505.00579).
