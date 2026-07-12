# Running on Kaggle (free GPU, no API keys)

`khmer_tts_kaggle.ipynb` runs the whole pipeline on Kaggle's free GPU.
Nothing here calls a paid API — the model runs locally on the Kaggle GPU.

## One-time setup

1. **Upload the notebook**: Kaggle ▸ Create ▸ New Notebook ▸ File ▸ Upload Notebook
   ▸ pick `khmer_tts_kaggle.ipynb`.
2. **Accelerator**: Settings ▸ Accelerator ▸ **GPU T4 x2** (or P100).
3. **Internet**: Settings ▸ **Internet ▸ On** (needed for pip + model download).
4. **Get the repo into the notebook** — either:
   - push this repo to GitHub and set `GITHUB_URL` in the Config cell, **or**
   - Add Data ▸ Upload this repo as a Dataset, set `DATASET_PATH`.
5. **(Optional) HF token**: only if the DDD dataset is gated. Add-ons ▸ Secrets ▸
   add `HF_TOKEN` (a **free** Hugging Face token — a download login, not a paid API).
6. **(Stage 2)** Upload your voice recordings + a `{audio_path, text}` JSONL as a
   Dataset, then set `MY_VOICE_MANIFEST`.

Then **Run All**.

## Costs

- **$0 in API/token costs.** No keys, no per-request billing.
- The only "cost" is GPU *time*, and Kaggle gives ~30 GPU-hours/week free.

## Session limits & scaling up

- 12h max per session, 20 GB in `/kaggle/working` (persisted on **Save Version**).
- The Config cell ships **small defaults** (`MAX_SAMPLES=200`, low step counts) so a
  first run finishes fast and proves the pipeline works. Then raise:
  - `MAX_SAMPLES = None` (full dataset)
  - `STAGE1_STEPS` toward `20000`, `STAGE2_STEPS` toward `3000`
  - and resume across sessions (Save Version keeps the checkpoints).

## Collaborative relay (pool GPUs with friends)

Train one shared model across several people's free Kaggle GPUs. Each person
streams a different **shard** of the 700h+ corpus, and a checkpoint is passed
around through a shared Hugging Face repo.

**How it flows (take turns):**
1. Person A: `ENABLE_RELAY=True`, `TRAINER_ID="friendA"`, `SHARD_INDEX=0`,
   `NUM_SHARDS=2` → Run All. It pulls the best checkpoint (none yet → base),
   streams `TAKE_PER_SESSION` clips of shard 0, trains, and **pushes** to
   `HF_CKPT_REPO`.
2. Person B: same notebook, `TRAINER_ID="friendB"`, `SHARD_INDEX=1` → Run All.
   It pulls A's checkpoint, trains on shard 1, pushes an improved one.
3. Repeat, alternating. Each session advances the same shared model.

**Setup:**
- Both people need write access to `HF_CKPT_REPO` and their `HF_TOKEN` secret.
- Give everyone a **unique** `TRAINER_ID` and **unique** `SHARD_INDEX`
  (0 .. `NUM_SHARDS`-1). Whole speakers are hashed to shards, so shards never
  overlap.

**Rules of the road:**
- **Take turns.** The relay holds an *advisory* lock in the repo, but HF has no
  atomic locking — if two people train simultaneously from the same checkpoint
  they diverge and one push is effectively lost. Coordinate ("I'm training now").
- "Best" is decided by validation loss (tracked in `registry.json` in the repo);
  if the logs can't be parsed, it falls back to most-trained (highest step).
- The relay code lives in `khmer_tts/collab/` and is engine-agnostic.

## Notes / gotchas

- **Stage 1 LoRA capacity**: `scripts/10` uses rank-8 LoRA. Learning a whole language
  may want more — bump to `r_32_alpha_64` or a full fine-tune inside that script.
- **Fish CLI drift**: the training scripts call Fish Speech's own CLI; flag names can
  change between versions. If a training step errors, check `vendor/fish-speech/tools/`.
- Cell 11 zips the trained models into `/kaggle/working/artifacts/` for download.
