#!/usr/bin/env bash
# Fine-tune Fish Speech on the Khmer base dataset (Section 9.1).
#
# Prerequisites (see kaggle/khmer_tts_kaggle.ipynb section 3 for the exact,
# tested version of this flow -- fish-speech's own pyproject.toml pins
# conflict with this repo's deps and cause pip's resolver to hang for 40+
# minutes if installed naively):
#   git clone https://github.com/fishaudio/fish-speech vendor/fish-speech
#   pip install --no-deps -e vendor/fish-speech
#   pip install hydra-core loguru natsort einops rich lightning tensorboard \
#     loralib pyrootutils "einx[torch]" zstandard ormsgpack tiktoken \
#     cachetools safetensors kui "transformers==4.56.1" "protobuf==4.25.5"
#   # DAC codec chain: MUST be --no-deps (descript-audiotools' stale
#   # protobuf<3.20 pin otherwise sends pip backtracking into broken sdists):
#   pip install --no-deps descript-audio-codec==1.0.0 descript-audiotools==0.7.2
#   pip install argbind julius pyloudnorm ffmpy flatten-dict markdown2 \
#     randomname pystoi torch-stoi importlib-resources matplotlib
#   Download base checkpoint (e.g. openaudio-s1-mini) into checkpoints/
#
# This follows Fish Speech's own documented flow:
#   1. VQ token extraction over the speaker-folder dataset
#   2. Build a packed protobuf dataset for fast training
#   3. LoRA fine-tune from the pretrained checkpoint
#
# Usage:
#   bash scripts/10_train_fish_khmer_base.sh

set -euo pipefail

FISH_DIR="vendor/fish-speech"
DATASET_DIR="data/fish/khmer_base"
PROTO_DIR="data/fish/khmer_base_protos"
# Held-out validation set (written by 09_convert_to_fish_format.py from the
# ddd_valid split). Without it, val_dataset would point at the TRAINING
# protos and "val_loss" would just be training loss -- which the HF relay
# then uses to pick its "best" checkpoint, making that choice meaningless.
VAL_DATASET_DIR="data/fish/khmer_base_val"
VAL_PROTO_DIR="data/fish/khmer_base_val_protos"
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"
OUTPUT_DIR="models/khmer_base"
CONFIG="configs/train_fish_khmer.yaml"

# Overridable via environment (used by kaggle/khmer_tts_kaggle.ipynb so the
# notebook never has to sed this file in place -- in-place edits dirty the
# git tree and break the notebook's own `git pull` on a persisted WORKDIR):
#   STAGE1_MAX_STEPS  training steps (default 20000)
#   PRETRAINED_CKPT   dir with model.pth/config.json/tokenizer to fine-tune
#                     FROM (default: the base checkpoint). The HF relay sets
#                     this to a pulled merged checkpoint to resume across
#                     sessions. codec.pth always comes from CHECKPOINT_DIR --
#                     the codec is frozen and merged checkpoints don't ship it.
#   TRAIN_BATCH_SIZE / TRAIN_MAX_LENGTH / GRAD_ACCUM
#                     per-GPU memory knobs. fish-speech's defaults
#                     (batch 4 x 4096-token PACKED sequences per GPU) are
#                     sized for 24GB+ cards and OOM a 16GB T4 -- the
#                     notebook sets 2/2048/2 on Kaggle. GRAD_ACCUM keeps the
#                     effective batch up when the per-GPU batch shrinks.
#   TRAIN_ACCELERATOR  overrides base.yaml's hardcoded trainer.accelerator=gpu.
#                     The notebook sets this to "tpu" when it detects a TPU
#                     and no CUDA GPU. Still EXPERIMENTAL -- fish-speech's
#                     model code was written against CUDA-specific ops, so a
#                     TPU run may still fail inside attention/codec kernels
#                     with no XLA path; this only gets it past the trainer
#                     construction Fish Speech would otherwise hardcode to gpu.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-20000}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CHECKPOINT_DIR}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-4096}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TRAIN_ACCELERATOR="${TRAIN_ACCELERATOR:-gpu}"
# Dataloader worker processes. Python 3.14 changed the default multiprocessing
# start method on Linux from fork to forkserver, and forkserver PICKLES the
# dataset to hand it to each worker -- which fails here because the dataset
# carries a tiktoken-backed FishTokenizer that cannot be pickled. The worker
# dies mid-handshake and the parent reports the unhelpful
# "BrokenPipeError: [Errno 32] Broken pipe" from popen_forkserver._launch.
# Under the old fork default nothing was pickled, so this only bites on 3.14+.
# Forcing fork back is NOT the fix: fork with CUDA already initialized is
# unsafe and is precisely what 3.14 moved away from. num_workers=0 loads in the
# main process, which is correct everywhere and merely slower. The notebook
# sets this to 0 when the interpreter's default start method is not fork.
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
# extract_vq sizing. The old hardcoded 4 workers x batch 16 is sized for
# Kaggle's 2x T4 and CUDA-OOMs a small card (measured: batch 4 alone already
# occupies ~3.9GB of a 4GB RTX 3050). This matters more than it looks --
# extract_vq spawns its workers with sp.Popen and only p.wait()s them WITHOUT
# checking returncodes, so an OOM-killed worker leaves the parent exiting 0
# with .npy sidecars silently missing. The notebook sets these from detected
# VRAM; the defaults below preserve the previous behaviour.
EXTRACT_WORKERS="${EXTRACT_WORKERS:-4}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"

if [ ! -d "$FISH_DIR" ]; then
  echo "ERROR: $FISH_DIR not found. Run:"
  echo "  git clone https://github.com/fishaudio/fish-speech $FISH_DIR"
  exit 1
fi

# extract_vq gives no progress output of its own worth relying on (its
# per-worker tqdm bars interleave into a mess once piped to a log file), and
# a 50k+ sample dataset at the conservative single-GPU batch/worker sizing
# can legitimately take hours -- with nothing else printed in between, that
# is indistinguishable from a hang and burns GPU-hours before anyone notices.
# Run it in the background and poll .npy sidecar count every 60s instead, so
# there is always a concrete "N/total done, ETA" line regardless of whatever
# extract_vq itself is or isn't printing.
run_extract_vq_with_heartbeat() {
  local dir="$1"
  local total
  total=$(find "$dir" -name '*.wav' | wc -l)
  echo "  extracting VQ tokens for $total file(s) in $dir ..."
  python "$FISH_DIR/tools/vqgan/extract_vq.py" \
    "$dir" \
    --num-workers "$EXTRACT_WORKERS" \
    --batch-size "$EXTRACT_BATCH_SIZE" \
    --config-name "modded_dac_vq" \
    --checkpoint-path "$CHECKPOINT_DIR/codec.pth" &
  local pid=$!
  local start
  start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    kill -0 "$pid" 2>/dev/null || break
    local done_n elapsed eta_min
    done_n=$(find "$dir" -name '*.npy' | wc -l)
    elapsed=$(( $(date +%s) - start ))
    eta_min=$(awk -v d="$done_n" -v t="$total" -v e="$elapsed" \
      'BEGIN { if (d > 0) printf "%.1f", (t - d) * e / d / 60; else print "?" }')
    echo "  [extract_vq heartbeat] ${done_n}/${total} sidecars written, ${elapsed}s elapsed, ~${eta_min} min remaining"
  done
  wait "$pid"
}

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

# Patch a real upstream bug: FishTokenizer passes the raw tokenizer.tiktoken
# *file* to transformers.AutoTokenizer.from_pretrained(), which has always
# required a *directory* -- confirmed against fishaudio/openaudio-s1-mini's
# actual published files (no tokenizer_config.json ships there). Idempotent.
python scripts/patch_fish_speech_tokenizer.py --fish-dir "$FISH_DIR"

echo "== Step 1: VQ token extraction =="
run_extract_vq_with_heartbeat "$DATASET_DIR"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 4

# build_dataset.py only opens its output file INSIDE its results loop, so
# zero input groups (e.g. extract_vq silently produced no .npy sidecars)
# means zero .protos files with no error raised. Left undetected, this
# reaches fish-speech's dataloader as an empty proto_files glob and crashes
# with ZeroDivisionError in split_by_rank_worker -- fail loudly here
# instead, before the training process (and its GPU time) even starts.
if [ -z "$(find "$PROTO_DIR" -name '*.proto*' -print -quit 2>/dev/null)" ]; then
  echo "ERROR: build_dataset produced zero proto files in $PROTO_DIR."
  echo "Check that $DATASET_DIR has wav/lab pairs and that extract_vq (Step 1"
  echo "above) actually wrote .npy sidecars next to them."
  exit 1
fi

# Same treatment for the held-out validation split, when it exists.
VAL_PROTO_ARG="$PROTO_DIR"
if [ -d "$VAL_DATASET_DIR" ] && [ -n "$(ls -A "$VAL_DATASET_DIR" 2>/dev/null)" ]; then
  echo "== Step 2b: VQ + protobuf for the held-out validation split =="
  mkdir -p "$VAL_PROTO_DIR"
  run_extract_vq_with_heartbeat "$VAL_DATASET_DIR"
  python "$FISH_DIR/tools/llama/build_dataset.py" \
    --input "$VAL_DATASET_DIR" \
    --output "$VAL_PROTO_DIR" \
    --text-extension .lab \
    --num-workers 4

  # build_dataset.py only opens its output file INSIDE its results loop
  # (tools/llama/build_dataset.py) -- if it finds zero groups (e.g. the val
  # split's extract_vq pass produced no .npy sidecars, which happens
  # silently, no error raised) it writes ZERO .protos files. Checking that
  # the source wav/lab folder was non-empty (above) is not enough -- that
  # only proves the INPUT existed, not that this step produced usable
  # OUTPUT. Without this check a broken/empty validation build passes
  # through as val_dataset.proto_files=[khmer_base_val_protos], and
  # fish-speech's split_by_rank_worker crashes with a ZeroDivisionError
  # (files*... // len(files), len==0) the moment sanity-checking starts --
  # after VQ extraction has already burned the session's GPU time.
  if [ -z "$(find "$VAL_PROTO_DIR" -name '*.proto*' -print -quit 2>/dev/null)" ]; then
    echo "WARNING: validation build produced zero proto files in $VAL_PROTO_DIR"
    echo "-- falling back to validating on the TRAINING set for this run."
    VAL_PROTO_ARG="$PROTO_DIR"
  else
    VAL_PROTO_ARG="$VAL_PROTO_DIR"
  fi
else
  echo "WARNING: no validation data at $VAL_DATASET_DIR -- validating on the"
  echo "TRAINING set, so val_loss will not detect overfitting and the relay's"
  echo "best-checkpoint selection is only as good as 'lowest training loss'."
  echo "Run 09_convert_to_fish_format.py on data/manifests/ddd_valid.jsonl."
fi

echo "== Step 3: LoRA fine-tune Khmer base model =="
# Checkpoint cadence: base.yaml's ModelCheckpoint fires every 5000 steps, so
# any run shorter than that would finish with NO saved checkpoint and Step 4
# below would fail after burning the whole GPU session. Save at least once
# per run, and keep only the latest (save_top_k=1): each Lightning .ckpt
# carries the full ~2GB model state, and only the newest is ever merged.
# Clear stale LoRA checkpoints before training. fish_speech/train.py
# auto-resumes from the newest file in results/<project>/checkpoints
# (get_latest_checkpoint, and it OVERRIDES cfg.ckpt_path, so there is no config
# switch to disable it). Every checkpoint these scripts produce is LoRA-only --
# ~15MB of adapter weights, not the 1.8GB full model -- because training always
# runs with +lora@model.model.lora_config. Lightning restores it with a strict
# load_state_dict against the full TextToSemantic, so resuming ALWAYS fails
# with "Missing key(s) in state_dict: model.embeddings.weight, ...". A session
# killed mid-training (Kaggle's 12h cap) therefore leaves a checkpoint that
# poisons the NEXT run. Resuming across sessions is done properly via
# PRETRAINED_CKPT / KHMER_BASE_CKPT (merged full weights), so nothing of value
# is lost by clearing these.
_CKPT_DIR="results/khmer_base/checkpoints"
if [ -n "$(find "$_CKPT_DIR" -name '*.ckpt' -print -quit 2>/dev/null)" ]; then
  echo "Removing stale LoRA-only checkpoint(s) in $_CKPT_DIR (train.py would"
  echo "auto-resume from them and fail with missing base-model keys):"
  find "$_CKPT_DIR" -name '*.ckpt' -print -delete
fi

CKPT_EVERY=$(( STAGE1_MAX_STEPS < 1000 ? STAGE1_MAX_STEPS : 1000 ))
# base.yaml hardcodes strategy: DDPStrategy(process_group_backend=nccl). nccl
# is CUDA-only, so on a CPU-only host the Trainer cannot even build its process
# group. Drop the whole strategy node there (~ is hydra's delete) and let
# Lightning choose its own default; on GPU/TPU leave it exactly as upstream has
# it, since that is what the multi-GPU Kaggle path relies on.
STRATEGY_OVERRIDE=()
if [ "$TRAIN_ACCELERATOR" = "cpu" ]; then
  STRATEGY_OVERRIDE=("~trainer.strategy")
fi
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  "${STRATEGY_OVERRIDE[@]}" \
  project=khmer_base \
  +lora@model.model.lora_config=r_32_alpha_16_fast \
  train_dataset.proto_files="[$PROTO_DIR]" \
  val_dataset.proto_files="[$VAL_PROTO_ARG]" \
  pretrained_ckpt_path="$PRETRAINED_CKPT" \
  max_length="$TRAIN_MAX_LENGTH" \
  data.batch_size="$TRAIN_BATCH_SIZE" \
  data.num_workers="$TRAIN_NUM_WORKERS" \
  trainer.accumulate_grad_batches="$GRAD_ACCUM" \
  trainer.accelerator="$TRAIN_ACCELERATOR" \
  trainer.max_steps="$STAGE1_MAX_STEPS" \
  trainer.val_check_interval=1000 \
  +trainer.log_every_n_steps=100 \
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY" \
  callbacks.model_checkpoint.save_top_k=1 \
  +callbacks.progress_bar.refresh_rate=100 \
  +logger.csv._target_=lightning.pytorch.loggers.csv_logs.CSVLogger \
  +logger.csv.save_dir="results/khmer_base" \
  +logger.csv.name=csv \
  hydra.run.dir="$OUTPUT_DIR"

echo "== Step 4: Merge LoRA weights into a usable inference checkpoint =="
# train.py's ModelCheckpoint callback writes LoRA-only weights under
# results/<project>/checkpoints/ (paths.run_dir in fish-speech's base.yaml --
# NOT $OUTPUT_DIR/hydra.run.dir, which only holds hydra's own config/log
# clutter). ModelManager/TTSInferenceEngine can't load a LoRA delta directly,
# so merge it onto the base weights first (tools/llama/merge_lora.py).
LATEST_CKPT=$(ls -t "results/khmer_base/checkpoints"/*.ckpt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
  echo "ERROR: no LoRA checkpoint found under results/khmer_base/checkpoints/"
  echo "(training may not have reached callbacks.model_checkpoint.every_n_train_steps yet)."
  exit 1
fi
echo "Merging $LATEST_CKPT"
python "$FISH_DIR/tools/llama/merge_lora.py" \
  --lora-config r_32_alpha_16_fast \
  --base-weight "$PRETRAINED_CKPT" \
  --lora-weight "$LATEST_CKPT" \
  --output "$OUTPUT_DIR/merged"

echo "Done. Khmer base checkpoint saved under $OUTPUT_DIR"
echo "Ready-for-inference merged checkpoint: $OUTPUT_DIR/merged"
echo "Next: scripts/13_generate_eval_samples.py to sanity check pronunciation."
