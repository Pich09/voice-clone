#!/usr/bin/env bash
# Step 3+4 of the Khmer base training flow, taking ALREADY-BUILT protobuf
# shards as input instead of building them from raw audio -- used by
# kaggle/train_khmer_base.ipynb on Kaggle, once kaggle/preprocess_khmer_vq.ipynb
# has finished VQ-extracting + packing every data shard. Skipping Step 1/2
# here is the whole point: Kaggle's 20GB /kaggle/working cap can hold packed
# token protobufs (small) but not the full raw audio dataset (~22GB) at once,
# so this script never touches raw audio at all.
#
#   3. LoRA fine-tune from the pretrained checkpoint, reading $PROTO_FILES
#   4. Merge the LoRA delta into a usable inference checkpoint
#
# Usage:
#   PROTO_FILES='[dir0,dir1]' VAL_PROTO_FILES='[val_dir]' \
#     bash scripts/10b_train_from_protos.sh

set -euo pipefail

FISH_DIR="vendor/fish-speech"
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"
OUTPUT_DIR="models/khmer_base"

# Hydra list-syntax strings, e.g. "[data/fish/shard0_protos,data/fish/shard1_protos]".
PROTO_FILES="${PROTO_FILES:?set PROTO_FILES to a Hydra list of protobuf dirs to train on}"
VAL_PROTO_FILES="${VAL_PROTO_FILES:-$PROTO_FILES}"

# Same overridable knobs as scripts/10_train_fish_khmer_base.sh -- see that
# script's header comment for the full rationale on each.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-20000}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CHECKPOINT_DIR}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-4096}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TRAIN_ACCELERATOR="${TRAIN_ACCELERATOR:-gpu}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"

if [ ! -d "$FISH_DIR" ]; then
  echo "ERROR: $FISH_DIR not found. Run:"
  echo "  git clone https://github.com/fishaudio/fish-speech $FISH_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
python scripts/patch_fish_speech_tokenizer.py --fish-dir "$FISH_DIR"

echo "== Step 3: LoRA fine-tune Khmer base model =="
echo "train protos: $PROTO_FILES"
echo "val protos  : $VAL_PROTO_FILES"
# Checkpoint cadence, stale-checkpoint clearing, and every override below are
# identical to scripts/10_train_fish_khmer_base.sh's Step 3 -- see that
# script's inline comments for why each one exists; not repeated here.
_CKPT_DIR="results/khmer_base/checkpoints"
if [ -n "$(find "$_CKPT_DIR" -name '*.ckpt' -print -quit 2>/dev/null)" ]; then
  echo "Removing stale LoRA-only checkpoint(s) in $_CKPT_DIR (train.py would"
  echo "auto-resume from them and fail with missing base-model keys):"
  find "$_CKPT_DIR" -name '*.ckpt' -print -delete
fi

CKPT_EVERY=$(( STAGE1_MAX_STEPS < 1000 ? STAGE1_MAX_STEPS : 1000 ))
STRATEGY_OVERRIDE=()
if [ "$TRAIN_ACCELERATOR" = "cpu" ]; then
  STRATEGY_OVERRIDE=("~trainer.strategy")
else
  # find_unused_parameters=true (tried first) fixes DDP hangs from PARTIALLY
  # unused params, but DualARTransformer's fast_layers run multiple times
  # per forward pass (once per codebook group) -- so the same LoRA param's
  # backward hook fires more than once in one iteration, which DDP rejects
  # by default ("Expected to mark a variable ready only once"), even with
  # find_unused_parameters. static_graph=true is what PyTorch's own error
  # message recommends for exactly this: it supersedes find_unused_parameters
  # (handles unused params automatically too) and explicitly supports a
  # parameter being touched multiple times per iteration, as long as which
  # parameters participate doesn't change run to run -- true here since the
  # forward pass is deterministic every step.
  STRATEGY_OVERRIDE=("+trainer.strategy.static_graph=true")
fi
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  "${STRATEGY_OVERRIDE[@]}" \
  project=khmer_base \
  +lora@model.model.lora_config=r_32_alpha_16_fast \
  train_dataset.proto_files="$PROTO_FILES" \
  val_dataset.proto_files="$VAL_PROTO_FILES" \
  pretrained_ckpt_path="$PRETRAINED_CKPT" \
  max_length="$TRAIN_MAX_LENGTH" \
  data.batch_size="$TRAIN_BATCH_SIZE" \
  data.num_workers="$TRAIN_NUM_WORKERS" \
  trainer.accumulate_grad_batches="$GRAD_ACCUM" \
  trainer.accelerator="$TRAIN_ACCELERATOR" \
  trainer.max_steps="$STAGE1_MAX_STEPS" \
  trainer.val_check_interval=1000 \
  +trainer.log_every_n_steps=20 \
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY" \
  callbacks.model_checkpoint.save_top_k=1 \
  +callbacks.progress_bar.refresh_rate=20 \
  +logger.csv._target_=lightning.pytorch.loggers.csv_logs.CSVLogger \
  +logger.csv.save_dir="results/khmer_base" \
  +logger.csv.name=csv \
  hydra.run.dir="$OUTPUT_DIR"

echo "== Step 4: Merge LoRA weights into a usable inference checkpoint =="
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
