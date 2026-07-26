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
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-20000}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$CHECKPOINT_DIR}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-4096}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

if [ ! -d "$FISH_DIR" ]; then
  echo "ERROR: $FISH_DIR not found. Run:"
  echo "  git clone https://github.com/fishaudio/fish-speech $FISH_DIR"
  exit 1
fi

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

# Patch a real upstream bug: FishTokenizer passes the raw tokenizer.tiktoken
# *file* to transformers.AutoTokenizer.from_pretrained(), which has always
# required a *directory* -- confirmed against fishaudio/openaudio-s1-mini's
# actual published files (no tokenizer_config.json ships there). Idempotent.
python scripts/patch_fish_speech_tokenizer.py --fish-dir "$FISH_DIR"

echo "== Step 1: VQ token extraction =="
python "$FISH_DIR/tools/vqgan/extract_vq.py" \
  "$DATASET_DIR" \
  --num-workers 4 \
  --batch-size 16 \
  --config-name "modded_dac_vq" \
  --checkpoint-path "$CHECKPOINT_DIR/codec.pth"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 4

# Same treatment for the held-out validation split, when it exists.
VAL_PROTO_ARG="$PROTO_DIR"
if [ -d "$VAL_DATASET_DIR" ] && [ -n "$(ls -A "$VAL_DATASET_DIR" 2>/dev/null)" ]; then
  echo "== Step 2b: VQ + protobuf for the held-out validation split =="
  mkdir -p "$VAL_PROTO_DIR"
  python "$FISH_DIR/tools/vqgan/extract_vq.py" \
    "$VAL_DATASET_DIR" \
    --num-workers 4 \
    --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "$CHECKPOINT_DIR/codec.pth"
  python "$FISH_DIR/tools/llama/build_dataset.py" \
    --input "$VAL_DATASET_DIR" \
    --output "$VAL_PROTO_DIR" \
    --text-extension .lab \
    --num-workers 4
  VAL_PROTO_ARG="$VAL_PROTO_DIR"
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
CKPT_EVERY=$(( STAGE1_MAX_STEPS < 1000 ? STAGE1_MAX_STEPS : 1000 ))
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  project=khmer_base \
  +lora@model.model.lora_config=r_32_alpha_16_fast \
  train_dataset.proto_files="[$PROTO_DIR]" \
  val_dataset.proto_files="[$VAL_PROTO_ARG]" \
  pretrained_ckpt_path="$PRETRAINED_CKPT" \
  max_length="$TRAIN_MAX_LENGTH" \
  data.batch_size="$TRAIN_BATCH_SIZE" \
  trainer.accumulate_grad_batches="$GRAD_ACCUM" \
  trainer.max_steps="$STAGE1_MAX_STEPS" \
  trainer.val_check_interval=1000 \
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY" \
  callbacks.model_checkpoint.save_top_k=1 \
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
