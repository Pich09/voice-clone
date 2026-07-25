#!/usr/bin/env bash
# Adapt the Khmer base model to your own voice (Section 9.2).
#
# Starts from the Khmer base checkpoint (NOT the original Fish Speech
# pretrained checkpoint) so the model already knows Khmer pronunciation,
# and only needs to learn your speaker identity.
#
# Usage:
#   bash scripts/12_train_fish_my_voice.sh

set -euo pipefail

FISH_DIR="vendor/fish-speech"
DATASET_DIR="data/fish/my_voice"
PROTO_DIR="data/fish/my_voice_protos"
BASE_CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"   # codec.pth lives here (frozen)
OUTPUT_DIR="models/my_voice"
CONFIG="configs/train_voice_clone.yaml"

# Overridable via environment (see scripts/10 for why -- the notebook sets
# these instead of sed'ing this file in place):
#   STAGE2_MAX_STEPS  training steps (default 3000)
#   KHMER_BASE_CKPT   the MERGED Stage-1 checkpoint dir (model.pth +
#                     config.json + tokenizer). Must be the merged/ dir --
#                     models/khmer_base itself only holds hydra clutter.
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-3000}"
KHMER_BASE_CKPT="${KHMER_BASE_CKPT:-models/khmer_base/merged}"

if [ ! -f "$KHMER_BASE_CKPT/model.pth" ]; then
  echo "ERROR: no merged Khmer base checkpoint at $KHMER_BASE_CKPT (model.pth missing)."
  echo "Run scripts/10_train_fish_khmer_base.sh first (its Step 4 writes the merged dir)."
  exit 1
fi

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

echo "== Step 1: VQ token extraction on your voice data =="
python "$FISH_DIR/tools/vqgan/extract_vq.py" \
  "$DATASET_DIR" \
  --num-workers 2 \
  --batch-size 8 \
  --config-name "modded_dac_vq" \
  --checkpoint-path "$BASE_CHECKPOINT_DIR/codec.pth"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 2

echo "== Step 3: Adapt Khmer base model to your voice (lower LR, fewer steps) =="
# pretrained_ckpt_path (NOT ckpt_path): ckpt_path is Lightning's
# resume-full-trainer-state parameter and expects a .ckpt FILE -- handing it
# a checkpoint directory crashes at trainer.fit(). pretrained_ckpt_path is
# fish-speech's own load-these-weights knob and is also what the config's
# tokenizer path (${pretrained_ckpt_path}/tokenizer.tiktoken) interpolates
# from, so the merged Stage-1 dir (which save_pretrained gave a tokenizer)
# satisfies both. Dataset overrides use the top-level train_dataset/
# val_dataset keys exactly like scripts/10 (the verified invocation) -- the
# config's data.* node just interpolates from those.
# Checkpoint cadence: same reasoning as scripts/10 -- base.yaml saves every
# 5000 steps, which a short run never reaches, leaving Step 4 nothing to merge.
CKPT_EVERY=$(( STAGE2_MAX_STEPS < 500 ? STAGE2_MAX_STEPS : 500 ))
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  project=my_voice \
  +lora@model.model.lora_config=r_8_alpha_16 \
  train_dataset.proto_files="[$PROTO_DIR]" \
  val_dataset.proto_files="[$PROTO_DIR]" \
  pretrained_ckpt_path="$KHMER_BASE_CKPT" \
  model.optimizer.lr=1e-5 \
  trainer.max_steps="$STAGE2_MAX_STEPS" \
  trainer.val_check_interval=200 \
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY" \
  callbacks.model_checkpoint.save_top_k=1 \
  hydra.run.dir="$OUTPUT_DIR"

echo "== Step 4: Merge LoRA weights into a usable inference checkpoint =="
# Same as scripts/10 Step 4: the ModelCheckpoint callback writes LoRA-only
# deltas under results/<project>/checkpoints/, which nothing can load
# directly -- merge onto the Stage-1 weights to get an inference-ready dir.
LATEST_CKPT=$(ls -t "results/my_voice/checkpoints"/*.ckpt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
  echo "ERROR: no LoRA checkpoint found under results/my_voice/checkpoints/"
  echo "(training may not have reached callbacks.model_checkpoint.every_n_train_steps yet)."
  exit 1
fi
echo "Merging $LATEST_CKPT"
python "$FISH_DIR/tools/llama/merge_lora.py" \
  --lora-config r_8_alpha_16 \
  --base-weight "$KHMER_BASE_CKPT" \
  --lora-weight "$LATEST_CKPT" \
  --output "$OUTPUT_DIR/merged"

echo "Done. Your voice clone checkpoint saved under $OUTPUT_DIR"
echo "Ready-for-inference merged checkpoint: $OUTPUT_DIR/merged"
echo "IMPORTANT: use a LOW learning rate here to avoid overfitting/forgetting"
echo "Khmer pronunciation learned in the base model (see Section 16 risks)."
