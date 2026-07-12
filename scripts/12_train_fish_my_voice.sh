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
KHMER_BASE_CKPT="models/khmer_base"
OUTPUT_DIR="models/my_voice"
CONFIG="configs/train_voice_clone.yaml"

if [ ! -d "$KHMER_BASE_CKPT" ]; then
  echo "ERROR: Khmer base checkpoint not found at $KHMER_BASE_CKPT."
  echo "Run scripts/10_train_fish_khmer_base.sh first."
  exit 1
fi

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

echo "== Step 1: VQ token extraction on your voice data =="
python "$FISH_DIR/tools/vqgan/extract_vq.py" \
  "$DATASET_DIR" \
  --num-workers 2 \
  --batch-size 8 \
  --config-name "modded_dac_vq"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 2

echo "== Step 3: Adapt Khmer base model to your voice (lower LR, fewer steps) =="
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  project=my_voice \
  +lora@model.model.lora_config=r_8_alpha_16 \
  data.train_dataset.proto_files="[\"$PROTO_DIR\"]" \
  data.val_dataset.proto_files="[\"$PROTO_DIR\"]" \
  ckpt_path="$KHMER_BASE_CKPT" \
  model.optimizer.lr=1e-5 \
  trainer.max_steps=3000 \
  trainer.val_check_interval=200 \
  hydra.run.dir="$OUTPUT_DIR"

echo "Done. Your voice clone checkpoint saved under $OUTPUT_DIR"
echo "IMPORTANT: use a LOW learning rate here to avoid overfitting/forgetting"
echo "Khmer pronunciation learned in the base model (see Section 16 risks)."
