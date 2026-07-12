#!/usr/bin/env bash
# Fine-tune Fish Speech on the Khmer base dataset (Section 9.1).
#
# Prerequisites:
#   git clone https://github.com/fishaudio/fish-speech vendor/fish-speech
#   cd vendor/fish-speech && pip install -e .
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
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"
OUTPUT_DIR="models/khmer_base"
CONFIG="configs/train_fish_khmer.yaml"

if [ ! -d "$FISH_DIR" ]; then
  echo "ERROR: $FISH_DIR not found. Run:"
  echo "  git clone https://github.com/fishaudio/fish-speech $FISH_DIR"
  exit 1
fi

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

echo "== Step 1: VQ token extraction =="
python "$FISH_DIR/tools/vqgan/extract_vq.py" \
  "$DATASET_DIR" \
  --num-workers 4 \
  --batch-size 16 \
  --config-name "modded_dac_vq"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 4

echo "== Step 3: LoRA fine-tune Khmer base model =="
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  project=khmer_base \
  +lora@model.model.lora_config=r_32_alpha_64 \
  data.train_dataset.proto_files="[\"$PROTO_DIR\"]" \
  data.val_dataset.proto_files="[\"$PROTO_DIR\"]" \
  ckpt_path="$CHECKPOINT_DIR" \
  trainer.max_steps=20000 \
  trainer.val_check_interval=1000 \
  hydra.run.dir="$OUTPUT_DIR"

echo "Done. Khmer base checkpoint saved under $OUTPUT_DIR"
echo "Next: scripts/13_generate_eval_samples.py to sanity check pronunciation."
