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
#     loralib pyrootutils resampy "einx[torch]" zstandard pydub ormsgpack \
#     tiktoken cachetools safetensors grpcio kui opencc-python-reimplemented \
#     modelscope descript-audio-codec gradio wandb silero-vad \
#     "transformers==4.56.1" "protobuf==4.25.5"
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

echo "== Step 3: LoRA fine-tune Khmer base model =="
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  project=khmer_base \
  +lora@model.model.lora_config=r_32_alpha_16_fast \
  train_dataset.proto_files="[$PROTO_DIR]" \
  val_dataset.proto_files="[$PROTO_DIR]" \
  pretrained_ckpt_path="$CHECKPOINT_DIR" \
  trainer.max_steps=20000 \
  trainer.val_check_interval=1000 \
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
  --base-weight "$CHECKPOINT_DIR" \
  --lora-weight "$LATEST_CKPT" \
  --output "$OUTPUT_DIR/merged"

echo "Done. Khmer base checkpoint saved under $OUTPUT_DIR"
echo "Ready-for-inference merged checkpoint: $OUTPUT_DIR/merged"
echo "Next: scripts/13_generate_eval_samples.py to sanity check pronunciation."
