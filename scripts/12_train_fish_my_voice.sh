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
#   TRAIN_BATCH_SIZE / TRAIN_MAX_LENGTH / GRAD_ACCUM
#                     per-GPU memory knobs, same semantics as scripts/10
#                     (defaults OOM a 16GB T4; notebook sets 2/2048/2).
#   TRAIN_ACCELERATOR  overrides base.yaml's hardcoded trainer.accelerator=gpu,
#                     same semantics as scripts/10 -- see that script's note.
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-3000}"
KHMER_BASE_CKPT="${KHMER_BASE_CKPT:-models/khmer_base/merged}"
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
# extract_vq sizing -- see scripts/10 for why these are not hardcoded.
EXTRACT_WORKERS="${EXTRACT_WORKERS:-2}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"

if [ ! -f "$KHMER_BASE_CKPT/model.pth" ]; then
  echo "ERROR: no merged Khmer base checkpoint at $KHMER_BASE_CKPT (model.pth missing)."
  echo "Run scripts/10_train_fish_khmer_base.sh first (its Step 4 writes the merged dir)."
  exit 1
fi

mkdir -p "$PROTO_DIR" "$OUTPUT_DIR"

echo "== Step 1: VQ token extraction on your voice data =="
python "$FISH_DIR/tools/vqgan/extract_vq.py" \
  "$DATASET_DIR" \
  --num-workers "$EXTRACT_WORKERS" \
  --batch-size "$EXTRACT_BATCH_SIZE" \
  --config-name "modded_dac_vq" \
  --checkpoint-path "$BASE_CHECKPOINT_DIR/codec.pth"

echo "== Step 2: Build protobuf dataset =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 2

# build_dataset.py only opens its output file INSIDE its results loop, so
# zero input groups (e.g. extract_vq silently produced no .npy sidecars)
# means zero .protos files with no error raised -- reaches fish-speech's
# dataloader as an empty proto_files glob and crashes with a
# ZeroDivisionError in split_by_rank_worker. Fail loudly here instead.
if [ -z "$(find "$PROTO_DIR" -name '*.proto*' -print -quit 2>/dev/null)" ]; then
  echo "ERROR: build_dataset produced zero proto files in $PROTO_DIR."
  echo "Check that $DATASET_DIR has wav/lab pairs and that extract_vq (Step 1"
  echo "above) actually wrote .npy sidecars next to them."
  exit 1
fi

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
_CKPT_DIR="results/my_voice/checkpoints"
if [ -n "$(find "$_CKPT_DIR" -name '*.ckpt' -print -quit 2>/dev/null)" ]; then
  echo "Removing stale LoRA-only checkpoint(s) in $_CKPT_DIR (train.py would"
  echo "auto-resume from them and fail with missing base-model keys):"
  find "$_CKPT_DIR" -name '*.ckpt' -print -delete
fi

CKPT_EVERY=$(( STAGE2_MAX_STEPS < 500 ? STAGE2_MAX_STEPS : 500 ))
# Same nccl-on-CPU problem as scripts/10 -- see the note there.
STRATEGY_OVERRIDE=()
if [ "$TRAIN_ACCELERATOR" = "cpu" ]; then
  STRATEGY_OVERRIDE=("~trainer.strategy")
fi
python "$FISH_DIR/fish_speech/train.py" \
  --config-name text2semantic_finetune \
  "${STRATEGY_OVERRIDE[@]}" \
  project=my_voice \
  +lora@model.model.lora_config=r_8_alpha_16 \
  train_dataset.proto_files="[$PROTO_DIR]" \
  val_dataset.proto_files="[$PROTO_DIR]" \
  pretrained_ckpt_path="$KHMER_BASE_CKPT" \
  max_length="$TRAIN_MAX_LENGTH" \
  data.batch_size="$TRAIN_BATCH_SIZE" \
  data.num_workers="$TRAIN_NUM_WORKERS" \
  trainer.accumulate_grad_batches="$GRAD_ACCUM" \
  trainer.accelerator="$TRAIN_ACCELERATOR" \
  model.optimizer.lr=1e-5 \
  trainer.max_steps="$STAGE2_MAX_STEPS" \
  trainer.val_check_interval=200 \
  callbacks.model_checkpoint.every_n_train_steps="$CKPT_EVERY" \
  callbacks.model_checkpoint.save_top_k=1 \
  +logger.csv._target_=lightning.pytorch.loggers.csv_logs.CSVLogger \
  +logger.csv.save_dir="results/my_voice" \
  +logger.csv.name=csv \
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
