#!/usr/bin/env bash
# Step 1+2 of the Khmer base training flow, factored out on their own so they
# can run independently of training -- used by kaggle/preprocess_khmer_vq.ipynb
# to VQ-extract + pack ONE data shard at a time (Kaggle's 20GB /kaggle/working
# cap can't hold the whole dataset, let alone the base checkpoint and training
# output alongside it). scripts/10_train_fish_khmer_base.sh still does both
# steps inline for the full-dataset Colab/local path, which has plenty of disk
# and doesn't need this split.
#
#   1. VQ token extraction over $DATASET_DIR (speaker-folder wav/lab pairs)
#   2. Build a packed protobuf dataset into $PROTO_DIR
#
# Usage:
#   DATASET_DIR=... PROTO_DIR=... bash scripts/10a_extract_vq_and_build_protos.sh

set -euo pipefail

FISH_DIR="vendor/fish-speech"
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"

DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the wav/lab folder to process}"
PROTO_DIR="${PROTO_DIR:?set PROTO_DIR to where the packed protobuf shards should go}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-4}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"

if [ ! -d "$FISH_DIR" ]; then
  echo "ERROR: $FISH_DIR not found. Run:"
  echo "  git clone https://github.com/fishaudio/fish-speech $FISH_DIR"
  exit 1
fi

# extract_vq gives no progress output of its own worth relying on (its
# per-worker tqdm bars interleave into a mess once piped to a log file), and
# a large shard at the conservative single-GPU batch/worker sizing can
# legitimately take a while -- with nothing else printed in between, that is
# indistinguishable from a hang. Run it in the background and poll the .npy
# sidecar count every 60s instead, so there is always a concrete
# "N/total done, ETA" line regardless of whatever extract_vq itself prints.
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

mkdir -p "$PROTO_DIR"

# Patch a real upstream bug: FishTokenizer passes the raw tokenizer.tiktoken
# *file* to transformers.AutoTokenizer.from_pretrained(), which has always
# required a *directory* -- confirmed against fishaudio/openaudio-s1-mini's
# actual published files (no tokenizer_config.json ships there). Idempotent.
python scripts/patch_fish_speech_tokenizer.py --fish-dir "$FISH_DIR"

echo "== Step 1: VQ token extraction ($DATASET_DIR) =="
run_extract_vq_with_heartbeat "$DATASET_DIR"

echo "== Step 2: Build protobuf dataset ($PROTO_DIR) =="
python "$FISH_DIR/tools/llama/build_dataset.py" \
  --input "$DATASET_DIR" \
  --output "$PROTO_DIR" \
  --text-extension .lab \
  --num-workers 4

# build_dataset.py only opens its output file INSIDE its results loop, so
# zero input groups (e.g. extract_vq silently produced no .npy sidecars)
# means zero .protos files with no error raised. Fail loudly here instead of
# letting it reach fish-speech's dataloader as an empty proto_files glob.
if [ -z "$(find "$PROTO_DIR" -name '*.proto*' -print -quit 2>/dev/null)" ]; then
  echo "ERROR: build_dataset produced zero proto files in $PROTO_DIR."
  echo "Check that $DATASET_DIR has wav/lab pairs and that extract_vq (Step 1"
  echo "above) actually wrote .npy sidecars next to them."
  exit 1
fi

echo "Done. Protobuf shards ready at $PROTO_DIR"
