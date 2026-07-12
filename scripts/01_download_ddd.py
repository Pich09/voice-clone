#!/usr/bin/env python3
"""
Download DDD Khmer dataset(s) from Hugging Face and export raw audio +
metadata into data/raw/ddd/.

Usage:
    python scripts/01_download_ddd.py \
        --dataset DDD-Cambodia/khm-asr-cultural \
        --split train \
        --out_dir data/raw/ddd

Requires: `datasets`, `huggingface_hub`, `soundfile`, and a HuggingFace
account/token if the dataset is gated (set HF_TOKEN env var, or run
`huggingface-cli login`).
"""
import argparse
import json
import os

import soundfile as sf


def list_shard_files(dataset: str, split: str) -> list[str]:
    """Repo files that make up `split`, in a stable order.

    Some HF dataset repos accumulate multiple overlapping shard-numbering
    generations over time (e.g. train-00000-of-00653.parquet alongside a
    newer train-00653-of-00724.parquet run) -- all of them can match the
    dataset's own "data/train-*" config glob. Letting `datasets` resolve
    and interleave/prefetch across all of them is what caused far more
    parquet downloads than --max_samples should need. Listing + sorting
    ourselves gives full control over exactly which files get touched.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(dataset, repo_type="dataset")
    prefix = f"data/{split}-"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    return shards


def rows_from_shard(dataset: str, shard_file: str, split: str):
    """Download one shard file and yield its rows (small, bounded download)."""
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    local_path = hf_hub_download(
        repo_id=dataset, repo_type="dataset", filename=shard_file,
        token=os.environ.get("HF_TOKEN"),
    )
    shard_ds = load_dataset("parquet", data_files={split: local_path}, split=split)
    yield from shard_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="data/raw/ddd")
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Optional cap for quick smoke tests")
    parser.add_argument("--max_shard_files", type=int, default=100,
                         help="Hard cap on how many parquet shard files may be "
                              "downloaded in the bounded (--max_samples) case, "
                              "even if that means fewer than --max_samples rows "
                              "get exported (e.g. many shards have empty/short "
                              "rows filtered out). Set to 0 to disable.")
    args = parser.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join("data", "manifests", "ddd_raw.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    n_written = 0

    def export_row(i, row, manifest_f):
        nonlocal n_written
        audio = row.get("audio")
        text = row.get("transcript") or row.get("text") or ""
        speaker_id = row.get("speaker_id") or row.get("speaker") or "unknown"

        if audio is None or not text.strip():
            return

        array = audio["array"]
        sr = audio["sampling_rate"]
        fname = f"{speaker_id}_{i:07d}.wav"
        out_path = os.path.join(audio_dir, fname)
        sf.write(out_path, array, sr)

        duration = len(array) / sr
        record = {
            "audio_path": out_path,
            "text": text.strip(),
            "speaker_id": speaker_id,
            "duration": round(duration, 3),
            "source": args.dataset,
        }
        manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_written += 1
        if n_written % 500 == 0:
            print(f"  ... {n_written} samples exported")

    if args.max_samples:
        # Bounded case: fetch shard files one at a time, stop the moment we
        # have enough -- never touches more of the dataset than necessary.
        print(f"Listing shard files for {args.dataset} [{args.split}] ...")
        shard_files = list_shard_files(args.dataset, args.split)
        if not shard_files:
            raise SystemExit(
                f"No shard files found matching data/{args.split}-*.parquet "
                f"in {args.dataset} -- check the dataset's actual file layout."
            )
        if args.max_shard_files:
            shard_files = shard_files[: args.max_shard_files]
        print(f"{len(shard_files)} shard file(s) available; pulling only as many as needed "
              f"for {args.max_samples} samples "
              f"(hard cap: {args.max_shard_files or 'none'} shard files).")

        i = 0
        shards_fetched = 0
        with open(manifest_path, "a", encoding="utf-8") as manifest_f:
            for shard_file in shard_files:
                if n_written >= args.max_samples:
                    break
                print(f"  fetching {shard_file} ...")
                shards_fetched += 1
                for row in rows_from_shard(args.dataset, shard_file, args.split):
                    if n_written >= args.max_samples:
                        break
                    export_row(i, row, manifest_f)
                    i += 1

        if n_written < args.max_samples and args.max_shard_files:
            print(f"  (stopped after hitting the {args.max_shard_files}-shard-file cap, "
                  f"only {n_written} samples collected -- raise --max_shard_files to get more)")
    else:
        # Unlimited: a real full run genuinely needs the whole split, so let
        # `datasets` stream it end to end.
        from datasets import load_dataset
        print(f"Loading full {args.dataset} [{args.split}] (streaming) ...")
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        with open(manifest_path, "a", encoding="utf-8") as manifest_f:
            for i, row in enumerate(ds):
                export_row(i, row, manifest_f)

    print(f"Done. Wrote {n_written} records to {manifest_path}")


if __name__ == "__main__":
    main()
