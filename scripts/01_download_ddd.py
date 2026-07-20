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


def _hf_hub_download_with_retry(dataset: str, shard_file: str, attempts: int = 5):
    """hf_hub_download over HF's Xet CDN occasionally 403s with a signature
    error ("invalid key pair id") -- usually a transient failure on HF's
    storage backend (a stale/rotated signing key on their edge), not a
    permissions problem despite the error text. Retrying after a short wait
    clears most occurrences. Occasionally it's NOT transient -- the same
    content-hash keeps failing across every retry with a fresh signed URL
    each time, meaning that specific stored blob has a broken signing config
    server-side. Callers should treat a raised exception here as "this file
    is currently unavailable" rather than assume one more retry will help.
    """
    import time
    from huggingface_hub import hf_hub_download

    last_exc = None
    for attempt in range(attempts):
        try:
            return hf_hub_download(
                repo_id=dataset, repo_type="dataset", filename=shard_file,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            last_exc = e
            wait = 5 * (attempt + 1)
            print(f"    hf_hub_download({shard_file}) attempt {attempt + 1}/{attempts} "
                  f"failed ({e!r}), retrying in {wait}s...")
            time.sleep(wait)
    raise last_exc


def rows_from_shard(dataset: str, shard_file: str, split: str):
    """Download one shard file and yield its rows (small, bounded download)."""
    from datasets import Audio, load_dataset

    local_path = _hf_hub_download_with_retry(dataset, shard_file)
    shard_ds = load_dataset("parquet", data_files={split: local_path}, split=split)
    # Keep audio as raw bytes rather than letting `datasets` auto-decode --
    # recent `datasets` versions require the extra `torchcodec` dependency for
    # that, which we don't otherwise need. We decode with `soundfile` (already
    # a required dep) in export_row() instead.
    if "audio" in shard_ds.features:
        shard_ds = shard_ds.cast_column("audio", Audio(decode=False))
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

        # audio is either already-decoded {"array", "sampling_rate"} (streaming
        # fallback path) or raw {"bytes"/"path"} (shard path, decode=False) --
        # handle both.
        if "array" in audio:
            array, sr = audio["array"], audio["sampling_rate"]
        else:
            import io
            data = audio.get("bytes")
            source = io.BytesIO(data) if data else audio["path"]
            array, sr = sf.read(source)

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
        shards_skipped = 0
        with open(manifest_path, "a", encoding="utf-8") as manifest_f:
            for shard_file in shard_files:
                if n_written >= args.max_samples:
                    break
                print(f"  fetching {shard_file} ...")
                shards_fetched += 1
                try:
                    for row in rows_from_shard(args.dataset, shard_file, args.split):
                        if n_written >= args.max_samples:
                            break
                        export_row(i, row, manifest_f)
                        i += 1
                except Exception as e:
                    # A single shard can be permanently unavailable (a broken
                    # signing config on one specific HF Xet CDN blob, seen in
                    # practice -- retries inside rows_from_shard already ruled
                    # out a transient blip). With many shards to pick from and
                    # only --max_samples needed, skip this one and keep going
                    # instead of aborting the whole download over one file.
                    shards_skipped += 1
                    print(f"  SKIPPING {shard_file} -- unavailable after retries ({e!r})")
                    continue
        if shards_skipped:
            print(f"Skipped {shards_skipped}/{len(shard_files)} unavailable shard file(s).")

        if n_written < args.max_samples and args.max_shard_files:
            print(f"  (stopped after hitting the {args.max_shard_files}-shard-file cap, "
                  f"only {n_written} samples collected -- raise --max_shard_files to get more)")
    else:
        # Unlimited: a real full run genuinely needs the whole split, so let
        # `datasets` stream it end to end.
        from datasets import Audio, load_dataset
        print(f"Loading full {args.dataset} [{args.split}] (streaming) ...")
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))
        with open(manifest_path, "a", encoding="utf-8") as manifest_f:
            for i, row in enumerate(ds):
                export_row(i, row, manifest_f)

    print(f"Done. Wrote {n_written} records to {manifest_path}")


if __name__ == "__main__":
    main()
