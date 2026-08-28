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

# `datasets.load_dataset` normally writes a reusable Arrow-format cache to
# ~/.cache/huggingface/datasets/ during parquet->Arrow conversion, separate
# from (and in addition to) the raw file `hf_hub_download` caches. Each shard
# gets its own cache dir there (keyed by a hash of the file path) that is
# NEVER reused since every shard is a distinct file -- so on a full run it
# silently accumulates a second near-full copy of the dataset on disk.
# `keep_in_memory=True` on the load_dataset call is NOT enough to prevent
# this (it only affects whether the *resulting* Dataset is memory-mapped,
# not whether the on-disk prepare step happens) -- disable_caching() is what
# actually skips writing that cache.
import datasets
datasets.disable_caching()


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
    """Download one shard file and yield its rows (small, bounded download).

    Two disk-footprint traps here, both hit in practice on a full ~130-shard
    run and both silent until the drive nearly fills:
    1. `hf_hub_download` on Windows without symlink support (no Developer
       Mode / not running as admin) writes a REAL copy per shard into the
       hub cache (~/.cache/huggingface/hub/datasets--.../snapshots/), not a
       symlink to a shared blob -- so it grows by the full shard size every
       time, never reused. We delete `local_path` ourselves once its rows
       are consumed instead of relying on hub cache eviction (there isn't
       any).
    2. `load_dataset("parquet", ...)` by default also materializes its OWN
       Arrow-format cache under ~/.cache/huggingface/datasets/, a SEPARATE
       copy of the same data again. `keep_in_memory=True` skips writing that
       cache at all -- one shard easily fits in memory.
    Net effect without both fixes: ~2x the compressed dataset size
    accumulates on disk as unrecoverable "cache", on top of the actual
    exported wav output -- enough to fill a modest drive well before the
    download itself finishes.
    """
    from datasets import Audio, load_dataset

    local_path = _hf_hub_download_with_retry(dataset, shard_file)
    try:
        shard_ds = load_dataset("parquet", data_files={split: local_path}, split=split,
                                 keep_in_memory=True)
        # Keep audio as raw bytes rather than letting `datasets` auto-decode --
        # recent `datasets` versions require the extra `torchcodec` dependency
        # for that, which we don't otherwise need. We decode with `soundfile`
        # (already a required dep) in export_row() instead.
        if "audio" in shard_ds.features:
            shard_ds = shard_ds.cast_column("audio", Audio(decode=False))
        yield from shard_ds
    finally:
        # Rows above are fully materialized in memory by keep_in_memory=True,
        # so it's safe to drop the cached parquet file as soon as we're done
        # with it here, even though callers (the shard loop in main()) keep
        # consuming/exporting rows after this generator returns.
        try:
            os.remove(local_path)
        except OSError:
            pass


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
    parser.add_argument("--append", action="store_true",
                         help="Add to an existing data/manifests/ddd_raw.jsonl "
                              "instead of replacing it. Off by default -- see "
                              "the truncation note below. Also required to "
                              "resume a shard-by-shard run (see --resumable).")
    parser.add_argument("--resumable", action="store_true",
                         help="Download shard-by-shard (like the bounded "
                              "--max_samples path) even without a sample cap, "
                              "recording each fully-downloaded shard in a "
                              "state file so a later run with --append can "
                              "skip shards already done instead of "
                              "restarting the whole (streamed) download from "
                              "scratch. Recommended for large unbounded runs.")
    args = parser.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join("data", "manifests", "ddd_raw.jsonl")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    state_path = manifest_path + ".shards_done"

    # Start the manifest (and shard state) fresh unless --append. The
    # per-shard writers below open the manifest in "a" mode so rows survive
    # if a long download is interrupted partway, but that also meant a
    # SECOND run silently stacked its rows on top of the first: run this
    # twice with --max_samples 200 and you get a 400-row manifest, half of it
    # pointing at audio that a later pipeline stage has since deleted
    # (03_audio_qc then grades those D on an unreadable-file exception, so it
    # degrades quietly rather than erroring). kaggle/khmer_tts_kaggle.ipynb
    # worked around this by truncating the file itself before calling this
    # script; owning it here means every caller gets the sane behaviour.
    if not args.append:
        if os.path.exists(manifest_path):
            prev = sum(1 for _ in open(manifest_path, encoding="utf-8"))
            open(manifest_path, "w").close()
            if prev:
                print(f"Replacing existing {manifest_path} ({prev} row(s)); "
                      "pass --append to add to it instead.")
        if os.path.exists(state_path):
            os.remove(state_path)

    shards_done = set()
    if args.append and os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            shards_done = {line.strip() for line in f if line.strip()}
        if shards_done:
            print(f"Resuming: {len(shards_done)} shard(s) already downloaded "
                  f"(from {state_path}), skipping those.")

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

        fname = f"{speaker_id}_{i}.wav"
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

    if args.max_samples or args.resumable:
        # Shard-by-shard case: fetch shard files one at a time. Used for
        # bounded (--max_samples) runs, and for --resumable unbounded runs so
        # a later --append run can skip shards already fully downloaded
        # (recorded in `state_path`) instead of restarting from scratch.
        print(f"Listing shard files for {args.dataset} [{args.split}] ...")
        shard_files = list_shard_files(args.dataset, args.split)
        if not shard_files:
            raise SystemExit(
                f"No shard files found matching data/{args.split}-*.parquet "
                f"in {args.dataset} -- check the dataset's actual file layout."
            )
        if args.max_shard_files:
            shard_files = shard_files[: args.max_shard_files]
        remaining = [f for f in shard_files if f not in shards_done]
        print(f"{len(shard_files)} shard file(s) available "
              f"({len(shard_files) - len(remaining)} already done, "
              f"{len(remaining)} remaining)"
              + (f"; pulling only as many as needed for {args.max_samples} samples"
                 if args.max_samples else "")
              + f" (hard cap: {args.max_shard_files or 'none'} shard files).")

        shards_fetched = 0
        shards_skipped = 0
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        state_f = open(state_path, "a", encoding="utf-8") if args.resumable else None
        try:
            for shard_idx, shard_file in enumerate(shard_files):
                if shard_file in shards_done:
                    continue
                if args.max_samples and n_written >= args.max_samples:
                    break
                print(f"  fetching {shard_file} ...")
                shards_fetched += 1
                try:
                    for row_idx, row in enumerate(rows_from_shard(args.dataset, shard_file, args.split)):
                        if args.max_samples and n_written >= args.max_samples:
                            break
                        export_row(f"{shard_idx:04d}_{row_idx:05d}", row, manifest_f)
                except Exception as e:
                    # A single shard can be permanently unavailable (a broken
                    # signing config on one specific HF Xet CDN blob, seen in
                    # practice -- retries inside rows_from_shard already ruled
                    # out a transient blip). With many shards to pick from,
                    # skip this one and keep going instead of aborting the
                    # whole download over one file.
                    shards_skipped += 1
                    print(f"  SKIPPING {shard_file} -- unavailable after retries ({e!r})")
                    continue
                else:
                    if state_f:
                        state_f.write(shard_file + "\n")
                        state_f.flush()
        finally:
            manifest_f.close()
            if state_f:
                state_f.close()
        if shards_skipped:
            print(f"Skipped {shards_skipped}/{len(shard_files)} unavailable shard file(s).")

        if args.max_samples and n_written < args.max_samples and args.max_shard_files:
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
