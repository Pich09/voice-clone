#!/usr/bin/env python3
"""
Build ready-to-train VQ token shards from a big HF speech corpus, chunk by
chunk, and publish them to an HF *dataset* repo.

Why this exists
---------------
Training reads ONLY Fish Speech's protobuf shards -- never the audio. Measured
on this corpus, those shards are ~451 bytes per second of audio, i.e. ~71x
smaller than the source WAVs: the whole ~1,014-hour DDD corpus becomes ~1.7GB.
So instead of every training session re-downloading and re-preprocessing audio
(the expensive part, dominated by GPU VQ extraction), you pay that cost ONCE,
publish the shards, and every later session pulls ~1.7GB and trains directly.

Design notes
------------
* Chunked + resumable. Kaggle gives 20GB in /kaggle/working and 12h sessions,
  and the corpus is ~128GB. Parquet files are fetched a chunk at a time,
  processed, uploaded, and DELETED. Progress lives in the HF repo (not on local
  disk), so a session timeout costs only the chunk in flight -- rerun and it
  picks up where it stopped.
* Fused single pass. The scripts/02-09 pipeline writes a full copy of the audio
  at every stage and spawns one ffmpeg per clip; across ~450k clips on a
  4-vCPU Kaggle box that process-spawning would dominate the runtime. Here each
  clip is decoded, QC'd, trimmed and loudness-normalised in memory with numpy,
  then written once.
* One resample, not two. scripts/06 resamples to 24kHz, then extract_vq
  resamples again to the codec's 44.1kHz. This keeps the corpus's native 16kHz
  on disk and lets extract_vq do a single resample on the GPU.
* De-duplicated on (speaker_id, sentence_id). The source repo holds several
  parquet "generations" from separate upload efforts. The 653-file and vN
  generations cover disjoint topics (verified) so they are additive, but the
  71-file `-of-00724` set looks like an abandoned re-pack of existing content,
  so it is skipped by default. A persisted hash set makes this robust anyway.

Usage
-----
    # measure throughput on a small sample first -- ALWAYS do this
    python scripts/15_build_token_shards.py --repo you/ddd-speech-shard --pilot 2000

    # then the real run; re-invoke after each session until it reports done
    python scripts/15_build_token_shards.py --repo you/ddd-speech-shard --chunk-gb 15
"""
import argparse
import gc
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROGRESS_PATH = "progress.json"
SEEN_PATH = "seen_keys.npy"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sanitize(name):
    safe = "".join(c if c.isalnum() else "_" for c in str(name))
    return safe or "unknown_speaker"


def key_hash(speaker, sentence_id):
    h = hashlib.blake2b(f"{speaker}|{sentence_id}".encode("utf-8"), digest_size=8)
    return np.frombuffer(h.digest(), dtype=np.uint64)[0]


def load_qc_module():
    """Reuse scripts/03_audio_qc.py's grade() so thresholds stay in one place
    (the filename starts with a digit, so it can't be imported normally)."""
    import importlib.util
    p = os.path.join(os.path.dirname(__file__), "03_audio_qc.py")
    spec = importlib.util.spec_from_file_location("audio_qc", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics_from_array(data, sr):
    """Same measurements as scripts/03, computed in memory."""
    n = len(data)
    peak = float(np.max(np.abs(data))) if n else 0.0
    rms = float(np.sqrt(np.mean(data ** 2))) if n else 0.0
    clipping = float(np.mean(np.abs(data) >= 0.999)) if n else 1.0
    frame_len = max(int(0.02 * sr), 1)
    nf = n // frame_len
    if nf == 0:
        silence = 1.0
    else:
        frames = data[: nf * frame_len].reshape(nf, frame_len)
        fdb = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-9))
        silence = float(np.mean(fdb < -40))
    return {
        "duration": round(n / sr, 3),
        "sample_rate": sr,
        "peak_db": round(20 * np.log10(max(peak, 1e-9)), 2),
        "rms_db": round(20 * np.log10(max(rms, 1e-9)), 2),
        "clipping_ratio": round(clipping, 4),
        "silence_ratio": round(silence, 4),
    }


def energy_trim(data, sr, thresh_db=-40.0, pad_ms=50):
    """Trim leading/trailing silence by frame energy.

    Deliberately not Silero VAD: at ~0.1-0.3s per clip a neural VAD costs
    hours of CPU across ~450k clips, and this corpus is already curated
    sentence-level segments. This is numpy-fast and gets the part that matters
    for TTS (clean starts/stops). Use --trim silero if you want the real thing.
    """
    frame_len = max(int(0.02 * sr), 1)
    nf = len(data) // frame_len
    if nf == 0:
        return data
    frames = data[: nf * frame_len].reshape(nf, frame_len)
    fdb = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-9))
    voiced = np.flatnonzero(fdb >= thresh_db)
    if voiced.size == 0:
        return data
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame_len - pad)
    end = min(len(data), (voiced[-1] + 1) * frame_len + pad)
    return data[start:end]


def normalize_loudness(data, sr, target_lufs, meter_cache={}):
    """Loudness-normalise with pyloudnorm (pure numpy -- no ffmpeg subprocess
    per clip, which is what makes this affordable at corpus scale)."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        return data
    if sr not in meter_cache:
        meter_cache[sr] = pyln.Meter(sr)
    try:
        loudness = meter_cache[sr].integrated_loudness(data)
        if not np.isfinite(loudness):
            raise ValueError("non-finite loudness")
        # pyloudnorm warns "Possible clipped samples" whenever the gain would
        # push past full scale. We rescale below, so the warning is noise --
        # and at ~5% of clips it would emit tens of thousands of lines.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pyln.normalize.loudness(data, loudness, target_lufs)
    except Exception:
        # Clips shorter than the meter's 400ms block, or pure silence.
        peak = np.max(np.abs(data)) if len(data) else 0.0
        out = data * (0.7 / peak) if peak > 0 else data
    # Keep true peak under control so the 16-bit write doesn't clip.
    peak = np.max(np.abs(out)) if len(out) else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out


# --------------------------------------------------------------------------
# HF repo state
# --------------------------------------------------------------------------
class ShardRepo:
    def __init__(self, repo_id, token, source_dataset):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        self.source_dataset = source_dataset

    def ensure(self):
        from huggingface_hub import create_repo
        create_repo(self.repo_id, token=self.token, repo_type="dataset",
                    private=False, exist_ok=True)
        try:
            self.api.file_exists(self.repo_id, "README.md", repo_type="dataset")
            has_readme = self.api.file_exists(self.repo_id, "README.md",
                                               repo_type="dataset")
        except Exception:
            has_readme = False
        if not has_readme:
            self._upload_bytes(self._readme().encode("utf-8"), "README.md")
            log("wrote dataset card (attribution + license)")

    def _readme(self):
        return f"""---
license: cc-by-sa-4.0
language: [km]
task_categories: [text-to-speech]
tags: [khmer, tts, fish-speech, vq-tokens, audio-tokens]
---

# Khmer TTS token shards (Fish Speech / OpenAudio S1-mini codec)

Pre-extracted **VQ audio tokens** + normalised Khmer transcripts, packed as
Fish Speech protobuf shards and ready to train on directly -- no audio
download, no preprocessing, no GPU token-extraction pass.

Each second of speech is ~451 bytes here versus ~32KB as 16-bit PCM, so the
whole corpus fits in a couple of GB and a training session can hold *all* of
it instead of streaming a slice.

## Provenance & license

Derived from [`{self.source_dataset}`](https://huggingface.co/datasets/{self.source_dataset})
by DDD-Cambodia, licensed **CC BY-SA 4.0**. These tokens are a derivative
work: they are decodable back to (lossy) audio with the public
`fishaudio/openaudio-s1-mini` codec, so this dataset is redistributed under
the **same CC BY-SA 4.0 license**, with attribution to DDD-Cambodia.

## Contents

- `shards/shard_XXXXX/*.protos` -- Fish Speech `TextData` records: normalised
  transcript + VQ codes per utterance, grouped by speaker.
- `shards/shard_XXXXX/meta.json` -- clip count, hours, source parquet files.
- `progress.json` -- which source files have been processed (build resumes
  from this).
- `seen_keys.npy` -- `(speaker_id, sentence_id)` hashes, for de-duplication.

## How the tokens were made

Per clip: decode -> QC grade (scripts/03 thresholds, grade D dropped) ->
energy-based silence trim -> loudness-normalise to the target LUFS -> Khmer
text normalisation (numbers/dates/currency spelled out) -> Fish Speech
`extract_vq` with the `modded_dac_vq` codec -> `build_dataset` protobuf pack.

**Tokens are codec-specific.** They are only meaningful for the exact codec
recorded in each shard's `meta.json`. A different codec checkpoint requires
rebuilding.

## Use

```python
# in a Fish Speech training run
train_dataset.proto_files="[shards]"   # the loader rglobs *.protos
```
"""

    def _upload_bytes(self, data, path_in_repo):
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path_in_repo,
                    repo_id=self.repo_id, repo_type="dataset", token=self.token)

    def _download_json(self, path):
        from huggingface_hub import hf_hub_download
        try:
            fp = hf_hub_download(self.repo_id, path, repo_type="dataset",
                                 token=self.token, force_download=True)
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def load_progress(self):
        return self._download_json(PROGRESS_PATH) or {
            "processed_files": [], "clips": 0, "hours": 0.0,
            "shards": [], "rejected": 0, "duplicates": 0,
        }

    def save_progress(self, prog):
        self._upload_bytes(json.dumps(prog, indent=1, ensure_ascii=False).encode(),
                           PROGRESS_PATH)

    def load_seen(self):
        from huggingface_hub import hf_hub_download
        try:
            fp = hf_hub_download(self.repo_id, SEEN_PATH, repo_type="dataset",
                                 token=self.token, force_download=True)
            return set(np.load(fp).tolist())
        except Exception:
            return set()

    def save_seen(self, seen):
        buf = io.BytesIO()
        np.save(buf, np.array(sorted(seen), dtype=np.uint64))
        self._upload_bytes(buf.getvalue(), SEEN_PATH)

    def upload_shard(self, local_dir, shard_id):
        self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset",
                               token=self.token, folder_path=local_dir,
                               path_in_repo=f"shards/{shard_id}",
                               commit_message=f"add {shard_id}")


# --------------------------------------------------------------------------
# source file listing
# --------------------------------------------------------------------------
def list_source_files(dataset, token, include_partial):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(dataset, repo_type="dataset")
             if f.endswith(".parquet")]
    complete = sorted(f for f in files if re.search(r"-of-00653\.parquet$", f))
    versioned = sorted(f for f in files if re.search(r"train-v\d+-\d+\.parquet$", f))
    partial = sorted(f for f in files if re.search(r"-of-00724\.parquet$", f))
    log(f"source generations: 653-gen={len(complete)}, vN-gen={len(versioned)}, "
        f"partial-724={len(partial)}")
    picked = complete + versioned
    if include_partial:
        picked += partial
        log("including the partial -of-00724 set (likely a re-pack; dedup will "
            "drop anything already seen)")
    else:
        log(f"skipping {len(partial)} partial -of-00724 files (probable re-pack "
            "of existing content; --include-partial-724 to include)")
    return picked


def fetch_parquet(dataset, filename, dest_dir, token):
    """Download one parquet into dest_dir. Avoids the shared HF cache so the
    file can simply be deleted afterwards to reclaim disk."""
    from huggingface_hub import hf_hub_download
    kwargs = dict(repo_id=dataset, filename=filename, repo_type="dataset",
                  token=token, local_dir=dest_dir)
    try:
        return hf_hub_download(**kwargs)
    except TypeError:
        p = hf_hub_download(repo_id=dataset, filename=filename,
                            repo_type="dataset", token=token)
        return p


# --------------------------------------------------------------------------
# the per-clip work
# --------------------------------------------------------------------------
def process_parquet(path, pending_dir, seen, qc, normalize_text, args, stats):
    """Decode + QC + trim + normalise every usable row into pending_dir as
    wav/lab pairs under a per-speaker folder. Returns clips written."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["audio", "transcript", "speaker_id",
                                          "sentence_id", "duration"])
    audio_col = table.column("audio")
    text_col = table.column("transcript").to_pylist()
    spk_col = table.column("speaker_id").to_pylist()
    sid_col = table.column("sentence_id").to_pylist()

    written = 0
    for i in range(table.num_rows):
        speaker = spk_col[i] or "unknown"
        sid = sid_col[i] or f"row{i}"
        kh = key_hash(speaker, sid)
        if kh in seen:
            stats["duplicates"] += 1
            continue

        raw_text = (text_col[i] or "").strip()
        if not raw_text:
            stats["rejected"] += 1
            continue

        cell = audio_col[i].as_py()
        blob = cell.get("bytes")
        if not blob:
            stats["rejected"] += 1
            continue
        try:
            data, sr = sf.read(io.BytesIO(blob), always_2d=False)
        except Exception:
            stats["rejected"] += 1
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)

        m = metrics_from_array(data, sr)
        grade = qc.grade(m, raw_text, args.a_grade_min_sr)
        if grade == "D" or (args.min_grade == "B" and grade == "C"):
            stats["rejected"] += 1
            continue

        if args.trim == "energy":
            data = energy_trim(data, sr)
        if len(data) < int(0.3 * sr):
            stats["rejected"] += 1
            continue
        data = normalize_loudness(data, sr, args.target_lufs)

        text = normalize_text(raw_text)
        if not text.strip():
            stats["rejected"] += 1
            continue

        spk_dir = pending_dir / sanitize(speaker)
        spk_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize(sid)
        sf.write(str(spk_dir / f"{stem}.wav"), data, sr, subtype="PCM_16")
        (spk_dir / f"{stem}.lab").write_text(text + "\n", encoding="utf-8")

        seen.add(kh)
        written += 1
        stats["clips"] += 1
        stats["seconds"] += len(data) / sr
        stats["grades"][grade] = stats["grades"].get(grade, 0) + 1

    del table, audio_col
    gc.collect()
    return written


def run(cmd, label):
    log(f"$ {' '.join(str(c) for c in cmd[:6])} ...")
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {r.returncode})")
    return time.time() - t0


def flush_to_shard(pending_dir, out_root, shard_id, args, source_files, timing):
    """extract_vq -> build_dataset -> a self-contained shard folder."""
    wavs = list(pending_dir.rglob("*.wav"))
    if not wavs:
        return None
    log(f"flushing {len(wavs)} clips -> {shard_id}")

    timing["extract_vq"] += run([
        sys.executable, Path(args.fish_dir) / "tools/vqgan/extract_vq.py",
        pending_dir,
        "--num-workers", args.extract_workers,
        "--batch-size", args.extract_batch_size,
        "--config-name", "modded_dac_vq",
        "--checkpoint-path", args.codec,
    ], "extract_vq")

    shard_dir = out_root / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    timing["build_dataset"] += run([
        sys.executable, Path(args.fish_dir) / "tools/llama/build_dataset.py",
        "--input", pending_dir,
        "--output", shard_dir,
        "--text-extension", ".lab",
        "--num-workers", args.build_workers,
    ], "build_dataset")

    n_npy = len(list(pending_dir.rglob("*.npy")))
    seconds = 0.0
    for w in wavs:
        try:
            info = sf.info(str(w))
            seconds += info.frames / info.samplerate
        except Exception:
            pass
    proto_bytes = sum(p.stat().st_size for p in shard_dir.glob("*.protos"))
    (shard_dir / "meta.json").write_text(json.dumps({
        "shard_id": shard_id,
        "clips": len(wavs),
        "vq_files": n_npy,
        "hours": round(seconds / 3600, 4),
        "proto_bytes": proto_bytes,
        "bytes_per_audio_second": round(proto_bytes / seconds, 1) if seconds else None,
        "codec_checkpoint": str(args.codec),
        "codec_config": "modded_dac_vq",
        "source_files": source_files,
        "target_lufs": args.target_lufs,
        "trim": args.trim,
        "created": time.time(),
    }, indent=1), encoding="utf-8")

    if args.keep_vq_npy:
        npy_dir = shard_dir / "vq"
        for p in pending_dir.rglob("*.npy"):
            d = npy_dir / p.parent.name
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(d / p.name))

    shutil.rmtree(pending_dir, ignore_errors=True)
    pending_dir.mkdir(parents=True, exist_ok=True)
    log(f"  {shard_id}: {len(wavs)} clips, {seconds/3600:.2f}h, "
        f"protos {human(proto_bytes)}")
    return {"shard_id": shard_id, "clips": len(wavs),
            "hours": round(seconds / 3600, 4), "proto_bytes": proto_bytes}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="DDD-Cambodia/khmer-speech-dataset")
    ap.add_argument("--repo", required=True, help="target HF *dataset* repo id")
    ap.add_argument("--chunk-gb", type=float, default=15.0,
                    help="parquet downloaded per chunk before it is processed "
                         "and deleted (default 15)")
    ap.add_argument("--vq-batch-clips", type=int, default=20000,
                    help="clips buffered before an extract_vq+build pass; "
                         "bounds peak disk (default 20000 ~= 5GB of wav)")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="stop after N chunks (0 = until the corpus is done)")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop cleanly after this many wall-clock hours, "
                         "checked at shard boundaries so the in-flight shard "
                         "still gets published. Set below the platform's "
                         "session limit (e.g. 10.5 for Kaggle's 12h).")
    ap.add_argument("--pilot", type=int, default=0,
                    help="process only ~N clips, report measured throughput, "
                         "and DO NOT upload. Run this first.")
    ap.add_argument("--parquet-dir", default="/kaggle/temp/ddd_parquet")
    ap.add_argument("--work-dir", default="/kaggle/working/shardbuild")
    ap.add_argument("--fish-dir", default="vendor/fish-speech")
    ap.add_argument("--codec", default="checkpoints/openaudio-s1-mini/codec.pth")
    ap.add_argument("--extract-workers", type=int, default=4,
                    help="extract_vq subprocesses; round-robined across GPUs, "
                         "so 4 keeps both Kaggle T4s busy")
    ap.add_argument("--extract-batch-size", type=int, default=16)
    ap.add_argument("--build-workers", type=int, default=4)
    ap.add_argument("--trim", choices=("energy", "none"), default="energy")
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--a-grade-min-sr", type=int, default=16000)
    ap.add_argument("--min-grade", choices=("B", "C"), default="B",
                    help="worst grade to keep (default B: drop C and D)")
    ap.add_argument("--include-partial-724", action="store_true")
    ap.add_argument("--keep-vq-npy", action="store_true",
                    help="also publish raw .npy VQ tokens (~14KB/clip) so text "
                         "changes can be re-packed without re-extraction")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if not token and not args.pilot:
        raise SystemExit("No HF token found -- needed to publish. Set HF_TOKEN.")

    if not os.path.isfile(args.codec):
        raise SystemExit(f"codec not found at {args.codec} -- download "
                         "fishaudio/openaudio-s1-mini first.")
    if not os.path.isdir(os.path.join(args.fish_dir, "fish_speech")):
        raise SystemExit(f"fish-speech checkout not found at {args.fish_dir}")

    from khmer_tts.text.normalize import normalize_khmer_text
    qc = load_qc_module()

    work = Path(args.work_dir)
    pending = work / "pending"
    out_root = work / "shards"
    pq_dir = Path(args.parquet_dir)
    for d in (pending, out_root, pq_dir):
        d.mkdir(parents=True, exist_ok=True)

    repo = ShardRepo(args.repo, token, args.dataset)
    if args.pilot:
        prog = {"processed_files": [], "clips": 0, "hours": 0.0, "shards": [],
                "rejected": 0, "duplicates": 0}
        seen = set()
        log(f"PILOT: ~{args.pilot} clips, nothing will be uploaded")
    else:
        repo.ensure()
        prog = repo.load_progress()
        seen = repo.load_seen()
        log(f"resuming: {len(prog['processed_files'])} files done, "
            f"{prog['clips']:,} clips, {prog['hours']:.1f}h, "
            f"{len(prog['shards'])} shards, {len(seen):,} known keys")

    all_files = list_source_files(args.dataset, token, args.include_partial_724)
    done = set(prog["processed_files"])
    todo = [f for f in all_files if f not in done]
    log(f"{len(todo)} of {len(all_files)} source files remaining")
    if not todo:
        log("NOTHING LEFT TO DO -- the corpus is fully processed.")
        return

    stats = {"clips": 0, "seconds": 0.0, "rejected": 0, "duplicates": 0,
             "grades": {}}
    timing = {"download": 0.0, "decode": 0.0, "extract_vq": 0.0,
              "build_dataset": 0.0, "upload": 0.0}
    t_start = time.time()
    chunk_no = 0
    shard_seq = len(prog["shards"])
    chunk_files = []
    pending_count = 0
    file_iter = iter(todo)
    exhausted = False

    try:
        while not exhausted:
            if args.max_chunks and chunk_no >= args.max_chunks:
                log(f"reached --max-chunks {args.max_chunks}, stopping cleanly")
                break
            chunk_no += 1
            chunk_bytes = 0
            log(f"--- chunk {chunk_no}: reading up to {args.chunk_gb}GB of source "
                f"parquet ---")

            # Download -> process -> DELETE one file at a time. Batching the
            # downloads first would park a full chunk-gb of parquet on disk;
            # interleaving keeps the parquet footprint at a single file (~113MB)
            # no matter how large --chunk-gb is.
            while chunk_bytes < args.chunk_gb * 1e9:
                try:
                    f = next(file_iter)
                except StopIteration:
                    exhausted = True
                    break

                t0 = time.time()
                try:
                    p = fetch_parquet(args.dataset, f, str(pq_dir), token)
                except Exception as e:
                    log(f"  download failed for {f}: {type(e).__name__} -- skipping")
                    continue
                timing["download"] += time.time() - t0
                chunk_bytes += os.path.getsize(p)

                t0 = time.time()
                try:
                    n = process_parquet(p, pending, seen, qc, normalize_khmer_text,
                                        args, stats)
                except Exception as e:
                    log(f"  processing failed for {f}: {type(e).__name__}: {e} "
                        "-- skipping (it stays unprocessed for a later run)")
                    n = None
                timing["decode"] += time.time() - t0

                try:
                    os.remove(p)
                except OSError:
                    pass
                shutil.rmtree(pq_dir / ".cache", ignore_errors=True)

                if n is None:
                    continue
                pending_count += n
                chunk_files.append(f)
                log(f"  {f.split('/')[-1]}: +{n} clips | pending {pending_count} "
                    f"| kept {stats['clips']:,} | {human(chunk_bytes)} of chunk")

                hit_pilot = args.pilot and stats["clips"] >= args.pilot
                if pending_count >= args.vq_batch_clips or hit_pilot:
                    shard_seq += 1
                    sid = f"shard_{shard_seq:05d}"
                    info = flush_to_shard(pending, out_root, sid, args,
                                          chunk_files, timing)
                    if info and not args.pilot:
                        t1 = time.time()
                        repo.upload_shard(out_root / sid, sid)
                        timing["upload"] += time.time() - t1
                        prog["shards"].append(info)
                        prog["processed_files"].extend(chunk_files)
                        prog["clips"] = stats["clips"]
                        prog["hours"] = round(stats["seconds"] / 3600, 3)
                        prog["rejected"] = stats["rejected"]
                        prog["duplicates"] = stats["duplicates"]
                        repo.save_progress(prog)
                        repo.save_seen(seen)
                        shutil.rmtree(out_root / sid, ignore_errors=True)
                        log(f"  published {sid} (+progress, +seen keys)")
                    chunk_files = []
                    pending_count = 0
                if hit_pilot:
                    exhausted = True
                    break
                if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                    log(f"reached --max-hours {args.max_hours} -- stopping cleanly "
                        "with everything so far published")
                    exhausted = True
                    break

        # trailing partial batch
        if pending_count and not args.pilot:
            shard_seq += 1
            sid = f"shard_{shard_seq:05d}"
            info = flush_to_shard(pending, out_root, sid, args, chunk_files, timing)
            if info:
                repo.upload_shard(out_root / sid, sid)
                prog["shards"].append(info)
                prog["processed_files"].extend(chunk_files)
                prog["clips"] = stats["clips"]
                prog["hours"] = round(stats["seconds"] / 3600, 3)
                repo.save_progress(prog)
                repo.save_seen(seen)
                shutil.rmtree(out_root / sid, ignore_errors=True)
                log(f"  published {sid} + progress")
    finally:
        elapsed = time.time() - t_start
        hrs = stats["seconds"] / 3600
        log("=" * 68)
        log(f"clips kept {stats['clips']:,} | rejected {stats['rejected']:,} | "
            f"duplicates {stats['duplicates']:,}")
        log(f"audio processed {hrs:.2f}h in {elapsed/60:.1f} min")
        log(f"grades: {stats['grades']}")
        for k, v in timing.items():
            log(f"  {k:<14} {v/60:>7.1f} min")
        if hrs > 0 and elapsed > 0:
            rate = hrs / (elapsed / 3600)
            log(f"throughput: {rate:.1f} audio-hours per wall-clock hour")
            remaining = 1014 - (prog.get("hours", 0) if not args.pilot else 0)
            if rate > 0:
                log(f"=> ~{remaining / rate:.1f} h to finish the remaining "
                    f"~{remaining:.0f} audio-hours of the corpus")
        if not args.pilot:
            log(f"repo total: {prog['clips']:,} clips / {prog['hours']:.1f}h "
                f"across {len(prog['shards'])} shards")
            log(f"re-run this script to continue; it resumes from progress.json")


if __name__ == "__main__":
    main()
