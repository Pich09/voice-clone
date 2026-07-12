"""Stream one shard of a big HF audio dataset to disk, per session.

Each collaborator trains on a different shard. Sharding is done by a stable
hash of the speaker id, so:
  * a speaker's clips never split across people (good for TTS),
  * shards are deterministic and non-overlapping across machines,
  * nobody has to download the whole 700h+ corpus.

Only `take_n` clips are pulled per session ("a small amount each epoch").
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Optional

from .registry import bucket_for_key

# Candidate column names across DDD / generic HF ASR datasets.
_AUDIO_KEYS = ("audio", "wav", "speech")
_TEXT_KEYS = ("transcript", "text", "sentence", "transcription")
_SPEAKER_KEYS = ("speaker_id", "speaker", "spk", "client_id")


def detect_keys(example: dict) -> dict:
    """Figure out which columns hold audio / text / speaker for this dataset.
    Raises if audio or text cannot be found (so failures are loud and early)."""
    def pick(cands, required):
        for k in cands:
            if k in example:
                return k
        if required:
            raise KeyError(
                f"None of {cands} in dataset columns {list(example.keys())}. "
                "Edit the *_KEYS lists in sharding.py to match this dataset."
            )
        return None

    return {
        "audio": pick(_AUDIO_KEYS, True),
        "text": pick(_TEXT_KEYS, True),
        "speaker": pick(_SPEAKER_KEYS, False),
    }


def stream_shard_to_disk(
    dataset_id: str,
    split: str,
    num_shards: int,
    shard_index: int,
    take_n: int,
    audio_out_dir: str,
    manifest_path: str,
    token: Optional[str] = None,
    seed: int = 42,
    shuffle_buffer: int = 5000,
) -> int:
    """Stream up to `take_n` clips belonging to this shard, writing wavs +
    a JSONL manifest compatible with the rest of the pipeline (audio_path,
    text, speaker_id, duration, source). Returns the number written."""
    import soundfile as sf
    from datasets import load_dataset

    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)

    ds = load_dataset(dataset_id, split=split, streaming=True, token=token)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    keys = None
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, row in enumerate(ds):
            if keys is None:
                keys = detect_keys(row)

            speaker = str(row.get(keys["speaker"], "unknown")) if keys["speaker"] else "unknown"
            # Assign whole speakers to shards; skip clips not in ours.
            if bucket_for_key(speaker, num_shards) != shard_index:
                continue

            text = (row.get(keys["text"]) or "").strip()
            audio = row.get(keys["audio"])
            if not text or audio is None:
                continue

            array = audio["array"]
            sr = audio["sampling_rate"]
            fname = f"{speaker}_{shard_index}_{written:07d}.wav"
            out_path = os.path.join(audio_out_dir, fname)
            sf.write(out_path, array, sr)

            mf.write(json.dumps({
                "audio_path": out_path,
                "text": text,
                "speaker_id": speaker,
                "duration": round(len(array) / sr, 3),
                "source": f"{dataset_id}#shard{shard_index}",
            }, ensure_ascii=False) + "\n")

            written += 1
            if written >= take_n:
                break

    return written


def read_val_loss(output_dir: str) -> Optional[float]:
    """Best-effort: scan a Lightning/Fish training output dir for the lowest
    recorded validation loss. Returns None if nothing parseable is found, in
    which case the relay falls back to ordering checkpoints by step."""
    candidates = []
    for pattern in ("**/metrics.csv", "**/*metrics*.csv"):
        candidates += glob.glob(os.path.join(output_dir, pattern), recursive=True)

    best: Optional[float] = None
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loss_col = None
                for row in reader:
                    if loss_col is None:
                        for c in row:
                            if c and "val" in c.lower() and "loss" in c.lower():
                                loss_col = c
                                break
                        if loss_col is None:
                            break
                    val = row.get(loss_col, "")
                    if val not in ("", None):
                        try:
                            v = float(val)
                            best = v if best is None else min(best, v)
                        except ValueError:
                            pass
        except Exception:
            continue
    return best
