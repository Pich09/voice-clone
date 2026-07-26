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

from .registry import assign_shards, bucket_for_key

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
    known_speakers: Optional[list] = None,
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

    # Balanced assignment when the speaker set is known (see assign_shards);
    # otherwise fall back to independent hashing, which risks empty shards.
    shard_map = None
    if known_speakers and num_shards > 1:
        shard_map = assign_shards(known_speakers, num_shards)
        mine = sorted(s for s, b in shard_map.items() if b == shard_index)
        if not mine:
            raise ValueError(
                f"shard_index={shard_index} of {num_shards} would get no "
                f"speakers from {len(set(known_speakers))} known speakers -- "
                "use fewer shards."
            )
        print(f"shard {shard_index}/{num_shards} covers {len(mine)} speaker(s): {mine}")

    def shard_of(spk):
        if shard_map is not None:
            # An unlisted speaker (dataset grew) still needs a home.
            return shard_map.get(spk, bucket_for_key(spk, num_shards))
        return bucket_for_key(spk, num_shards)

    keys = None
    written = 0
    # Defensive de-duplication. This repo's HF dataset carries several
    # overlapping parquet shard-numbering generations (e.g.
    # data/train-00000-of-00653.parquet alongside data/train-v39-00687.parquet)
    # that the dataset's own "data/train-*" glob all matches, so the same
    # utterance can be yielded more than once. Training twice on one clip
    # wastes steps and nudges the model toward memorizing it. Keyed on
    # (speaker, sentence_id) when available, else (speaker, text).
    seen = set()
    n_dup = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, row in enumerate(ds):
            if keys is None:
                keys = detect_keys(row)

            speaker = str(row.get(keys["speaker"], "unknown")) if keys["speaker"] else "unknown"
            # Assign whole speakers to shards; skip clips not in ours.
            if shard_of(speaker) != shard_index:
                continue

            text = (row.get(keys["text"]) or "").strip()
            audio = row.get(keys["audio"])
            if not text or audio is None:
                continue

            dedup_key = (speaker, row.get("sentence_id") or text)
            if dedup_key in seen:
                n_dup += 1
                continue
            seen.add(dedup_key)

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

    if n_dup:
        print(f"skipped {n_dup} duplicate clip(s) already streamed this session")
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
