"""Pure, network-free logic for the checkpoint registry.

The registry is a small JSON blob stored in the shared HF repo that lists
every checkpoint anyone has pushed, so "best" and "latest" are always
well-defined regardless of who trained when.

Kept free of any Hugging Face / IO dependency so it can be unit-tested
directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CheckpointEntry:
    step: int                 # cumulative training step this checkpoint reached
    val_loss: Optional[float] # validation loss, or None if it could not be read
    trainer_id: str           # who produced it (e.g. "friendA")
    shard_index: int          # which data shard they trained on
    path: str                 # repo subfolder, e.g. "checkpoints/step_0004000_friendA"
    created: float            # unix timestamp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointEntry":
        return cls(
            step=int(d["step"]),
            val_loss=(None if d.get("val_loss") is None else float(d["val_loss"])),
            trainer_id=str(d.get("trainer_id", "unknown")),
            shard_index=int(d.get("shard_index", 0)),
            path=str(d["path"]),
            created=float(d.get("created", 0.0)),
        )


def _entry(e) -> dict:
    return e.to_dict() if isinstance(e, CheckpointEntry) else dict(e)


def select_best(entries) -> Optional[dict]:
    """Pick the best checkpoint.

    Preference order:
      1. Lowest validation loss (checkpoints that recorded one always win
         over those that did not).
      2. Among checkpoints with no val_loss, the highest step (most trained).
    Returns the raw dict, or None if there are no entries.
    """
    items = [_entry(e) for e in entries]
    if not items:
        return None

    def key(e: dict):
        vl = e.get("val_loss")
        if vl is not None:
            return (0, float(vl), -int(e.get("step", 0)))
        return (1, 0.0, -int(e.get("step", 0)))

    return sorted(items, key=key)[0]


def select_latest(entries) -> Optional[dict]:
    """Pick the checkpoint with the highest cumulative step."""
    items = [_entry(e) for e in entries]
    if not items:
        return None
    return max(items, key=lambda e: int(e.get("step", 0)))


def bucket_for_key(key: str, num_shards: int) -> int:
    """Deterministically assign a stable key (e.g. a speaker_id) to one of
    `num_shards` buckets. Uses md5 so it is stable across processes/machines
    (Python's builtin hash() is salted per-process and must not be used)."""
    if num_shards <= 1:
        return 0
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards
