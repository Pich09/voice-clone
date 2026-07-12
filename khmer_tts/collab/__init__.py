"""Collaborative checkpoint-relay training helpers.

Lets several people train one model on different data shards using free
Kaggle GPUs, passing a shared checkpoint through a Hugging Face repo:

    pull best checkpoint -> train on your shard -> push new checkpoint

See `hf_relay.HFCheckpointRelay` and `sharding.stream_shard_to_disk`.
"""

from .registry import CheckpointEntry, select_best, select_latest, bucket_for_key
from .hf_relay import HFCheckpointRelay
from .sharding import stream_shard_to_disk, detect_keys, read_val_loss

__all__ = [
    "CheckpointEntry",
    "select_best",
    "select_latest",
    "bucket_for_key",
    "HFCheckpointRelay",
    "stream_shard_to_disk",
    "detect_keys",
    "read_val_loss",
]
