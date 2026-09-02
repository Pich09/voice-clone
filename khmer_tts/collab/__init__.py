"""Collaborative checkpoint-relay training helpers.

Lets several people train one model on different data shards using free
Kaggle GPUs, passing a shared checkpoint through a Hugging Face repo:

    pull best checkpoint -> train on your shard -> push new checkpoint

See `hf_relay.HFCheckpointRelay` and `sharding.stream_shard_to_disk`.
"""

from .registry import CheckpointEntry, select_best, select_latest, bucket_for_key
from .hf_relay import HFCheckpointRelay
from .checkpoint_store import CheckpointStore
from .sharding import stream_shard_to_disk, detect_keys, read_val_loss
from .data_cache import (
    cache_key, download_and_restore, pack_and_upload, exists,
    get_shard_count, download_and_restore_shard, download_and_restore_val,
    pack_and_upload_sharded,
    get_ready_shards, mark_shard_ready, all_shards_ready,
    pack_and_upload_ready_shard, download_and_restore_ready_shard,
    is_val_ready, pack_and_upload_ready_val, download_and_restore_ready_val,
    get_next_train_shard_index, advance_train_shard_cursor,
)

__all__ = [
    "CheckpointEntry",
    "select_best",
    "select_latest",
    "bucket_for_key",
    "HFCheckpointRelay",
    "CheckpointStore",
    "stream_shard_to_disk",
    "detect_keys",
    "read_val_loss",
    "cache_key",
    "download_and_restore",
    "pack_and_upload",
    "exists",
    "get_shard_count",
    "download_and_restore_shard",
    "download_and_restore_val",
    "pack_and_upload_sharded",
    "get_ready_shards",
    "mark_shard_ready",
    "all_shards_ready",
    "pack_and_upload_ready_shard",
    "download_and_restore_ready_shard",
    "is_val_ready",
    "pack_and_upload_ready_val",
    "download_and_restore_ready_val",
    "get_next_train_shard_index",
    "advance_train_shard_cursor",
]
