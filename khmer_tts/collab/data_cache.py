"""Cache the PROCESSED training data in a Hugging Face *dataset* repo.

The 01-09 pipeline (download -> QC -> denoise -> VAD -> loudness -> normalize
-> split -> fish-format convert) is the slowest part of a session. Its output
-- the fish-format wav/lab folders plus the train/valid manifests -- fully
determines what training consumes, so once produced it can be tarred, pushed
to an HF dataset repo, and pulled back verbatim next time instead of rerun.

The cache is keyed on what defines the processed dataset: dataset id, split,
sample count, shard params, and a manual revision number -- NOT the per-session
stream seed. So the FIRST run for a given config builds the processed data and
uploads it, and EVERY later run finds it and skips preprocessing entirely.
Bump `rev` (or set FORCE_REPROCESS_DATA in the notebook) to rebuild.

Uses repo_type="dataset" (a *data* repo), distinct from the model repo the
checkpoint relay uses.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from typing import Optional, Sequence

# What the pipeline produces and training consumes. Missing entries are skipped
# (e.g. a tiny run whose split yielded no validation set).
DEFAULT_PATHS = (
    "data/fish/khmer_base",
    "data/fish/khmer_base_val",
    "data/manifests/ddd_train.jsonl",
    "data/manifests/ddd_valid.jsonl",
)


def cache_key(*, dataset_id: str, split: str, sample_count, num_shards: int,
              shard_index: int, rev: int) -> str:
    """Stable short key identifying one processed-data configuration.

    Deliberately independent of the per-session stream seed: once the processed
    data for a (dataset, split, sample_count, shard, rev) config exists in the
    cache, every later run reuses it instead of re-streaming a different slice.
    """
    raw = (f"{dataset_id}|{split}|n={sample_count}|shards={num_shards}"
           f"|idx={shard_index}|rev={rev}")
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _archive_name(key: str) -> str:
    return f"processed/khmer_base__{key}.tar"


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract, refusing any member that would escape `dest` (path traversal)."""
    dest_abs = os.path.abspath(dest)
    for m in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, m.name))
        if target != dest_abs and not target.startswith(dest_abs + os.sep):
            raise ValueError(f"unsafe path in cache archive: {m.name!r}")
    tar.extractall(dest)


def exists(repo_id: str, key: str, token: Optional[str] = None) -> bool:
    """True if a cache archive for `key` is already in the dataset repo."""
    from huggingface_hub import HfApi
    try:
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return False
    return _archive_name(key) in files


def download_and_restore(repo_id: str, key: str, workdir: str,
                         token: Optional[str] = None) -> bool:
    """Download the cache archive for `key` and extract it into `workdir`.
    Returns True on a hit, False if there is no such cache (or on any error --
    a missing/unreachable cache must degrade to "just preprocess", never crash
    the run)."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _archive_name(key), repo_type="dataset",
                             token=token)
    except Exception:
        return False
    try:
        with tarfile.open(fp, "r") as tar:
            _safe_extract(tar, workdir)
    except Exception:
        return False
    return True


def pack_and_upload(repo_id: str, key: str, workdir: str,
                    paths: Sequence[str] = DEFAULT_PATHS,
                    token: Optional[str] = None, private: bool = True) -> str:
    """Tar the given paths (relative to `workdir`) and upload as the cache
    archive for `key`. Creates the dataset repo if needed. Returns the archive
    path in the repo."""
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", token=token, private=private,
                exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp.close()
    try:
        n = 0
        with tarfile.open(tmp.name, "w") as tar:
            for p in paths:
                full = os.path.join(workdir, p)
                if os.path.exists(full):
                    tar.add(full, arcname=p)
                    n += 1
        if n == 0:
            raise ValueError(f"none of {list(paths)} exist under {workdir!r}")
        upload_file(path_or_fileobj=tmp.name, path_in_repo=_archive_name(key),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"processed-data cache {key}")
    finally:
        os.remove(tmp.name)
    return _archive_name(key)
