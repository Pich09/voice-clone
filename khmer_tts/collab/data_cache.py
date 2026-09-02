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
import json
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
    finally:
        # `fp` is a real file (or the real blob a symlink points at) sitting
        # in the huggingface_hub cache -- left alone, it doubles peak disk
        # usage for the rest of the session (the ~22GB archive PLUS its
        # ~22GB extracted contents, both live at once). Kaggle/Colab's local
        # disk is nowhere near big enough to carry that alongside the base
        # checkpoint, VQ tokens, protobuf shards, and training output too.
        # Safe to delete once extraction succeeds (or even on failure --
        # hf_hub_download just re-fetches it next time): the archive itself
        # is never read again after this point.
        try:
            os.remove(os.path.realpath(fp))
        except OSError:
            pass
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


# ---------------------------------------------------------------------------
# Sharded variant -- for platforms whose local disk can't hold the whole
# processed dataset at once (Kaggle's /kaggle/working is hard-capped at 20GB,
# smaller than a single ~22GB DEFAULT_PATHS archive). Splits the TRAIN audio
# only into N roughly-equal shards uploaded as separate archives, so a
# session only ever needs one shard's worth of audio on disk; the (much
# smaller) validation set stays a single unsharded archive since ~400MB is
# cheap to pull in full every session. Building these shards is a one-off,
# local, CPU-only operation -- see scripts/reshard_processed_data.py -- done
# once per (key), not something the training notebook itself does.
# ---------------------------------------------------------------------------

def _shard_archive_name(key: str, shard_idx: int, n_shards: int) -> str:
    return f"processed/khmer_base__{key}__shard{shard_idx}of{n_shards}.tar"


def _val_archive_name(key: str) -> str:
    return f"processed/khmer_base__{key}__val.tar"


def _shards_manifest_name(key: str) -> str:
    return f"processed/khmer_base__{key}__shards.json"


def get_shard_count(repo_id: str, key: str, token: Optional[str] = None) -> Optional[int]:
    """Number of train shards uploaded for `key`, or None if not sharded
    (caller should fall back to the unsharded download_and_restore)."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _shards_manifest_name(key), repo_type="dataset",
                             token=token, force_download=True)
        with open(fp, encoding="utf-8") as f:
            return int(json.load(f)["n_shards"])
    except Exception:
        return None


def download_and_restore_shard(repo_id: str, key: str, shard_idx: int, n_shards: int,
                               workdir: str, token: Optional[str] = None) -> bool:
    """Like download_and_restore, but for one train-audio shard only."""
    from huggingface_hub import hf_hub_download
    name = _shard_archive_name(key, shard_idx, n_shards)
    try:
        fp = hf_hub_download(repo_id, name, repo_type="dataset", token=token)
    except Exception:
        return False
    try:
        with tarfile.open(fp, "r") as tar:
            _safe_extract(tar, workdir)
    except Exception:
        return False
    finally:
        try:
            os.remove(os.path.realpath(fp))
        except OSError:
            pass
    return True


def download_and_restore_val(repo_id: str, key: str, workdir: str,
                             token: Optional[str] = None) -> bool:
    """Restore the (unsharded) validation set for `key`."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _val_archive_name(key), repo_type="dataset",
                             token=token)
    except Exception:
        return False
    try:
        with tarfile.open(fp, "r") as tar:
            _safe_extract(tar, workdir)
    except Exception:
        return False
    finally:
        try:
            os.remove(os.path.realpath(fp))
        except OSError:
            pass
    return True


def pack_and_upload_sharded(repo_id: str, key: str, workdir: str,
                            train_dir: str = "data/fish/khmer_base",
                            val_dir: str = "data/fish/khmer_base_val",
                            n_shards: int = 6, token: Optional[str] = None,
                            private: bool = False) -> None:
    """One-off local tool: split the TRAIN audio under `train_dir` into
    `n_shards` archives (contiguous blocks of wav/lab pairs, in filename-sorted
    order so re-running this is deterministic) and upload each, plus a single
    unsharded archive for `val_dir` and a small shards.json manifest. Resets
    the round-robin cursor to 0.

    Run this once per dataset revision from wherever the full processed data
    already lives (see scripts/reshard_processed_data.py) -- the training
    notebook only ever CONSUMES these shards, it never builds them."""
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", token=token, private=private,
                exist_ok=True)

    full_train_dir = os.path.join(workdir, train_dir)
    wavs = sorted(
        os.path.relpath(os.path.join(dp, fn), workdir)
        for dp, _, fns in os.walk(full_train_dir)
        for fn in fns if fn.endswith(".wav")
    )
    if not wavs:
        raise ValueError(f"no .wav files found under {full_train_dir!r}")

    # Each wav's .lab sibling travels in the same shard as its wav.
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for i, wav_rel in enumerate(wavs):
        group = shards[i % n_shards]
        group.append(wav_rel)
        lab_rel = wav_rel[:-4] + ".lab"
        if os.path.exists(os.path.join(workdir, lab_rel)):
            group.append(lab_rel)

    for i, members in enumerate(shards):
        tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
        tmp.close()
        try:
            with tarfile.open(tmp.name, "w") as tar:
                for rel in members:
                    tar.add(os.path.join(workdir, rel), arcname=rel)
            name = _shard_archive_name(key, i, n_shards)
            upload_file(path_or_fileobj=tmp.name, path_in_repo=name,
                        repo_id=repo_id, repo_type="dataset", token=token,
                        commit_message=f"processed-data shard {i}/{n_shards} for {key}")
            print(f"  shard {i}/{n_shards}: {len(members)} file(s) -> {name}")
        finally:
            os.remove(tmp.name)

    full_val_dir = os.path.join(workdir, val_dir)
    if os.path.isdir(full_val_dir) and os.listdir(full_val_dir):
        tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
        tmp.close()
        try:
            with tarfile.open(tmp.name, "w") as tar:
                tar.add(full_val_dir, arcname=val_dir)
            upload_file(path_or_fileobj=tmp.name, path_in_repo=_val_archive_name(key),
                        repo_id=repo_id, repo_type="dataset", token=token,
                        commit_message=f"processed-data val set for {key}")
            print(f"  val set -> {_val_archive_name(key)}")
        finally:
            os.remove(tmp.name)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8")
    try:
        json.dump({"n_shards": n_shards, "key": key,
                   "shard_sizes": [len(s) for s in shards]}, tmp)
        tmp.close()
        upload_file(path_or_fileobj=tmp.name, path_in_repo=_shards_manifest_name(key),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"shards manifest for {key} ({n_shards} shards)")
    finally:
        os.remove(tmp.name)

    # Fresh shard set -- clear the readiness manifest so preprocessing starts
    # every shard over (a previous key's finished shards mean nothing here).
    _write_ready_manifest(repo_id, key, set(), token=token)


# ---------------------------------------------------------------------------
# VQ-preprocessing readiness -- kaggle/preprocess_khmer_vq.ipynb works through
# the shards above one at a time (download raw audio -> extract VQ tokens ->
# build protobuf shards -> upload the RESULT -> delete local raw audio -> next
# shard), so kaggle/train_khmer_base.ipynb never has to touch raw audio on
# Kaggle at all -- it just waits until every shard is marked ready, then pulls
# the (much smaller) packed protobufs for all of them and trains directly.
# ---------------------------------------------------------------------------

def _ready_manifest_name(key: str) -> str:
    return f"processed/khmer_base__{key}__ready.json"


def _ready_shard_archive_name(key: str, shard_idx: int, n_shards: int) -> str:
    return f"processed/khmer_base__{key}__ready_shard{shard_idx}of{n_shards}.tar"


def _ready_val_archive_name(key: str) -> str:
    return f"processed/khmer_base__{key}__ready_val.tar"


def get_ready_shards(repo_id: str, key: str, token: Optional[str] = None) -> set:
    """Set of shard indices whose protobuf output has already been uploaded."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _ready_manifest_name(key), repo_type="dataset",
                             token=token, force_download=True)
        with open(fp, encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    except Exception:
        return set()


def _write_ready_manifest(repo_id: str, key: str, done: set,
                          token: Optional[str] = None) -> None:
    from huggingface_hub import upload_file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8")
    try:
        json.dump({"done": sorted(done)}, tmp)
        tmp.close()
        upload_file(path_or_fileobj=tmp.name, path_in_repo=_ready_manifest_name(key),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"VQ-ready shards for {key}: {sorted(done)}")
    finally:
        os.remove(tmp.name)


def mark_shard_ready(repo_id: str, key: str, shard_idx: int,
                     token: Optional[str] = None) -> None:
    done = get_ready_shards(repo_id, key, token=token)
    done.add(shard_idx)
    _write_ready_manifest(repo_id, key, done, token=token)


def all_shards_ready(repo_id: str, key: str, n_shards: int,
                     token: Optional[str] = None) -> bool:
    done = get_ready_shards(repo_id, key, token=token)
    return len(done) >= n_shards and all(i in done for i in range(n_shards))


def pack_and_upload_ready_shard(repo_id: str, key: str, shard_idx: int, n_shards: int,
                                workdir: str, proto_dir: str,
                                token: Optional[str] = None,
                                private: bool = False) -> None:
    """Upload one shard's already-built protobuf output (proto_dir, relative
    to workdir) and mark that shard ready. Best-effort caller responsibility
    to have actually run scripts/10a_extract_vq_and_build_protos.sh first."""
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", token=token, private=private,
                exist_ok=True)

    full = os.path.join(workdir, proto_dir)
    if not os.path.isdir(full) or not os.listdir(full):
        raise ValueError(f"nothing to upload: {full!r} is empty or missing")

    tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp.close()
    try:
        with tarfile.open(tmp.name, "w") as tar:
            tar.add(full, arcname=proto_dir)
        upload_file(path_or_fileobj=tmp.name,
                    path_in_repo=_ready_shard_archive_name(key, shard_idx, n_shards),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"VQ-ready protos for shard {shard_idx}/{n_shards} ({key})")
    finally:
        os.remove(tmp.name)
    mark_shard_ready(repo_id, key, shard_idx, token=token)


def download_and_restore_ready_shard(repo_id: str, key: str, shard_idx: int, n_shards: int,
                                     workdir: str, token: Optional[str] = None) -> bool:
    """Restore one shard's already-built protobuf output into workdir."""
    from huggingface_hub import hf_hub_download
    name = _ready_shard_archive_name(key, shard_idx, n_shards)
    try:
        fp = hf_hub_download(repo_id, name, repo_type="dataset", token=token)
    except Exception:
        return False
    try:
        with tarfile.open(fp, "r") as tar:
            _safe_extract(tar, workdir)
    except Exception:
        return False
    finally:
        try:
            os.remove(os.path.realpath(fp))
        except OSError:
            pass
    return True


def is_val_ready(repo_id: str, key: str, token: Optional[str] = None) -> bool:
    from huggingface_hub import HfApi
    try:
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return False
    return _ready_val_archive_name(key) in files


def pack_and_upload_ready_val(repo_id: str, key: str, workdir: str, val_proto_dir: str,
                              token: Optional[str] = None, private: bool = False) -> None:
    """Upload the (unsharded) validation set's already-built protobuf output."""
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", token=token, private=private,
                exist_ok=True)

    full = os.path.join(workdir, val_proto_dir)
    if not os.path.isdir(full) or not os.listdir(full):
        raise ValueError(f"nothing to upload: {full!r} is empty or missing")

    tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp.close()
    try:
        with tarfile.open(tmp.name, "w") as tar:
            tar.add(full, arcname=val_proto_dir)
        upload_file(path_or_fileobj=tmp.name, path_in_repo=_ready_val_archive_name(key),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"VQ-ready val protos for {key}")
    finally:
        os.remove(tmp.name)


def download_and_restore_ready_val(repo_id: str, key: str, workdir: str,
                                   token: Optional[str] = None) -> bool:
    """Restore the validation set's already-built protobuf output."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _ready_val_archive_name(key), repo_type="dataset",
                             token=token)
    except Exception:
        return False
    try:
        with tarfile.open(fp, "r") as tar:
            _safe_extract(tar, workdir)
    except Exception:
        return False
    finally:
        try:
            os.remove(os.path.realpath(fp))
        except OSError:
            pass
    return True


# ---------------------------------------------------------------------------
# Training-time shard rotation -- kaggle/train_khmer_base.ipynb trains on one
# VQ-ready shard per session (not all combined), then rotates to the next
# shard next session, freeing local disk between sessions. This cursor is
# separate from the readiness tracking above: readiness says which shards
# PREPROCESSING has finished; this says which one TRAINING should use next.
# ---------------------------------------------------------------------------

def _train_cursor_name(key: str) -> str:
    return f"processed/khmer_base__{key}__train_cursor.json"


def get_next_train_shard_index(repo_id: str, key: str, n_shards: int,
                               token: Optional[str] = None) -> Optional[int]:
    """The shard training should use this session: the next index after
    wherever the cursor last left off that is ALREADY VQ-ready, wrapping
    around, so training can proceed on whatever preprocessing has finished so
    far rather than blocking on every shard being done. None if nothing is
    ready yet at all."""
    from huggingface_hub import hf_hub_download
    try:
        fp = hf_hub_download(repo_id, _train_cursor_name(key), repo_type="dataset",
                             token=token, force_download=True)
        with open(fp, encoding="utf-8") as f:
            start = int(json.load(f).get("next_index", 0)) % n_shards
    except Exception:
        start = 0
    done = get_ready_shards(repo_id, key, token=token)
    if not done:
        return None
    for offset in range(n_shards):
        idx = (start + offset) % n_shards
        if idx in done:
            return idx
    return None


def advance_train_shard_cursor(repo_id: str, key: str, n_shards: int, used_index: int,
                               token: Optional[str] = None) -> None:
    """Move the training cursor past `used_index` so next session picks up
    the following shard. Called as soon as a shard is claimed for this
    session (not gated on that session's success) so a failed session still
    rotates forward instead of retrying the same shard forever."""
    from huggingface_hub import upload_file
    nxt = (used_index + 1) % n_shards
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8")
    try:
        json.dump({"next_index": nxt}, tmp)
        tmp.close()
        upload_file(path_or_fileobj=tmp.name, path_in_repo=_train_cursor_name(key),
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message=f"train shard cursor -> {nxt}")
    finally:
        os.remove(tmp.name)
