"""Hugging Face-backed checkpoint relay.

Flow for one training session:

    relay = HFCheckpointRelay("you/khmer-tts-relay", token=HF_TOKEN)
    relay.ensure_repo()
    resume_dir = relay.pull_best("checkpoints/_resume")   # None on the very first run
    ...  # train from `resume_dir` (or the base checkpoint if None)
    relay.publish(new_ckpt_dir, step=8000, val_loss=2.13,
                  trainer_id="friendA", shard_index=0)

A tiny JSON registry + lock file live in the repo so several people can
take turns safely. The lock is advisory (HF has no atomic compare-and-set),
so still coordinate who trains when for anything important.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

from .registry import CheckpointEntry, select_best, select_latest

REGISTRY_PATH = "registry.json"
LOCK_PATH = "lock.json"


class HFCheckpointRelay:
    def __init__(self, repo_id: str, token: Optional[str] = None,
                 cache_dir: str = "hf_relay_cache", private: bool = True):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")
        self.private = private
        self.api = HfApi(token=self.token)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ---- repo bootstrap -------------------------------------------------
    def ensure_repo(self) -> None:
        from huggingface_hub import create_repo
        create_repo(self.repo_id, token=self.token, repo_type="model",
                    private=self.private, exist_ok=True)

    # ---- small-file helpers --------------------------------------------
    def _download_json(self, path_in_repo: str) -> Optional[dict]:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            fp = hf_hub_download(
                self.repo_id, path_in_repo, repo_type="model",
                token=self.token, cache_dir=self.cache_dir, force_download=True,
            )
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except EntryNotFoundError:
            return None
        except Exception:
            # Missing file / transient error -> treat as absent.
            return None

    def _upload_json(self, obj: dict, path_in_repo: str) -> None:
        from huggingface_hub import upload_file
        tmp = os.path.join(self.cache_dir, os.path.basename(path_in_repo))
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo,
                    repo_id=self.repo_id, repo_type="model", token=self.token)

    # ---- registry -------------------------------------------------------
    def load_registry(self) -> dict:
        return self._download_json(REGISTRY_PATH) or {"checkpoints": []}

    def best_checkpoint(self) -> Optional[dict]:
        return select_best(self.load_registry().get("checkpoints", []))

    def latest_checkpoint(self) -> Optional[dict]:
        return select_latest(self.load_registry().get("checkpoints", []))

    # ---- pulling a checkpoint ------------------------------------------
    def pull_best(self, dest_dir: str) -> Optional[str]:
        """Download the best checkpoint into `dest_dir`. Returns the local
        path, or None if the repo has no checkpoints yet (first run)."""
        entry = self.best_checkpoint()
        if entry is None:
            return None
        return self._download_checkpoint(entry, dest_dir)

    def _download_checkpoint(self, entry: dict, dest_dir: str) -> str:
        from huggingface_hub import snapshot_download
        repo_subdir = entry["path"].rstrip("/")
        snap = snapshot_download(
            self.repo_id, repo_type="model", token=self.token,
            cache_dir=self.cache_dir, allow_patterns=[f"{repo_subdir}/**"],
        )
        src = os.path.join(snap, repo_subdir)
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src, dest_dir)
        return dest_dir

    # ---- publishing a checkpoint ---------------------------------------
    def publish(self, local_ckpt_dir: str, step: int,
                val_loss: Optional[float], trainer_id: str,
                shard_index: int = 0) -> dict:
        """Upload a checkpoint folder and record it in the registry."""
        from huggingface_hub import upload_folder

        if not os.path.isdir(local_ckpt_dir) or not os.listdir(local_ckpt_dir):
            raise ValueError(f"Nothing to publish at {local_ckpt_dir!r}")

        subfolder = f"checkpoints/step_{int(step):07d}_{trainer_id}"
        upload_folder(repo_id=self.repo_id, repo_type="model", token=self.token,
                      folder_path=local_ckpt_dir, path_in_repo=subfolder,
                      commit_message=f"{trainer_id}: step {step} val_loss={val_loss}")

        entry = CheckpointEntry(
            step=int(step), val_loss=val_loss, trainer_id=trainer_id,
            shard_index=int(shard_index), path=subfolder, created=time.time(),
        )
        reg = self.load_registry()
        reg.setdefault("checkpoints", []).append(entry.to_dict())
        self._upload_json(reg, REGISTRY_PATH)
        return entry.to_dict()

    # ---- advisory lock --------------------------------------------------
    def acquire_lock(self, trainer_id: str, ttl_sec: int = 3600):
        """Best-effort lock. Returns (ok, current_lock). Not atomic — treat
        as a courtesy guard against two people training at the same time."""
        cur = self._download_json(LOCK_PATH)
        now = time.time()
        if cur and cur.get("owner") not in (None, trainer_id) \
                and cur.get("expires", 0) > now:
            return False, cur
        lock = {"owner": trainer_id, "acquired": now, "expires": now + ttl_sec}
        self._upload_json(lock, LOCK_PATH)
        return True, lock

    def release_lock(self, trainer_id: str) -> None:
        cur = self._download_json(LOCK_PATH)
        if cur and cur.get("owner") == trainer_id:
            self._upload_json({"owner": None, "acquired": 0, "expires": 0}, LOCK_PATH)
