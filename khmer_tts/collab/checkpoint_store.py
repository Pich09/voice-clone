"""Minimal single-lineage checkpoint store: keeps only `latest/` and `best/`
(each overwritten in place on every publish) plus a small metadata.json,
instead of the full per-session history `HFCheckpointRelay` keeps.

Use this for a solo "resume where I left off across my own sessions" flow
where the full history isn't needed and minimizing storage matters (see
HFCheckpointRelay's docstring for the multi-person collaborative case that
DOES want history -- that one is unaffected by this module).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

METADATA_PATH = "metadata.json"
LOCK_PATH = "lock.json"


class CheckpointStore:
    def __init__(self, repo_id: str, token: Optional[str] = None,
                 cache_dir: str = "hf_checkpoint_cache"):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")
        self.api = HfApi(token=self.token)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def ensure_repo(self, private: bool = True) -> None:
        from huggingface_hub import create_repo
        create_repo(self.repo_id, token=self.token, repo_type="model",
                    private=private, exist_ok=True)

    # ---- metadata ---------------------------------------------------
    def load_metadata(self) -> dict:
        from huggingface_hub import hf_hub_download
        try:
            fp = hf_hub_download(self.repo_id, METADATA_PATH, repo_type="model",
                                 token=self.token, cache_dir=self.cache_dir,
                                 force_download=True)
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _upload_json(self, obj: dict, path_in_repo: str) -> None:
        from huggingface_hub import upload_file
        tmp = os.path.join(self.cache_dir, os.path.basename(path_in_repo))
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo,
                    repo_id=self.repo_id, repo_type="model", token=self.token)

    # ---- pulling ------------------------------------------------------
    def pull_latest(self, dest_dir: str) -> bool:
        """Download `latest/` into dest_dir. Returns False on the first-ever
        run (nothing published yet) -- caller should fall back to the base
        pretrained checkpoint in that case."""
        return self._pull("latest", dest_dir)

    def pull_best(self, dest_dir: str) -> bool:
        return self._pull("best", dest_dir)

    def _pull(self, subfolder: str, dest_dir: str) -> bool:
        from huggingface_hub import snapshot_download
        try:
            snap = snapshot_download(self.repo_id, repo_type="model", token=self.token,
                                     cache_dir=self.cache_dir,
                                     allow_patterns=[f"{subfolder}/**"])
        except Exception:
            return False
        src = os.path.join(snap, subfolder)
        if not os.path.isdir(src) or not os.listdir(src):
            return False
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src, dest_dir)
        return True

    # ---- publishing -----------------------------------------------------
    def publish(self, local_ckpt_dir: str, step: int,
                val_loss: Optional[float], trainer_id: str) -> dict:
        """Overwrite `latest/` with this checkpoint; also overwrite `best/`
        if this run's val_loss beats the recorded best (or none is recorded
        yet). Returns the updated metadata dict."""
        from huggingface_hub import upload_folder

        if not os.path.isdir(local_ckpt_dir) or not os.listdir(local_ckpt_dir):
            raise ValueError(f"Nothing to publish at {local_ckpt_dir!r}")

        meta = self.load_metadata()
        now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        upload_folder(repo_id=self.repo_id, repo_type="model", token=self.token,
                      folder_path=local_ckpt_dir, path_in_repo="latest",
                      commit_message=f"{trainer_id}: latest @ step {step} val_loss={val_loss}")
        meta["latest_step"] = int(step)
        meta["latest_val_loss"] = val_loss
        meta["latest_updated"] = now
        meta["sessions"] = int(meta.get("sessions", 0)) + 1

        prev_best_vl = meta.get("best_val_loss")
        is_better = val_loss is not None and (prev_best_vl is None or val_loss < prev_best_vl)
        if is_better or "best_step" not in meta:
            upload_folder(repo_id=self.repo_id, repo_type="model", token=self.token,
                          folder_path=local_ckpt_dir, path_in_repo="best",
                          commit_message=f"{trainer_id}: best @ step {step} val_loss={val_loss}")
            meta["best_step"] = int(step)
            meta["best_val_loss"] = val_loss
            meta["best_updated"] = now

        self._upload_json(meta, METADATA_PATH)
        return meta

    # ---- advisory lock (same pattern as HFCheckpointRelay) --------------
    def acquire_lock(self, trainer_id: str, ttl_sec: int = 3600):
        cur = self.load_metadata().get("_lock")
        now = time.time()
        if cur and cur.get("owner") not in (None, trainer_id) \
                and cur.get("expires", 0) > now:
            return False, cur
        meta = self.load_metadata()
        meta["_lock"] = {"owner": trainer_id, "acquired": now, "expires": now + ttl_sec}
        self._upload_json(meta, METADATA_PATH)
        return True, meta["_lock"]

    def release_lock(self, trainer_id: str) -> None:
        meta = self.load_metadata()
        if meta.get("_lock", {}).get("owner") == trainer_id:
            meta["_lock"] = {"owner": None, "acquired": 0, "expires": 0}
            self._upload_json(meta, METADATA_PATH)
