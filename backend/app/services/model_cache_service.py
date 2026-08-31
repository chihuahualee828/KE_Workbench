from __future__ import annotations

import os
import json
import signal
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings
from app.services.model_registry import model_registry
from app.storage.local_store import local_store

MODEL_DOWNLOAD_IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
]

ACTIVE_DOWNLOAD_STATUSES = {"queued", "downloading", "canceling"}
HUB_SIZE_CACHE_TTL_SECONDS = 6 * 60 * 60

SNAPSHOT_DOWNLOAD_SCRIPT = """
import json
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["KEW_HF_REPO_ID"],
    ignore_patterns=json.loads(os.environ["KEW_HF_IGNORE_PATTERNS"]),
    token=os.getenv("HF_TOKEN") or None,
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelCacheService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._runtime_hf_token: str | None = None
        self._hub_sizes = _read_hub_size_cache()
        self._lock = threading.Lock()

    def list_cache_statuses(self) -> list[dict[str, Any]]:
        return [self.cache_status(model["id"]) for model in model_registry.list_models()]

    def hf_token_status(self) -> dict[str, Any]:
        with self._lock:
            has_runtime_token = bool(self._runtime_hf_token)
        has_env_token = bool(os.getenv("HF_TOKEN"))
        source = "runtime" if has_runtime_token else "env" if has_env_token else "none"
        return {
            "configured": has_runtime_token or has_env_token,
            "source": source,
        }

    def set_hf_token(self, token: str) -> dict[str, Any]:
        token = token.strip()
        if not token:
            raise ValueError("HF token cannot be empty.")
        with self._lock:
            self._runtime_hf_token = token
        return self.hf_token_status()

    def clear_hf_token(self) -> dict[str, Any]:
        with self._lock:
            self._runtime_hf_token = None
        return self.hf_token_status()

    def cache_status(self, model_id: str) -> dict[str, Any]:
        return self._cache_status(
            model_id,
            include_active_job=True,
            hub_size_bytes=self._hub_size_bytes(model_id),
        )

    def _cache_status(
        self,
        model_id: str,
        include_active_job: bool,
        hub_size_bytes: int | None = None,
    ) -> dict[str, Any]:
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            return _local_model_cache_status(model, include_active_job=False)
        cache_path = self._model_cache_path(model["hf_id"])
        inspection = _inspect_model_cache(cache_path)
        active_job = self._active_job_for_model(model_id) if include_active_job else None
        if active_job:
            cache_state = active_job["status"]
            cached = False
        elif inspection["complete"]:
            cache_state = "cached"
            cached = True
        elif inspection["size_bytes"] > 0:
            cache_state = "partial"
            cached = False
        else:
            cache_state = "not_cached"
            cached = False
        return {
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "cached": cached,
            "cache_state": cache_state,
            "cache_path": str(cache_path),
            "size_bytes": inspection["size_bytes"],
            "hub_size_bytes": hub_size_bytes,
            "snapshot_count": inspection["snapshot_count"],
            "has_model_weights": inspection["has_model_weights"],
            "has_config": inspection["has_config"],
            "incomplete_file_count": inspection["incomplete_file_count"],
            "active_job": active_job,
        }

    def _hub_size_bytes(self, model_id: str) -> int | None:
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            return None
        hf_id = model["hf_id"]
        now = time.time()
        with self._lock:
            cached = self._hub_sizes.get(hf_id)
            if cached and now - float(cached.get("checked_at", 0)) < HUB_SIZE_CACHE_TTL_SECONDS:
                size_bytes = cached.get("size_bytes")
                return int(size_bytes) if isinstance(size_bytes, int) else None

        try:
            from huggingface_hub import HfApi

            with self._lock:
                runtime_token = self._runtime_hf_token
            info = HfApi().model_info(
                hf_id,
                files_metadata=True,
                timeout=10,
                token=runtime_token or os.getenv("HF_TOKEN") or None,
            )
            size_bytes = sum(
                size
                for sibling in info.siblings or []
                if not _is_ignored_download_file(sibling.rfilename)
                for size in [_hub_file_size(sibling)]
                if size is not None
            )
        except Exception:
            size_bytes = None

        with self._lock:
            self._hub_sizes[hf_id] = {"size_bytes": size_bytes, "checked_at": now}
            _write_hub_size_cache(self._hub_sizes)
        return size_bytes

    def start_download(self, model_id: str) -> dict[str, Any]:
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            return {
                "job_id": f"model_download_{uuid4().hex[:10]}",
                "model_id": model_id,
                "hf_id": model["hf_id"],
                "status": "completed",
                "message": "Local model path does not need downloading",
                "cache": self.cache_status(model_id),
                "created_at": _now(),
                "updated_at": _now(),
            }
        cache_status = self._cache_status(model_id, include_active_job=False)
        active_job = self._active_job_for_model(model_id)
        if active_job:
            return active_job
        job_id = f"model_download_{uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "status": "completed" if cache_status["cached"] else "queued",
            "message": "Model is already cached" if cache_status["cached"] else "Queued",
            "cache": cache_status if cache_status["cached"] else None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            _write_job(job)

        if cache_status["cached"]:
            return job

        thread = threading.Thread(target=self._download_model, args=(job_id,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return dict(job)
        job = _read_job(job_id)
        if not job:
            return None
        return self._reconcile_persisted_job(job, detached=True)

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        in_memory = False
        process_pid = None
        with self._lock:
            job = self._jobs.get(job_id)
            in_memory = job is not None
            if job is None:
                job = _read_job(job_id)
            if not job:
                raise ValueError("Model download job not found.")
            if job["status"] not in ACTIVE_DOWNLOAD_STATUSES:
                raise RuntimeError(f"Cannot cancel download job with status {job['status']}.")
            job.update(
                {
                    "status": "canceling",
                    "cancel_requested": True,
                    "message": "Canceling download and deleting partial cache",
                    "updated_at": _now(),
                }
            )
            process_pid = job.get("process_pid") if in_memory else None
            if in_memory:
                self._jobs[job_id] = job
            _write_job(job)

        if process_pid:
            _terminate_process(int(process_pid))
        if not in_memory:
            return self._reconcile_persisted_job(job, detached=True)
        return dict(job)

    def delete_model_cache(self, model_id: str) -> dict[str, Any]:
        active_job = self._active_job_for_model(model_id)
        if active_job:
            raise RuntimeError("Cannot delete model cache while a download is active.")
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            raise RuntimeError("Local model files are not deleted by cache management. Remove the registration instead.")
        cache_path = self._model_cache_path(model["hf_id"])
        if cache_path.exists():
            self._delete_with_huggingface_cache_api(model["hf_id"], cache_path)
        return self.cache_status(model_id)

    def _download_model(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            model_id = job["model_id"]
            if job.get("cancel_requested"):
                job.update(
                    {
                        "status": "canceled",
                        "message": "Download canceled",
                        "updated_at": _now(),
                    }
                )
                _write_job(job)
                return
            job["status"] = "downloading"
            job["message"] = "Downloading tokenizer and model weights from Hugging Face"
            job["updated_at"] = _now()
            _write_job(job)

        try:
            model = model_registry.get(model_id)
            process = self._start_snapshot_download_process(model["hf_id"])
            with self._lock:
                self._jobs[job_id]["process_pid"] = process.pid
                _write_job(self._jobs[job_id])

            while process.poll() is None:
                if self._download_cancel_requested(job_id):
                    _terminate_process(process.pid)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.5)

            if self._download_cancel_requested(job_id):
                self._delete_download_cache(model["hf_id"])
                status = self._cache_status(model_id, include_active_job=False)
                with self._lock:
                    self._jobs[job_id].update(
                        {
                            "status": "canceled",
                            "message": "Download canceled and partial cache deleted",
                            "cache": status,
                            "updated_at": _now(),
                        }
                    )
                    self._jobs[job_id].pop("process_pid", None)
                    _write_job(self._jobs[job_id])
                return

            if process.returncode != 0:
                raise RuntimeError(f"Download process exited with code {process.returncode}.")

            status = self._cache_status(model_id, include_active_job=False)
            if not status["cached"]:
                raise RuntimeError(
                    "Download command finished, but the local Hugging Face cache is incomplete. "
                    f"State={status['cache_state']}, size={status['size_bytes']} bytes."
                )
            with self._lock:
                self._jobs[job_id].update(
                    {
                        "status": "completed",
                        "message": "Model is cached",
                        "cache": status,
                        "updated_at": _now(),
                    }
                )
                self._jobs[job_id].pop("process_pid", None)
                _write_job(self._jobs[job_id])
        except Exception as exc:
            with self._lock:
                self._jobs[job_id].update(
                    {
                        "status": "failed",
                        "message": str(exc),
                        "updated_at": _now(),
                    }
                )
                self._jobs[job_id].pop("process_pid", None)
                _write_job(self._jobs[job_id])

    def _model_cache_path(self, hf_id: str) -> Path:
        cache_root = _hf_cache_root()
        return cache_root / f"models--{hf_id.replace('/', '--')}"

    def _hf_token(self) -> str | None:
        with self._lock:
            runtime_token = self._runtime_hf_token
        print(f"Using HF token from {'runtime' if runtime_token else 'env' if os.getenv('HF_TOKEN') else 'none'}")
        return runtime_token or os.getenv("HF_TOKEN") or None

    def _start_snapshot_download_process(self, hf_id: str) -> subprocess.Popen:
        env = os.environ.copy()
        env["KEW_HF_REPO_ID"] = hf_id
        env["KEW_HF_IGNORE_PATTERNS"] = json.dumps(MODEL_DOWNLOAD_IGNORE_PATTERNS)
        token = self._hf_token()
        if token:
            env["HF_TOKEN"] = token
        return subprocess.Popen(
            [sys.executable, "-c", SNAPSHOT_DOWNLOAD_SCRIPT],
            env=env,
            start_new_session=True,
        )

    def _download_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    def _delete_download_cache(self, hf_id: str) -> None:
        cache_path = self._model_cache_path(hf_id)
        if cache_path.exists():
            self._delete_with_huggingface_cache_api(hf_id, cache_path)

    def _delete_with_huggingface_cache_api(self, hf_id: str, cache_path: Path) -> None:
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(cache_dir=_hf_cache_root())
            repo = next(
                (
                    repo
                    for repo in cache_info.repos
                    if repo.repo_id == hf_id and repo.repo_type == "model"
                ),
                None,
            )
            if repo and repo.revisions:
                strategy = cache_info.delete_revisions(
                    *(revision.commit_hash for revision in repo.revisions)
                )
                strategy.execute()
        except Exception:
            # Partial/interrupted downloads are sometimes not represented cleanly
            # by the Hub cache scanner, so fall back to the exact model repo dir.
            pass

        if cache_path.exists():
            shutil.rmtree(cache_path)

    def _active_job_for_model(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            jobs = [
                dict(job)
                for job in self._jobs.values()
                if job["model_id"] == model_id and job["status"] in ACTIVE_DOWNLOAD_STATUSES
            ]
        if not jobs:
            jobs = [
                job
                for job in _read_jobs()
                if job["model_id"] == model_id and job["status"] in ACTIVE_DOWNLOAD_STATUSES
            ]
            jobs = [self._reconcile_persisted_job(job, detached=True) for job in jobs]
            jobs = [job for job in jobs if job["status"] in ACTIVE_DOWNLOAD_STATUSES]
            if not jobs:
                return None
        jobs.sort(key=lambda item: item["updated_at"], reverse=True)
        return jobs[0]

    def _reconcile_persisted_job(self, job: dict[str, Any], detached: bool) -> dict[str, Any]:
        if job["status"] not in ACTIVE_DOWNLOAD_STATUSES:
            return job
        if job["status"] == "canceling":
            model = model_registry.get(job["model_id"])
            self._delete_download_cache(model["hf_id"])
            status = self._cache_status(job["model_id"], include_active_job=False)
            job.update(
                {
                    "status": "canceled",
                    "message": "Download canceled and partial cache deleted",
                    "cache": status,
                    "updated_at": _now(),
                }
            )
            job.pop("process_pid", None)
            _write_job(job)
            return job
        status = self._cache_status(job["model_id"], include_active_job=False)
        if status["cached"]:
            job.update(
                {
                    "status": "completed",
                    "message": "Model is cached",
                    "cache": status,
                    "updated_at": _now(),
                }
            )
            _write_job(job)
            return job
        if detached:
            job.update(
                {
                    "status": "failed",
                    "message": (
                        "Download job is no longer active in the backend process; "
                        f"cache state is {status['cache_state']}."
                    ),
                    "cache": status,
                    "updated_at": _now(),
                }
            )
            _write_job(job)
            return job
        return job


def _hf_cache_root() -> Path:
    explicit = (
        os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
    )
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _read_hub_size_cache() -> dict[str, dict[str, Any]]:
    try:
        payload = local_store.read_json("models/hub_size_cache.json")
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_hub_size_cache(cache: dict[str, dict[str, Any]]) -> None:
    local_store.write_json("models/hub_size_cache.json", cache)


def _is_ignored_download_file(filename: str) -> bool:
    return any(fnmatch(filename, pattern) for pattern in MODEL_DOWNLOAD_IGNORE_PATTERNS)


def _hub_file_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    if size is None:
        size = getattr(getattr(sibling, "lfs", None), "size", None)
    return int(size) if isinstance(size, int) else None


def _terminate_process(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def _directory_size(path: Path) -> int:
    total = 0
    seen_files: set[tuple[int, int]] = set()
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            try:
                stat = item.stat()
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            total += stat.st_size
    return total


def _inspect_model_cache(path: Path) -> dict[str, Any]:
    size_bytes = _directory_size(path) if path.exists() else 0
    snapshots_root = path / "snapshots"
    snapshots = [item for item in snapshots_root.iterdir() if item.is_dir()] if snapshots_root.exists() else []
    incomplete_files = list(path.rglob("*.incomplete")) if path.exists() else []
    complete_snapshots = [_snapshot_is_complete(snapshot) for snapshot in snapshots]
    return {
        "size_bytes": size_bytes,
        "snapshot_count": len(snapshots),
        "has_config": any(item["has_config"] for item in complete_snapshots),
        "has_model_weights": any(item["has_model_weights"] for item in complete_snapshots),
        "incomplete_file_count": len(incomplete_files),
        "complete": any(item["complete"] for item in complete_snapshots) and not incomplete_files,
    }


def _local_model_cache_status(model: dict[str, Any], include_active_job: bool) -> dict[str, Any]:
    path = Path(model["local_path"])
    inspection = _inspect_local_model_path(path)
    if inspection["complete"]:
        cache_state = "cached"
    elif path.exists():
        cache_state = "partial"
    else:
        cache_state = "not_cached"
    return {
        "model_id": model["id"],
        "hf_id": model["hf_id"],
        "cached": inspection["complete"],
        "cache_state": cache_state,
        "cache_path": str(path),
        "size_bytes": inspection["size_bytes"],
        "hub_size_bytes": None,
        "snapshot_count": 1 if path.exists() else 0,
        "has_model_weights": inspection["has_model_weights"],
        "has_config": inspection["has_config"],
        "incomplete_file_count": 0,
        "active_job": None,
        "is_local": True,
    }


def _inspect_local_model_path(path: Path) -> dict[str, Any]:
    size_bytes = _directory_size(path) if path.exists() else 0
    if not path.exists():
        return {
            "size_bytes": size_bytes,
            "has_config": False,
            "has_model_weights": False,
            "complete": False,
        }
    snapshot = _snapshot_is_complete(path)
    return {
        "size_bytes": size_bytes,
        "has_config": snapshot["has_config"],
        "has_model_weights": snapshot["has_model_weights"],
        "complete": snapshot["complete"],
    }


def _snapshot_is_complete(snapshot: Path) -> dict[str, bool]:
    has_config = (snapshot / "config.json").exists()
    has_weights = any(
        item.is_file()
        and (
            item.name.endswith(".safetensors")
            or item.name.endswith(".bin")
        )
        for item in snapshot.rglob("*")
    )
    return {
        "has_config": has_config,
        "has_model_weights": has_weights,
        "complete": has_config and has_weights,
    }


def _jobs_dir() -> Path:
    path = settings.data_dir / "model_cache_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    _job_path(job["job_id"]).write_text(json.dumps(job, indent=2), encoding="utf-8")


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jobs() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _jobs_dir().glob("model_download_*.json")
    ]


model_cache_service = ModelCacheService()
