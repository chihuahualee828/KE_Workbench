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
from typing import Any, BinaryIO
from uuid import uuid4

from app.config import settings
from app.services.model_registry import model_registry
from app.services.model_cache_paths import model_cache_paths
from app.storage.local_store import local_store

MODEL_DOWNLOAD_IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
]
LEGACY_PYTORCH_WEIGHT_PATTERNS = [
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
]

ACTIVE_DOWNLOAD_STATUSES = {"queued", "downloading", "canceling"}
ACTIVE_IMPORT_STATUSES = {"uploading", "validating", "registering", "canceling"}
ACTIVE_MODEL_JOB_STATUSES = ACTIVE_DOWNLOAD_STATUSES | ACTIVE_IMPORT_STATUSES
HUB_SIZE_CACHE_TTL_SECONDS = 6 * 60 * 60
HUB_SIZE_CACHE_POLICY_VERSION = 2

SNAPSHOT_DOWNLOAD_SCRIPT = """
import json
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
snapshot_download(
    repo_id=os.environ["KEW_HF_REPO_ID"],
    cache_dir=os.environ["KEW_HF_CACHE_DIR"],
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
        self._local_import_lock = threading.RLock()
        model_cache_paths.remember_environment()

    def cache_location(self) -> dict[str, Any]:
        return model_cache_paths.configuration()

    def set_cache_location(self, download_dir: str) -> dict[str, Any]:
        with self._lock:
            if any(job["status"] in ACTIVE_DOWNLOAD_STATUSES
                   and job.get("job_type") != "local_import" for job in self._jobs.values()):
                raise RuntimeError("Wait for active model downloads to finish before changing the cache folder.")
            return model_cache_paths.set_directory(download_dir)

    def cache_root_for_model(self, hf_id: str) -> Path:
        if Path(hf_id).is_dir():
            return _hf_cache_root()
        path, inspection = self._cache_inspection(hf_id)
        return path.parent if inspection["complete"] else _hf_cache_root()

    def _cache_inspection(self, hf_id: str) -> tuple[Path, dict[str, Any]]:
        fallback = None
        for root in model_cache_paths.roots():
            path = root / f"models--{hf_id.replace('/', '--')}"
            try:
                inspection = _inspect_model_cache(path)
            except OSError:
                continue
            if inspection["complete"]:
                return path, inspection
            if fallback is None or (not fallback[1]["size_bytes"] and inspection["size_bytes"]):
                fallback = (path, inspection)
        if fallback is not None:
            return fallback
        path = self._model_cache_path(hf_id)
        return path, _inspect_model_cache(path)

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
        cache_root: Path | None = None,
    ) -> dict[str, Any]:
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            return _local_model_cache_status(model, include_active_job=False)
        active_job = self._active_job_for_model(model_id) if include_active_job else None
        if active_job:
            cache_root = _download_cache_root(active_job)
        if cache_root is not None:
            cache_path = cache_root / f"models--{model['hf_id'].replace('/', '--')}"
            inspection = _inspect_model_cache(cache_path)
        else:
            cache_path, inspection = self._cache_inspection(model["hf_id"])
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
            if (
                cached
                and cached.get("policy_version") == HUB_SIZE_CACHE_POLICY_VERSION
                and now - float(cached.get("checked_at", 0)) < HUB_SIZE_CACHE_TTL_SECONDS
            ):
                size_bytes = cached.get("size_bytes")
                return int(size_bytes) if isinstance(size_bytes, int) else None

        try:
            info = self._hub_model_info(hf_id)
            ignore_patterns = _download_ignore_patterns(
                [sibling.rfilename for sibling in info.siblings or []]
            )
            size_bytes = sum(
                size
                for sibling in info.siblings or []
                if not _is_ignored_download_file(sibling.rfilename, ignore_patterns)
                for size in [_hub_file_size(sibling)]
                if size is not None
            )
        except Exception:
            size_bytes = None

        with self._lock:
            self._hub_sizes[hf_id] = {
                "size_bytes": size_bytes,
                "checked_at": now,
                "policy_version": HUB_SIZE_CACHE_POLICY_VERSION,
            }
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
        job.update(_download_progress_snapshot(job, cache_status))
        with self._lock:
            job["cache_dir"] = str(_hf_cache_root())
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

    def list_jobs(self) -> list[dict[str, Any]]:
        persisted_jobs = {job["job_id"]: job for job in _read_jobs()}
        with self._lock:
            in_memory_jobs = {job_id: dict(job) for job_id, job in self._jobs.items()}

        jobs = {**persisted_jobs, **in_memory_jobs}
        for job_id, job in jobs.items():
            if job_id not in in_memory_jobs:
                jobs[job_id] = self._reconcile_persisted_job(job, detached=True)
        return sorted(jobs.values(), key=lambda job: job["updated_at"], reverse=True)

    def clear_finished_jobs(self) -> int:
        with self._lock:
            active_job_ids = {
                job_id
                for job_id, job in self._jobs.items()
                if job["status"] in ACTIVE_MODEL_JOB_STATUSES
            }
            for job_id in [
                job_id
                for job_id, job in self._jobs.items()
                if job["status"] not in ACTIVE_MODEL_JOB_STATUSES
            ]:
                self._jobs.pop(job_id, None)

        deleted = 0
        for path in _jobs_dir().glob("model_*.json"):
            if path.stem in active_job_ids:
                continue
            job = _read_job(path.stem)
            if job and job["status"] in ACTIVE_MODEL_JOB_STATUSES:
                continue
            path.unlink(missing_ok=True)
            if job and job.get("job_type") == "local_import":
                shutil.rmtree(_local_import_job_dir(path.stem), ignore_errors=True)
            deleted += 1
        return deleted

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        in_memory = False
        process_pid = None
        with self._lock:
            job = self._jobs.get(job_id)
            in_memory = job is not None
            if job is None:
                job = _read_job(job_id)
            if not job:
                raise ValueError("Model transfer job not found.")
            if job["status"] not in ACTIVE_MODEL_JOB_STATUSES:
                raise RuntimeError(f"Cannot cancel transfer job with status {job['status']}.")
            if job.get("job_type") == "local_import":
                job.update(
                    {
                        "status": "canceled",
                        "cancel_requested": True,
                        "message": "Local model import canceled",
                        "updated_at": _now(),
                    }
                )
                if in_memory:
                    self._jobs[job_id] = job
                _write_job(job)
                shutil.rmtree(_local_import_job_dir(job_id), ignore_errors=True)
                return dict(job)
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
        for root in model_cache_paths.roots():
            cache_path = root / f"models--{model['hf_id'].replace('/', '--')}"
            if cache_path.exists():
                self._delete_with_huggingface_cache_api(model["hf_id"], cache_path)
        return self.cache_status(model_id)

    def stage_local_model(self, files: list[tuple[str, BinaryIO]]) -> dict[str, Any]:
        """Stage a folder import and return a validated, unregistered model preview."""
        if not files:
            raise ValueError("Select a model folder containing config.json and model weights.")

        total_bytes = sum(_binary_stream_size(stream) for _, stream in files)
        job = self._create_local_import_job(
            source_name="Selected model folder",
            source_kind="folder",
            total_bytes=total_bytes,
        )
        job_id = job["job_id"]
        staging_root = _local_import_job_dir(job_id)
        source_root = staging_root / "source"
        try:
            with self._local_import_lock:
                source_root.mkdir(parents=True, exist_ok=False)
                seen_paths: set[Path] = set()
                processed_bytes = 0
                for filename, stream in files:
                    relative_path = _safe_uploaded_relative_path(filename)
                    if relative_path in seen_paths:
                        raise ValueError(f"The selected folder contains duplicate file path: {relative_path}")
                    seen_paths.add(relative_path)
                    destination_path = source_root / relative_path
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    copied = _copy_stream_to_path(stream, destination_path)
                    processed_bytes += copied
                    self._update_local_import_job(
                        job_id,
                        progress_current_bytes=processed_bytes,
                        message="Copying selected model files into the import staging area",
                    )
            return self._validate_staged_local_import(job_id, source_root)
        except Exception as exc:
            self._fail_local_import_job(job_id, str(exc))
            raise

    def start_folder_import_job(self, source_name: str, total_bytes: int) -> dict[str, Any]:
        """Create a visible import job before the browser starts transferring files."""
        if total_bytes <= 0:
            raise ValueError("The selected model folder does not contain any files to upload.")
        display_name = Path(source_name).name.strip() or "Selected model folder"
        return self._create_local_import_job(
            source_name=display_name,
            source_kind="folder",
            total_bytes=total_bytes,
        )

    def append_folder_import_chunk(
        self,
        job_id: str,
        relative_filename: str,
        offset: int,
        stream: BinaryIO,
    ) -> dict[str, Any]:
        """Append one browser upload chunk and persist its byte-level progress."""
        if offset < 0:
            raise ValueError("Upload chunk offset cannot be negative.")
        relative_path = _safe_uploaded_relative_path(relative_filename)
        with self._local_import_lock:
            job = self._require_local_import_job(job_id)
            if job.get("source_kind") != "folder":
                raise ValueError("This import job does not accept model folder files.")
            if job.get("status") != "uploading":
                raise RuntimeError("This local import is no longer accepting uploaded files.")

            uploaded_files = dict(job.get("uploaded_files") or {})
            uploaded_for_file = int(uploaded_files.get(str(relative_path), 0))
            if offset != uploaded_for_file:
                raise RuntimeError("The upload chunk offset does not match the staged file.")

            source_root = _local_import_job_dir(job_id) / "source"
            source_root.mkdir(parents=True, exist_ok=True)
            destination = (source_root / relative_path).resolve()
            if not _is_path_within(destination, source_root.resolve()):
                raise ValueError("Uploaded model files must have safe relative paths.")
            destination.parent.mkdir(parents=True, exist_ok=True)

            total_bytes = int(job.get("progress_total_bytes") or 0)
            already_uploaded = int(job.get("progress_current_bytes") or 0)
            remaining_bytes = total_bytes - already_uploaded
            if remaining_bytes < 0:
                raise RuntimeError("The import job has invalid upload progress.")
            copied = _append_stream_to_path(stream, destination, remaining_bytes)
            uploaded_files[str(relative_path)] = uploaded_for_file + copied
            current_bytes = already_uploaded + copied
            return self._update_local_import_job(
                job_id,
                progress_current_bytes=current_bytes,
                uploaded_files=uploaded_files,
                message=f"Uploading model files ({current_bytes} of {total_bytes} bytes)",
            )

    def complete_folder_import_upload(self, job_id: str) -> dict[str, Any]:
        """Validate a fully transferred folder and produce its registration preview."""
        with self._local_import_lock:
            job = self._require_local_import_job(job_id)
            if job.get("source_kind") != "folder":
                raise ValueError("This import job is not a model folder upload.")
            if job.get("status") != "uploading":
                raise RuntimeError("This local import is no longer awaiting upload completion.")
            total_bytes = int(job.get("progress_total_bytes") or 0)
            current_bytes = int(job.get("progress_current_bytes") or 0)
            if current_bytes != total_bytes:
                raise RuntimeError(
                    f"The folder upload is incomplete ({current_bytes} of {total_bytes} bytes received)."
                )
            try:
                return self._validate_staged_local_import(job_id, _local_import_job_dir(job_id) / "source")
            except Exception as exc:
                self._fail_local_import_job(job_id, str(exc))
                raise

    def register_staged_local_model(self, import_job_id: str, model: dict[str, Any]) -> dict[str, Any]:
        """Move a successful staged upload into the managed cache on Register."""
        with self._local_import_lock:
            job = self._require_local_import_job(import_job_id)
            if job["status"] != "completed" or not job.get("preview"):
                raise RuntimeError("Finish and validate the uploaded model before registering it.")
            if job.get("registered_model_id"):
                raise RuntimeError(f"Import job already registered {job['registered_model_id']}.")

            model_id = str(model.get("id") or "").strip()
            if not model_id:
                raise ValueError("Model ID is required.")
            if any(item["id"] == model_id for item in model_registry.list_models()):
                raise ValueError(f"A model named {model_id} is already registered.")

            staging_root = _local_import_job_dir(import_job_id)
            staged_model_path = _staged_model_path(job, staging_root)
            if not staged_model_path.is_dir() or not _is_path_within(staged_model_path, staging_root):
                raise RuntimeError("The staged local import is no longer available.")

            cache_root = _local_model_cache_root()
            destination = cache_root / model_id
            if destination.exists():
                raise RuntimeError(f"Managed local cache path already exists: {destination}")
            self._update_local_import_job(
                import_job_id,
                status="registering",
                message="Registering validated model in the managed local cache",
            )
            moved = False
            try:
                cache_root.mkdir(parents=True, exist_ok=True)
                staged_model_path.rename(destination)
                moved = True
                preview = dict(job["preview"])
                registration = {
                    **preview,
                    "id": model_id,
                    "name": str(model.get("name") or preview.get("display_base_name") or preview["name"]),
                    "import_metadata": {
                        "import_id": import_job_id,
                        "uploaded_at": job.get("created_at"),
                        "source_folder": job.get("source_name"),
                    },
                    # The inspected architecture remains the authority for an
                    # upload. Analysis/editing capabilities are not manual
                    # registration switches.
                    **{
                        key: bool(preview[key])
                        for key in (
                            "supports_generation",
                            "supports_logit_lens",
                            "supports_causal_trace",
                            "supports_rome",
                            "supports_r_rome",
                            "supports_memit",
                            "supports_ft",
                            "supports_lora",
                            "supports_kn",
                        )
                    },
                }
                registration.pop("import_job_id", None)
                registration["local_path"] = str(destination)
                stored = model_registry.add_local_model(registration, managed_local=True)
            except Exception:
                if moved and destination.exists() and not staged_model_path.exists():
                    destination.rename(staged_model_path)
                self._update_local_import_job(
                    import_job_id,
                    status="completed",
                    message="Registration failed; staged files were retained for retry.",
                )
                raise

            self._update_local_import_job(
                import_job_id,
                status="completed",
                message=f"Registered local model {stored['id']}",
                registered_model_id=stored["id"],
                progress_current_bytes=int(job.get("progress_total_bytes") or 0),
                progress_total_bytes=int(job.get("progress_total_bytes") or 0),
            )
            shutil.rmtree(staging_root, ignore_errors=True)
            return stored

    def delete_local_model(self, model_id: str) -> dict[str, Any]:
        """Remove a local registration and its files only when we imported them."""
        model = model_registry.get(model_id)
        if model.get("source") != "local":
            raise ValueError("Only local model registrations can be removed here.")

        local_path = Path(model.get("local_path") or "").expanduser().resolve()
        managed_local = bool(model.get("managed_local"))
        cache_root = _local_model_cache_root().resolve()
        if managed_local and not _is_path_within(local_path, cache_root):
            raise RuntimeError("Refusing to delete a managed local model outside the workbench cache.")

        removed = model_registry.delete_local_model(model_id)
        if managed_local and local_path.exists():
            shutil.rmtree(local_path)
        return {
            "status": "deleted",
            "model_id": model_id,
            "deleted_cache_files": managed_local,
            "cache_path": str(local_path),
            "model": removed,
        }

    def _create_local_import_job(
        self,
        *,
        source_name: str,
        source_kind: str,
        total_bytes: int,
    ) -> dict[str, Any]:
        if total_bytes < 0:
            raise ValueError("Upload size cannot be negative.")
        job_id = f"model_import_{uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "job_type": "local_import",
            # Import jobs intentionally do not map to a cache-table model until
            # the user has reviewed the inferred fields and clicked Register.
            "model_id": f"local-import-{job_id.rsplit('_', 1)[-1]}",
            "hf_id": "",
            "source_name": source_name,
            "source_kind": source_kind,
            "status": "uploading",
            "message": "Waiting for local model files",
            "progress_current_bytes": 0,
            "progress_total_bytes": total_bytes,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            _write_job(job)
        return dict(job)

    def _require_local_import_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            job = _read_job(job_id)
            if job is not None:
                with self._lock:
                    self._jobs[job_id] = job
        if not job or job.get("job_type") != "local_import":
            raise ValueError("Local model import job not found.")
        return dict(job)

    def _update_local_import_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = _read_job(job_id)
            if not job or job.get("job_type") != "local_import":
                raise ValueError("Local model import job not found.")
            job.update(updates)
            job["updated_at"] = _now()
            self._jobs[job_id] = job
            _write_job(job)
            return dict(job)

    def _fail_local_import_job(self, job_id: str, message: str) -> None:
        try:
            self._update_local_import_job(job_id, status="failed", message=message)
        except ValueError:
            return
        shutil.rmtree(_local_import_job_dir(job_id), ignore_errors=True)

    def _validate_staged_local_import(
        self,
        job_id: str,
        staging_root: Path,
    ) -> dict[str, Any]:
        self._update_local_import_job(
            job_id,
            status="validating",
            message="Reading config.json and validating model compatibility",
        )
        model_root = _uploaded_model_root(staging_root)
        config = _read_uploaded_model_config(model_root)
        metadata = model_registry.inspect_imported_local_model_config(config)
        if not any(_is_transformer_weight_file(path) for path in model_root.rglob("*")):
            raise ValueError(
                "The selected model is missing supported Transformer weights "
                "(model.safetensors or pytorch_model.bin)."
            )
        if not _has_tokenizer_assets(model_root):
            raise ValueError(
                "The selected model is missing tokenizer files. Include tokenizer.json, "
                "tokenizer.model, vocab.txt, or GPT-2 vocab.json plus merges.txt."
            )

        model_id = model_registry.allocate_imported_model_id()
        preview = {
            "id": model_id,
            "name": metadata["display_base_name"],
            "display_base_name": metadata["display_base_name"],
            "local_path": str(model_root),
            "base_model_id": metadata["base_model_id"],
            "architecture_model_type": metadata["architecture_model_type"],
            "model_type": metadata["model_type"],
            "n_layers": metadata["n_layers"],
            "supports_generation": metadata["supports_generation"],
            "supports_logit_lens": metadata["supports_logit_lens"],
            "supports_causal_trace": metadata["supports_causal_trace"],
            "supports_rome": metadata["supports_rome"],
            "supports_r_rome": metadata["supports_r_rome"],
            "supports_memit": metadata["supports_memit"],
            "supports_ft": metadata["supports_ft"],
            "supports_lora": metadata["supports_lora"],
            "supports_kn": metadata["supports_kn"],
            "trust_remote_code": metadata["trust_remote_code"],
            "generation_use_cache": metadata["generation_use_cache"],
            "rome_hparams": metadata["easyedit"].get("rome_hparams"),
            "r_rome_hparams": metadata["easyedit"].get("r_rome_hparams"),
            "memit_hparams": metadata["easyedit"].get("memit_hparams"),
            "ft_hparams": metadata["easyedit"].get("ft_hparams"),
            "lora_hparams": metadata["easyedit"].get("lora_hparams"),
            "kn_hparams": metadata["easyedit"].get("kn_hparams"),
        }
        relative_model_path = model_root.relative_to(_local_import_job_dir(job_id))
        job = self._require_local_import_job(job_id)
        total_bytes = int(job.get("progress_total_bytes") or 0)
        return self._update_local_import_job(
            job_id,
            status="completed",
            message="Upload validated. Review the filled fields and click Register.",
            progress_current_bytes=total_bytes,
            progress_total_bytes=total_bytes,
            preview=preview,
            metadata=metadata,
            staged_model_relative=str(relative_model_path),
        )

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
            total_bytes = self._hub_size_bytes(model_id)
            with self._lock:
                self._jobs[job_id]["progress_total_bytes"] = total_bytes
            ignore_patterns = self._download_ignore_patterns_for_model(model["hf_id"])
            process = self._start_snapshot_download_process(
                model["hf_id"], ignore_patterns, _download_cache_root(job)
            )
            with self._lock:
                self._jobs[job_id]["process_pid"] = process.pid
                _write_job(self._jobs[job_id])

            last_snapshot = 0.0
            while process.poll() is None:
                if time.monotonic() - last_snapshot >= 2:
                    try:
                        self._capture_download_progress(job_id)
                    except Exception:
                        # A transient inspection/storage error must not orphan
                        # the running downloader. Retry on the next interval.
                        pass
                    last_snapshot = time.monotonic()
                if self._download_cancel_requested(job_id):
                    _terminate_process(process.pid)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(0.5)

            _, error_output = process.communicate()
            # Capture the transfer before cancellation deletes its partial files.
            status = self._capture_download_progress(job_id)

            if self._download_cancel_requested(job_id):
                self._delete_download_cache(model["hf_id"], _download_cache_root(job))
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
                raise RuntimeError(
                    _download_failure_message(
                        hf_id=model["hf_id"],
                        returncode=process.returncode,
                        error_output=error_output,
                    )
                )

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
            try:
                self._capture_download_progress(job_id)
            except Exception:
                # Retain the last snapshot if cache inspection also fails.
                pass
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

    def _capture_download_progress(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = dict(self._jobs[job_id])
        status = self._cache_status(
            job["model_id"],
            include_active_job=False,
            hub_size_bytes=job.get("progress_total_bytes"),
            cache_root=_download_cache_root(job),
        )
        with self._lock:
            current = self._jobs[job_id]
            if current["status"] in ACTIVE_DOWNLOAD_STATUSES:
                current.update(_download_progress_snapshot(current, status))
                _write_job(current)
        return status

    def _model_cache_path(self, hf_id: str) -> Path:
        cache_root = _hf_cache_root()
        return cache_root / f"models--{hf_id.replace('/', '--')}"

    def _hf_token(self) -> str | None:
        with self._lock:
            runtime_token = self._runtime_hf_token
        print(f"Using HF token from {'runtime' if runtime_token else 'env' if os.getenv('HF_TOKEN') else 'none'}")
        return runtime_token or os.getenv("HF_TOKEN") or None

    def _hub_model_info(self, hf_id: str):
        from huggingface_hub import HfApi

        with self._lock:
            runtime_token = self._runtime_hf_token
        return HfApi().model_info(
            hf_id,
            files_metadata=True,
            timeout=10,
            token=runtime_token or os.getenv("HF_TOKEN") or None,
        )

    def _download_ignore_patterns_for_model(self, hf_id: str) -> list[str]:
        try:
            info = self._hub_model_info(hf_id)
        except Exception:
            # If metadata is temporarily unavailable, retain legacy weights so a
            # model with no safetensors alternative can still be loaded.
            return list(MODEL_DOWNLOAD_IGNORE_PATTERNS)
        return _download_ignore_patterns([sibling.rfilename for sibling in info.siblings or []])

    def _start_snapshot_download_process(
        self,
        hf_id: str,
        ignore_patterns: list[str],
        cache_root: Path | None = None,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env["KEW_HF_REPO_ID"] = hf_id
        env["KEW_HF_CACHE_DIR"] = str(cache_root or _hf_cache_root())
        env["KEW_HF_IGNORE_PATTERNS"] = json.dumps(ignore_patterns)
        token = self._hf_token()
        if token:
            env["HF_TOKEN"] = token
        return subprocess.Popen(
            [sys.executable, "-c", SNAPSHOT_DOWNLOAD_SCRIPT],
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )

    def _download_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    def _delete_download_cache(self, hf_id: str, cache_root: Path | None = None) -> None:
        cache_path = (cache_root or _hf_cache_root()) / f"models--{hf_id.replace('/', '--')}"
        if cache_path.exists():
            self._delete_with_huggingface_cache_api(hf_id, cache_path)

    def _delete_with_huggingface_cache_api(self, hf_id: str, cache_path: Path) -> None:
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(cache_dir=cache_path.parent)
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
                if job.get("job_type") != "local_import"
                and job["model_id"] == model_id
                and job["status"] in ACTIVE_DOWNLOAD_STATUSES
            ]
        if not jobs:
            jobs = [
                job
                for job in _read_jobs()
                if job.get("job_type") != "local_import"
                and job["model_id"] == model_id
                and job["status"] in ACTIVE_DOWNLOAD_STATUSES
            ]
            jobs = [self._reconcile_persisted_job(job, detached=True) for job in jobs]
            jobs = [job for job in jobs if job["status"] in ACTIVE_DOWNLOAD_STATUSES]
            if not jobs:
                return None
        jobs.sort(key=lambda item: item["updated_at"], reverse=True)
        return jobs[0]

    def _reconcile_persisted_job(self, job: dict[str, Any], detached: bool) -> dict[str, Any]:
        if job.get("job_type") == "local_import":
            if job["status"] not in ACTIVE_IMPORT_STATUSES:
                return job
            if job["status"] == "uploading":
                # Upload chunks can be resumed after a backend restart because
                # the received archive and byte offset are persisted.
                return job
            if job["status"] == "canceling" or job.get("cancel_requested"):
                job.update(
                    {
                        "status": "canceled",
                        "message": "Local model import canceled",
                        "updated_at": _now(),
                    }
                )
                _write_job(job)
                shutil.rmtree(_local_import_job_dir(job["job_id"]), ignore_errors=True)
                return job
            if detached:
                job.update(
                    {
                        "status": "failed",
                        "message": "Local import processing stopped when the backend restarted. Upload again.",
                        "updated_at": _now(),
                    }
                )
                _write_job(job)
                shutil.rmtree(_local_import_job_dir(job["job_id"]), ignore_errors=True)
            return job
        if job["status"] not in ACTIVE_DOWNLOAD_STATUSES:
            return job
        if job["status"] == "canceling":
            model = model_registry.get(job["model_id"])
            status = self._cache_status(job["model_id"], include_active_job=False,
                                        cache_root=_download_cache_root(job))
            job.update(_download_progress_snapshot(job, status))
            self._delete_download_cache(model["hf_id"], _download_cache_root(job))
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
        status = self._cache_status(job["model_id"], include_active_job=False,
                                    cache_root=_download_cache_root(job))
        job.update(_download_progress_snapshot(job, status))
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


def _download_progress_snapshot(job: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Freeze progress on this job, independently of subsequent cache changes."""
    downloaded = status["size_bytes"]
    total = status.get("hub_size_bytes") or job.get("progress_total_bytes")
    if total is None and status.get("cached"):
        total = downloaded
    return {
        "cache": {**status, "active_job": None},
        "progress_current_bytes": downloaded,
        "progress_total_bytes": total,
    }


def _hf_cache_root() -> Path:
    return model_cache_paths.download_dir()


def _download_cache_root(job: dict[str, Any]) -> Path:
    if job.get("cache_dir"):
        return Path(job["cache_dir"])
    previous_path = (job.get("cache") or {}).get("cache_path")
    return Path(previous_path).parent if previous_path else _hf_cache_root()


def _local_model_cache_root() -> Path:
    """The app-owned persistent store for browser-imported model snapshots.

    Keeping these files under the backend data directory separates imported
    local models from Hugging Face Hub's cache and makes them travel with the
    workbench's own persisted data.
    """
    return settings.data_dir / "models" / "imported_models"


def _safe_uploaded_relative_path(filename: str) -> Path:
    if not filename or filename.startswith(("/", "\\")) or "\\" in filename:
        raise ValueError("Uploaded model files must have safe relative paths.")
    relative_path = Path(filename)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("Uploaded model files must have safe relative paths.")
    return relative_path


def _binary_stream_size(stream: BinaryIO) -> int:
    """Read a seekable upload's remaining size without changing its position."""
    try:
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(position)
    except (AttributeError, OSError):
        raise ValueError("Could not read an uploaded model file.") from None
    return max(0, end - position)


def _copy_stream_to_path(stream: BinaryIO, destination: Path) -> int:
    copied = 0
    with destination.open("wb") as target:
        while chunk := stream.read(1024 * 1024):
            target.write(chunk)
            copied += len(chunk)
    return copied


def _append_stream_to_path(stream: BinaryIO, destination: Path, maximum_bytes: int) -> int:
    """Append one request body while refusing to exceed the declared import size."""
    copied = 0
    with destination.open("ab") as target:
        while chunk := stream.read(1024 * 1024):
            copied += len(chunk)
            if copied > maximum_bytes:
                raise ValueError("Uploaded data exceeds the declared folder size.")
            target.write(chunk)
    return copied


def _local_import_staging_root() -> Path:
    path = settings.data_dir / "model_cache_jobs" / "local_import_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_import_job_dir(job_id: str) -> Path:
    if not job_id.startswith("model_import_") or any(character not in "0123456789abcdef_" for character in job_id.removeprefix("model_import_")):
        raise ValueError("Invalid local import job identifier.")
    return _local_import_staging_root() / job_id


def _staged_model_path(job: dict[str, Any], staging_root: Path) -> Path:
    relative_path = job.get("staged_model_relative")
    if not isinstance(relative_path, str):
        raise RuntimeError("The staged local import has no model path.")
    path = (staging_root / relative_path).resolve()
    if not _is_path_within(path, staging_root.resolve()):
        raise RuntimeError("The staged local import path is unsafe.")
    return path


def _uploaded_model_root(staging_root: Path) -> Path:
    config_paths = [path for path in staging_root.rglob("config.json") if path.is_file()]
    if len(config_paths) != 1:
        raise ValueError(
            "Select exactly one Hugging Face model folder. It must contain one top-level config.json."
        )
    return config_paths[0].parent


def _read_uploaded_model_config(model_root: Path) -> dict[str, Any]:
    config_path = model_root / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The selected config.json is not valid UTF-8 JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")
    return config


def _has_tokenizer_assets(model_root: Path) -> bool:
    filenames = {path.name for path in model_root.rglob("*") if path.is_file()}
    if {"vocab.json", "merges.txt"}.issubset(filenames):
        return True
    return bool(
        {
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.txt",
        }.intersection(filenames)
    )


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_hub_size_cache() -> dict[str, dict[str, Any]]:
    try:
        payload = local_store.read_json("models/hub_size_cache.json")
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_hub_size_cache(cache: dict[str, dict[str, Any]]) -> None:
    local_store.write_json("models/hub_size_cache.json", cache)


def _download_ignore_patterns(filenames: list[str]) -> list[str]:
    patterns = list(MODEL_DOWNLOAD_IGNORE_PATTERNS)
    if _has_safetensors_model_weights(filenames):
        patterns.extend(LEGACY_PYTORCH_WEIGHT_PATTERNS)
    return patterns


def _is_ignored_download_file(filename: str, ignore_patterns: list[str]) -> bool:
    return any(fnmatch(filename, pattern) for pattern in ignore_patterns)


def _has_safetensors_model_weights(filenames: list[str]) -> bool:
    for filename in filenames:
        name = Path(filename).name
        if name in {"model.safetensors", "model.safetensors.index.json"}:
            return True
        if name.startswith("model-") and name.endswith(".safetensors"):
            return True
    return False


def _hub_file_size(sibling: Any) -> int | None:
    size = getattr(sibling, "size", None)
    if size is None:
        size = getattr(getattr(sibling, "lfs", None), "size", None)
    return int(size) if isinstance(size, int) else None


def _download_failure_message(hf_id: str, returncode: int, error_output: str) -> str:
    error_text = error_output.lower()
    if (
        "gated repo" in error_text
        or "gatedrepoerror" in error_text
        or "authorized list" in error_text
    ):
        return (
            f"Access to {hf_id} is restricted. Request access on its Hugging Face model page, "
            "then add a token from the approved account in Settings."
        )
    if "401" in error_text or "403" in error_text or "repository not found" in error_text:
        return (
            f"Hugging Face denied access to {hf_id}. Check the token in Settings and confirm "
            "that its account can access this model."
        )
    return f"Download process exited with code {returncode}. Check the backend logs for details."


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
    snapshots_root = path / "snapshots"
    snapshots = [item for item in snapshots_root.iterdir() if item.is_dir()] if snapshots_root.exists() else []
    size_bytes = _selected_model_cache_size(path, snapshots)
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


def _selected_model_cache_size(path: Path, snapshots: list[Path]) -> int:
    """Measure only files selected by this app's snapshot download policy.

    Hugging Face snapshots link into a blob directory. We count snapshot targets
    once, omit legacy PyTorch weights when a safetensors equivalent is present,
    and retain unreferenced blobs so partial-download progress remains visible.
    """
    if not path.exists():
        return 0

    snapshots_root = path / "snapshots"
    snapshot_target_ids: set[tuple[int, int]] = set()
    counted_ids: set[tuple[int, int]] = set()
    total = 0

    for snapshot in snapshots:
        files = [item for item in snapshot.rglob("*") if item.is_file()]
        ignore_patterns = _download_ignore_patterns(
            [str(item.relative_to(snapshot)) for item in files]
        )
        for item in files:
            try:
                stat = item.stat()
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            snapshot_target_ids.add(identity)
            if _is_ignored_download_file(str(item.relative_to(snapshot)), ignore_patterns):
                continue
            if identity not in counted_ids:
                counted_ids.add(identity)
                total += stat.st_size

    for item in path.rglob("*"):
        if snapshots_root in item.parents or not item.is_file():
            continue
        try:
            stat = item.stat()
        except OSError:
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in snapshot_target_ids or identity in counted_ids:
            continue
        counted_ids.add(identity)
        total += stat.st_size

    return total


def _local_model_cache_status(model: dict[str, Any], include_active_job: bool) -> dict[str, Any]:
    path = Path(str(model.get("local_path") or ""))
    inspection = _inspect_local_model_path(path)
    if inspection["complete"]:
        cache_state = "cached"
    elif inspection["exists"]:
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
        "snapshot_count": 1 if inspection["exists"] else 0,
        "has_model_weights": inspection["has_model_weights"],
        "has_config": inspection["has_config"],
        "incomplete_file_count": 0,
        "active_job": None,
        "is_local": True,
        "managed_local": bool(model.get("managed_local")),
    }


def _inspect_local_model_path(path: Path) -> dict[str, Any]:
    try:
        path_exists = path.exists()
    except (OSError, ValueError):
        path_exists = False
    if not path_exists:
        return {
            "exists": False,
            "size_bytes": 0,
            "has_config": False,
            "has_model_weights": False,
            "complete": False,
        }
    try:
        snapshot = _snapshot_is_complete(path)
        return {
            "exists": True,
            "size_bytes": _directory_size(path),
            "has_config": snapshot["has_config"],
            "has_model_weights": snapshot["has_model_weights"],
            "complete": snapshot["complete"],
        }
    except (OSError, ValueError):
        return {
            "exists": True,
            "size_bytes": 0,
            "has_config": False,
            "has_model_weights": False,
            "complete": False,
        }


def _snapshot_is_complete(snapshot: Path) -> dict[str, bool]:
    has_config = (snapshot / "config.json").exists()
    has_weights = any(_is_transformer_weight_file(item) for item in snapshot.rglob("*"))
    return {
        "has_config": has_config,
        "has_model_weights": has_weights,
        "complete": has_config and has_weights,
    }


def _is_transformer_weight_file(path: Path) -> bool:
    if not path.is_file():
        return False
    filename = path.name
    if filename in {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }:
        return True
    if filename.startswith(("model-", "pytorch_model-")) and filename.endswith(
        (".safetensors", ".bin")
    ):
        return True
    return False


def _jobs_dir() -> Path:
    path = settings.data_dir / "model_cache_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    local_store.write_json(f"model_cache_jobs/{job['job_id']}.json", job)


def _read_job(job_id: str) -> dict[str, Any] | None:
    try:
        return local_store.read_json(f"model_cache_jobs/{job_id}.json")
    except FileNotFoundError:
        return None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in _jobs_dir().glob("model_*.json"):
        job = _read_job(path.stem)
        if job is not None:
            jobs.append(job)
    return jobs


model_cache_service = ModelCacheService()
