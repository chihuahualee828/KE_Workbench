from __future__ import annotations

import contextlib
import gc
import json
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ml.adapters.gpt2_adapter import adapter_runtime_lock, get_gpt2_adapter, unload_all_adapters, unload_other_adapters
from app.services.model_cache_service import model_cache_service
from app.services.model_registry import model_registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


ACTIVE_RUNTIME_STATUSES = {"queued", "loading"}


class ModelNotLoadedError(RuntimeError):
    pass


class ModelWorkerError(RuntimeError):
    """A model worker command failed before producing an application result."""

    pass


class ModelRuntimeService:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._runtime_version = 0

    def list_runtime_statuses(self) -> list[dict[str, Any]]:
        return [self.runtime_status(model["id"]) for model in model_registry.list_models()]

    def runtime_status(self, model_id: str) -> dict[str, Any]:
        model = model_registry.get(model_id)
        adapter = get_gpt2_adapter(
            model_id=model["hf_id"],
            n_layers=model["n_layers"],
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            generation_use_cache=bool(model.get("generation_use_cache", True)),
        )
        active_job = self._active_job_for_model(model_id)
        return {
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "loaded": adapter.is_model_loaded(),
            "tokenizer_loaded": adapter.is_tokenizer_loaded(),
            "device": adapter.model_device() or "not_loaded",
            "model_vram_bytes": adapter.model_vram_bytes(),
            "gpu": _runtime_memory(),
            "runtime_state": active_job["status"] if active_job else ("loaded" if adapter.is_model_loaded() else "not_loaded"),
            "active_job": active_job,
        }

    def start_load(self, model_id: str) -> dict[str, Any]:
        model = model_registry.get(model_id)
        status = self.runtime_status(model_id)
        if status["loaded"]:
            return self._completed_job(model_id, model["hf_id"], "Model is already loaded")
        active_job = self._active_job_for_model(model_id)
        if active_job:
            return active_job

        job_id = f"model_load_{uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "status": "queued",
            "message": "Queued",
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._condition:
            self._jobs[job_id] = job
            self._runtime_version += 1
            self._condition.notify_all()

        thread = threading.Thread(target=self._load_model, args=(job_id,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def wait_job(self, job_id: str, timeout: float = 60.0) -> dict[str, Any] | None:
        timeout = min(max(timeout, 1.0), 120.0)
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                if job["status"] not in ACTIVE_RUNTIME_STATUSES:
                    return dict(job)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return dict(job)
                self._condition.wait(timeout=remaining)

    def unload(self, model_id: str) -> dict[str, Any]:
        model_registry.get(model_id)
        with adapter_runtime_lock():
            loaded_model_ids = self._loaded_model_ids()
            unload_all_adapters()
            from app.services.patch_service import patch_service

            discarded_patches_by_model = {
                loaded_model_id: patch_service.deactivate_runtime_patches_for_model(loaded_model_id)
                for loaded_model_id in loaded_model_ids
            }
            gc.collect()
            cleanup = _release_cuda_allocator()
            print("Model runtime unloaded and GPU memory cleared.")
            status = self.runtime_status(model_id)
            status["memory"] = _runtime_memory()
            status["cuda_cleanup"] = cleanup
            status["discarded_patches_by_model"] = discarded_patches_by_model
        self._publish_runtime_update()
        return status

    def unload_all(self) -> None:
        with adapter_runtime_lock():
            unload_all_adapters()
            gc.collect()
            _release_cuda_allocator()
        self._publish_runtime_update()

    def cancel_active_operation(self, _reason: str) -> dict[str, Any]:
        """The in-process runtime cannot safely interrupt arbitrary PyTorch work."""
        return {"worker_stopped": False}

    def runtime_status_events(self):
        """Yield state changes and keepalives for SSE runtime-status subscribers."""
        observed_version = -1
        while True:
            with self._condition:
                if observed_version == self._runtime_version:
                    self._condition.wait(timeout=15.0)
                changed = observed_version != self._runtime_version
                observed_version = self._runtime_version
            if changed:
                yield {"type": "runtime", "statuses": self.list_runtime_statuses()}
            else:
                yield {"type": "keepalive"}

    def loaded_runtime_bundle(self, model_id: str) -> dict[str, Any] | None:
        model = model_registry.get(model_id)
        adapter = get_gpt2_adapter(
            model_id=model["hf_id"],
            n_layers=model["n_layers"],
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            generation_use_cache=bool(model.get("generation_use_cache", True)),
        )
        loaded_model = adapter.loaded_model()
        if loaded_model is None:
            return None

        local_files_only = model_cache_service.cache_status(model_id)["cached"]
        tokenizer = adapter.loaded_tokenizer() or adapter.load_tokenizer(local_files_only=local_files_only)
        from app.services.patch_service import patch_service

        patch_service.sync_active_patches_to_model(loaded_model, model_id)
        return {
            "model": loaded_model,
            "tokenizer": tokenizer,
            "device": adapter.model_device(),
        }

    def require_loaded_runtime_bundle(self, model_id: str) -> dict[str, Any]:
        """Return the active runtime model without ever loading model weights."""
        bundle = self.loaded_runtime_bundle(model_id)
        if bundle is None:
            raise ModelNotLoadedError(
                f"Load {model_id} into the model runtime before using this operation."
            )
        return bundle

    def sync_loaded_model_to_active_patches(self, model_id: str) -> None:
        with adapter_runtime_lock():
            model = model_registry.get(model_id)
            adapter = get_gpt2_adapter(
                model_id=model["hf_id"],
                n_layers=model["n_layers"],
                trust_remote_code=bool(model.get("trust_remote_code", False)),
                generation_use_cache=bool(model.get("generation_use_cache", True)),
            )
            loaded_model = adapter.loaded_model()
            if loaded_model is None:
                return
            from app.services.patch_service import patch_service

            patch_service.sync_active_patches_to_model(loaded_model, model_id)

    def export_active_model(self, model_id: str, output_path: str) -> dict[str, Any]:
        model_registry.get(model_id)
        from app.services.patch_service import patch_service

        export_path = Path(output_path).expanduser().resolve()
        export_path.mkdir(parents=True, exist_ok=True)
        with adapter_runtime_lock():
            bundle = self.require_loaded_runtime_bundle(model_id)
            model = bundle["model"]
            tokenizer = bundle["tokenizer"]
            try:
                model.save_pretrained(str(export_path), safe_serialization=True)
            except TypeError:
                model.save_pretrained(str(export_path))
            tokenizer.save_pretrained(str(export_path))

            # A full model export intentionally contains no workbench provenance.
            # The edited weights are baked into standard Hugging Face model files.
            (export_path / "kew_lineage.json").unlink(missing_ok=True)
        return {
            "model_id": model_id,
            "output_path": str(export_path),
            "baked_patches": len(patch_service.active_patches_for_model(model_id)),
        }

    def export_active_model_archive(self, model_id: str) -> tuple[Path, Path]:
        """Build a standard Hugging Face model archive for browser download.

        The returned archive and its containing temporary directory must be removed
        after the HTTP response has finished streaming.
        """
        model_registry.get(model_id)
        safe_model_id = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in model_id
        ).strip(".") or "edited-model"
        temporary_root = Path(tempfile.mkdtemp(prefix="kew-model-export-"))
        model_path = temporary_root / safe_model_id
        archive_path = temporary_root / f"{safe_model_id}-edited.zip"

        try:
            with adapter_runtime_lock():
                bundle = self.require_loaded_runtime_bundle(model_id)
                model = bundle["model"]
                tokenizer = bundle["tokenizer"]
                try:
                    model.save_pretrained(str(model_path), safe_serialization=True)
                except TypeError:
                    model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))

            # Use stored entries: model checkpoints are already compressed, and this
            # keeps archive creation fast without duplicating large buffers in RAM.
            with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for source_path in model_path.rglob("*"):
                    if source_path.is_file():
                        archive.write(source_path, source_path.relative_to(temporary_root))
        except Exception:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise

        return archive_path, temporary_root

    def exclusive_runtime_access(self):
        return adapter_runtime_lock()

    def _load_model(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job["status"] = "loading"
            job["message"] = "Loading model weights onto GPU if available"
            job["updated_at"] = _now()
            model_id = job["model_id"]
            self._runtime_version += 1
            self._condition.notify_all()

        try:
            model = model_registry.get(model_id)
            adapter = get_gpt2_adapter(
                model_id=model["hf_id"],
                n_layers=model["n_layers"],
                trust_remote_code=bool(model.get("trust_remote_code", False)),
                generation_use_cache=bool(model.get("generation_use_cache", True)),
            )
            local_files_only = model_cache_service.cache_status(model_id)["cached"]
            adapter.load_tokenizer(local_files_only=local_files_only)
            with adapter_runtime_lock():
                replaced_model_ids = [loaded_model_id for loaded_model_id in self._loaded_model_ids() if loaded_model_id != model_id]
                if replaced_model_ids:
                    unload_other_adapters(adapter)
                    from app.services.patch_service import patch_service

                    for replaced_model_id in replaced_model_ids:
                        patch_service.deactivate_runtime_patches_for_model(replaced_model_id)
                adapter.load_model(local_files_only=local_files_only)
            device = adapter.model_device() or "unknown device"
            with self._condition:
                self._jobs[job_id].update(
                    {
                        "status": "completed",
                        "message": f"Model is loaded on {device}",
                        "device": device,
                        "updated_at": _now(),
                    }
                )
                self._runtime_version += 1
                self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._jobs[job_id].update(
                    {
                        "status": "failed",
                        "message": str(exc),
                        "updated_at": _now(),
                    }
                )
                self._runtime_version += 1
                self._condition.notify_all()

    def _publish_runtime_update(self) -> None:
        with self._condition:
            self._runtime_version += 1
            self._condition.notify_all()

    def _active_job_for_model(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            jobs = [
                dict(job)
                for job in self._jobs.values()
                if job["model_id"] == model_id and job["status"] in ACTIVE_RUNTIME_STATUSES
            ]
        if not jobs:
            return None
        jobs.sort(key=lambda item: item["updated_at"], reverse=True)
        return jobs[0]

    def _completed_job(self, model_id: str, hf_id: str, message: str) -> dict[str, Any]:
        return {
            "job_id": f"model_load_{uuid4().hex[:10]}",
            "model_id": model_id,
            "hf_id": hf_id,
            "status": "completed",
            "message": message,
            "created_at": _now(),
            "updated_at": _now(),
        }

    def _loaded_model_ids(self) -> list[str]:
        loaded_model_ids: list[str] = []
        for model in model_registry.list_models():
            adapter = get_gpt2_adapter(
                model_id=model["hf_id"],
                n_layers=model["n_layers"],
                trust_remote_code=bool(model.get("trust_remote_code", False)),
                generation_use_cache=bool(model.get("generation_use_cache", True)),
            )
            if adapter.is_model_loaded():
                loaded_model_ids.append(model["id"])
        return loaded_model_ids


def _runtime_memory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        from app.ml.device import mps_memory

        return mps_memory()
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    smi_memory = _nvidia_smi_memory(device)
    used_bytes = int(total_bytes - free_bytes)
    memory_source = "torch"
    if smi_memory:
        total_bytes = smi_memory["total_bytes"]
        used_bytes = smi_memory["used_bytes"]
        free_bytes = max(0, total_bytes - used_bytes)
        memory_source = "nvidia-smi"
    return {
        "cuda_available": True,
        "device": str(device),
        "device_name": smi_memory.get("device_name") if smi_memory else torch.cuda.get_device_name(device),
        "memory_source": memory_source,
        "used_bytes": int(used_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _cuda_allocator_memory() -> dict[str, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _release_cuda_allocator() -> dict[str, Any] | None:
    """Run a full PyTorch allocator cleanup and report this process's memory change."""
    before = _cuda_allocator_memory()
    if before is None:
        return None
    try:
        import torch

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    except RuntimeError as exc:
        print(f"CUDA allocator cleanup failed: {exc}")
        return {"before": before, "after": _cuda_allocator_memory(), "error": str(exc)}
    return {"before": before, "after": _cuda_allocator_memory()}


def _nvidia_smi_memory(device: int) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        used_mib = int(parts[1])
        total_mib = int(parts[2])
    except ValueError:
        return None
    return {
        "device_name": parts[0],
        "used_bytes": used_mib * 1024 * 1024,
        "total_bytes": total_mib * 1024 * 1024,
    }


class HardUnloadModelRuntimeService:
    """Controller for one disposable process that owns all accelerator model state.

    The FastAPI process deliberately never loads model weights. Stopping this
    worker therefore releases every accelerator allocation made by a loaded model,
    including allocations held by third-party editing code.
    """

    uses_hard_worker = True
    _WORKER_READY_TIMEOUT_SECONDS = 20.0
    _WORKER_STOP_TIMEOUT_SECONDS = 10.0

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._worker_lock = threading.RLock()
        self._termination_lock = threading.Lock()
        self._runtime_version = 0
        self._process: Any | None = None
        self._connection: Any | None = None
        self._loaded_model_id: str | None = None
        self._loaded_device = "not_loaded"
        self._model_vram_bytes = 0
        self._worker_memory: Any | None = None

    def list_runtime_statuses(self) -> list[dict[str, Any]]:
        self._reconcile_worker_liveness()
        return [self.runtime_status(model["id"]) for model in model_registry.list_models()]

    def runtime_status(self, model_id: str) -> dict[str, Any]:
        model = model_registry.get(model_id)
        self._reconcile_worker_liveness()
        active_job = self._active_job_for_model(model_id)
        with self._lock:
            loaded = self._loaded_model_id == model_id and self._worker_is_alive_locked()
            device = self._loaded_device if loaded else "not_loaded"
            model_vram_bytes = self._model_vram_bytes if loaded else 0
        return {
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "loaded": loaded,
            "tokenizer_loaded": loaded,
            "device": device,
            "model_vram_bytes": model_vram_bytes,
            "gpu": self._gpu_memory(),
            "runtime_state": active_job["status"] if active_job else ("loaded" if loaded else "not_loaded"),
            "active_job": active_job,
            "hard_unload": True,
        }

    def _gpu_memory(self) -> dict[str, Any]:
        memory = self._worker_memory
        if memory is not None:
            allocated, driver, recommended, sampled_at = memory[:]
            if sampled_at:
                return {
                    "cuda_available": False, "mps_available": True,
                    "device": "mps", "device_name": "Apple GPU (MPS)",
                    "memory_source": "torch.mps (worker process)",
                    "allocated_bytes": int(allocated),
                    "driver_allocated_bytes": int(driver),
                    "recommended_max_bytes": int(recommended),
                    "sampled_at": sampled_at,
                }
        return _controller_runtime_memory()

    def start_load(self, model_id: str) -> dict[str, Any]:
        model = model_registry.get(model_id)
        status = self.runtime_status(model_id)
        if status["loaded"]:
            return self._completed_job(model_id, model["hf_id"], "Model is already loaded")
        active_job = self._active_job_for_model(model_id)
        if active_job:
            return active_job

        job_id = f"model_load_{uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "model_id": model_id,
            "hf_id": model["hf_id"],
            "status": "queued",
            "message": "Queued",
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._condition:
            self._jobs[job_id] = job
            self._runtime_version += 1
            self._condition.notify_all()
        threading.Thread(target=self._load_model, args=(job_id,), daemon=True).start()
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def wait_job(self, job_id: str, timeout: float = 60.0) -> dict[str, Any] | None:
        timeout = min(max(timeout, 1.0), 120.0)
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None or job["status"] not in ACTIVE_RUNTIME_STATUSES:
                    return dict(job) if job else None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return dict(job)
                self._condition.wait(timeout=remaining)

    def unload(self, model_id: str) -> dict[str, Any]:
        model_registry.get(model_id)
        with self._worker_lock:
            discarded = self._stop_worker_locked("Model runtime unloaded by user.")
        status = self.runtime_status(model_id)
        status["memory"] = _controller_runtime_memory()
        status["hard_unloaded"] = True
        status["discarded_patches_by_model"] = discarded
        self._publish_runtime_update()
        print("Model worker stopped; accelerator model memory was released with its process.")
        return status

    def unload_all(self) -> None:
        with self._worker_lock:
            self._stop_worker_locked("Application is shutting down.")
        self._publish_runtime_update()

    def cancel_active_operation(self, reason: str) -> dict[str, Any]:
        """Terminate a busy model worker without waiting for its command lock.

        Editing runs in the disposable worker, so process termination is the only
        dependable way to interrupt third-party CUDA/PyTorch code mid-operation.
        The loaded model and its active runtime patch stack are consequently
        released along with that process.
        """
        with self._termination_lock:
            with self._lock:
                process = self._process
                connection = self._connection
                loaded_model_id = self._loaded_model_id
                self._process = None
                self._connection = None
                self._worker_memory = None
                self._loaded_model_id = None
                self._loaded_device = "not_loaded"
                self._model_vram_bytes = 0
            if process is None:
                return {"worker_stopped": False}
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self._WORKER_STOP_TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._WORKER_STOP_TIMEOUT_SECONDS)
            discarded_patches: dict[str, list[str]] = {}
            if loaded_model_id:
                from app.services.patch_service import patch_service

                discarded_patches = {
                    loaded_model_id: patch_service.deactivate_runtime_patches_for_model(loaded_model_id)
                }
            self._publish_runtime_update()
            print(f"Model worker stopped: {reason}")
            return {"worker_stopped": True, "discarded_patches_by_model": discarded_patches}

    def shutdown(self) -> None:
        self.unload_all()

    def runtime_status_events(self):
        observed_version = -1
        while True:
            with self._condition:
                if observed_version == self._runtime_version:
                    self._condition.wait(timeout=5.0)
                changed = observed_version != self._runtime_version
                observed_version = self._runtime_version
            self._reconcile_worker_liveness()
            if changed or self._worker_memory is not None:
                yield {"type": "runtime", "statuses": self.list_runtime_statuses()}
            else:
                yield {"type": "keepalive"}

    def loaded_runtime_bundle(self, model_id: str) -> dict[str, Any] | None:
        status = self.runtime_status(model_id)
        if not status["loaded"]:
            return None
        return {
            "model": None,
            "tokenizer": None,
            "device": status["device"],
            "worker_owned": True,
        }

    def require_loaded_runtime_bundle(self, model_id: str) -> dict[str, Any]:
        bundle = self.loaded_runtime_bundle(model_id)
        if bundle is None:
            raise ModelNotLoadedError(
                f"Load {model_id} into the model runtime before using this operation."
            )
        return bundle

    def execute(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float = 3600.0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        from app.ml.editing.edit_cancellation import current_edit

        transaction = current_edit.get()
        if transaction is not None:
            payload = {**payload, "_edit_job_id": transaction.job_id}
        model_id = payload.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("Model worker commands require model_id.")
        self.require_loaded_runtime_bundle(model_id)
        with self._worker_lock:
            self.require_loaded_runtime_bundle(model_id)
            return self._call_worker_locked(action, payload, timeout=timeout, progress_callback=progress_callback)

    def sync_loaded_model_to_active_patches(self, model_id: str) -> None:
        if self.loaded_runtime_bundle(model_id) is None:
            return
        self.execute("sync_patches", {"model_id": model_id}, timeout=120.0)

    def export_active_model(self, model_id: str, output_path: str) -> dict[str, Any]:
        return self.execute(
            "export_model",
            {"model_id": model_id, "output_path": output_path},
            timeout=3600.0,
        )

    def export_active_model_archive(self, model_id: str) -> tuple[Path, Path]:
        result = self.execute("export_model_archive", {"model_id": model_id}, timeout=3600.0)
        return Path(result["archive_path"]), Path(result["temporary_root"])

    @contextlib.contextmanager
    def exclusive_runtime_access(self):
        with self._worker_lock:
            yield

    def _load_model(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.update(
                {
                    "status": "loading",
                    "message": "Starting isolated model worker and loading weights onto GPU if available",
                    "updated_at": _now(),
                }
            )
            self._runtime_version += 1
            self._condition.notify_all()
            model_id = job["model_id"]

        try:
            # A canceled edit may still be terminating its isolated CUDA
            # process. Do not start another worker until that teardown ends.
            with self._termination_lock:
                pass
            with self._worker_lock:
                self._stop_worker_locked("Replacing loaded model.")
                self._start_worker_locked()
                status = self._call_worker_locked("load", {"model_id": model_id}, timeout=3600.0)
                with self._lock:
                    self._loaded_model_id = model_id
                    self._loaded_device = str(status.get("device") or "unknown device")
                    self._model_vram_bytes = int(status.get("model_vram_bytes") or 0)
            message = f"Model is loaded on {self._loaded_device}"
        except Exception as exc:
            with self._worker_lock:
                self._stop_worker_locked("Model load failed.")
            self._finish_job(job_id, "failed", str(exc))
            return

        self._finish_job(job_id, "completed", message, device=self._loaded_device)

    def _finish_job(self, job_id: str, status: str, message: str, **details: Any) -> None:
        with self._condition:
            self._jobs[job_id].update(
                {
                    "status": status,
                    "message": message,
                    "updated_at": _now(),
                    **details,
                }
            )
            self._runtime_version += 1
            self._condition.notify_all()

    def _start_worker_locked(self) -> None:
        from app.services.model_worker import model_worker_entry

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._worker_memory = context.Array("d", 4, lock=False) if sys.platform == "darwin" else None
        process = context.Process(
            target=model_worker_entry,
            args=(child_connection, self._worker_memory),
            name="kew-model-worker",
        )
        process.start()
        child_connection.close()
        self._process = process
        self._connection = parent_connection
        if not parent_connection.poll(self._WORKER_READY_TIMEOUT_SECONDS):
            self._terminate_worker_locked()
            raise ModelWorkerError("Timed out while starting the isolated model worker.")
        try:
            message = parent_connection.recv()
        except EOFError as exc:
            self._terminate_worker_locked()
            raise ModelWorkerError("The isolated model worker exited during startup.") from exc
        if message.get("type") != "ready":
            self._terminate_worker_locked()
            raise ModelWorkerError("The isolated model worker did not initialize correctly.")

    def _call_worker_locked(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        connection = self._connection
        if connection is None or not self._worker_is_alive_locked():
            raise ModelWorkerError("The isolated model worker is not running.")
        try:
            connection.send({"action": action, "payload": payload})
        except (EOFError, OSError) as exc:
            raise ModelWorkerError("The isolated model worker exited unexpectedly.") from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelWorkerError(f"Timed out while running model worker action {action!r}.")
            try:
                has_message = connection.poll(min(0.25, remaining))
            except (EOFError, OSError) as exc:
                raise ModelWorkerError("The isolated model worker exited unexpectedly.") from exc
            if not has_message:
                if not self._worker_is_alive_locked():
                    raise ModelWorkerError("The isolated model worker exited unexpectedly.")
                continue
            try:
                message = connection.recv()
            except (EOFError, OSError) as exc:
                raise ModelWorkerError("The isolated model worker exited unexpectedly.") from exc
            if message.get("type") == "log":
                print(str(message.get("text", "")), end="")
                continue
            if message.get("type") == "progress":
                progress = message.get("progress")
                if isinstance(progress, dict) and progress.get("patch_state_changed"):
                    from app.ml.editing.edit_cancellation import current_edit

                    transaction = current_edit.get()
                    if transaction is not None:
                        transaction.notify_patch_state_changed()
                    continue
                if isinstance(progress, dict) and progress_callback is not None:
                    progress_callback(progress)
                continue
            if message.get("type") != "result":
                continue
            if message.get("ok"):
                return dict(message.get("result") or {})
            error = str(message.get("message") or "Model worker action failed.")
            if message.get("error_type") == "value_error":
                raise ValueError(error)
            raise ModelWorkerError(error)

    def _stop_worker_locked(self, reason: str) -> dict[str, list[str]]:
        with self._lock:
            loaded_model_id = self._loaded_model_id
        if self._process is None:
            return {}
        try:
            if self._worker_is_alive_locked() and self._connection is not None:
                try:
                    self._call_worker_locked("shutdown", {}, timeout=self._WORKER_STOP_TIMEOUT_SECONDS)
                except (ModelWorkerError, EOFError, OSError):
                    pass
            self._terminate_worker_locked()
        finally:
            with self._lock:
                self._loaded_model_id = None
                self._loaded_device = "not_loaded"
                self._model_vram_bytes = 0
        if not loaded_model_id:
            return {}
        from app.services.patch_service import patch_service

        return {loaded_model_id: patch_service.deactivate_runtime_patches_for_model(loaded_model_id)}

    def _terminate_worker_locked(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self._worker_memory = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        process.join(timeout=self._WORKER_STOP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self._WORKER_STOP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=self._WORKER_STOP_TIMEOUT_SECONDS)

    def _reconcile_worker_liveness(self) -> None:
        lost_model_id: str | None = None
        if not self._worker_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                if self._process is None or self._worker_is_alive_locked():
                    return
                lost_model_id = self._loaded_model_id
            self._terminate_worker_locked()
            with self._lock:
                self._loaded_model_id = None
                self._loaded_device = "not_loaded"
                self._model_vram_bytes = 0
            if lost_model_id:
                from app.services.patch_service import patch_service

                patch_service.deactivate_runtime_patches_for_model(lost_model_id)
            self._publish_runtime_update()
        finally:
            self._worker_lock.release()

    def _worker_is_alive_locked(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _publish_runtime_update(self) -> None:
        with self._condition:
            self._runtime_version += 1
            self._condition.notify_all()

    def _active_job_for_model(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            jobs = [
                dict(job)
                for job in self._jobs.values()
                if job["model_id"] == model_id and job["status"] in ACTIVE_RUNTIME_STATUSES
            ]
        return max(jobs, key=lambda item: item["updated_at"]) if jobs else None

    def _completed_job(self, model_id: str, hf_id: str, message: str) -> dict[str, Any]:
        return {
            "job_id": f"model_load_{uuid4().hex[:10]}",
            "model_id": model_id,
            "hf_id": hf_id,
            "status": "completed",
            "message": message,
            "created_at": _now(),
            "updated_at": _now(),
        }


def _controller_runtime_memory() -> dict[str, Any]:
    """Read GPU totals without initializing PyTorch/CUDA in the API process."""
    smi_memory = _nvidia_smi_memory(0)
    if smi_memory is None:
        if sys.platform == "darwin":
            return dict(_mac_gpu_capability())
        return {"cuda_available": False}
    used_bytes = int(smi_memory["used_bytes"])
    total_bytes = int(smi_memory["total_bytes"])
    return {
        "cuda_available": True,
        "device": "0",
        "device_name": smi_memory["device_name"],
        "memory_source": "nvidia-smi",
        "used_bytes": used_bytes,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "free_bytes": max(0, total_bytes - used_bytes),
        "total_bytes": total_bytes,
    }


@lru_cache(maxsize=1)
def _mac_gpu_capability() -> dict[str, Any]:
    """Probe in a short-lived process, never initialize Metal in the API server."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", (
                "import json, torch; "
                "available = torch.backends.mps.is_available(); "
                "print(json.dumps({'mps_available': available, 'recommended_max_bytes': "
                "torch.mps.recommended_max_memory() if available else 0}))"
            )],
            capture_output=True, text=True, timeout=20, check=True,
        )
        capability = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"cuda_available": False, "mps_available": False}
    return {
        **capability, "cuda_available": False,
        "device": "mps", "device_name": "Apple GPU (MPS)",
        "memory_source": "torch.mps (no model worker)",
        "allocated_bytes": 0, "driver_allocated_bytes": 0,
    }


model_runtime_service = HardUnloadModelRuntimeService()
