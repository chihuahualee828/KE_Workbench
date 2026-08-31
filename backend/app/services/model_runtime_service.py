from __future__ import annotations

import gc
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import zipfile

from app.ml.adapters.gpt2_adapter import adapter_runtime_lock, get_gpt2_adapter, unload_all_adapters
from app.services.model_cache_service import model_cache_service
from app.services.model_registry import model_registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


ACTIVE_RUNTIME_STATUSES = {"queued", "loading"}


class ModelNotLoadedError(RuntimeError):
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
        adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
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
        with self._lock:
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
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass
            status = self.runtime_status(model_id)
            status["memory"] = _runtime_memory()
            status["discarded_patches_by_model"] = discarded_patches_by_model
            self._publish_runtime_update()
            return status

    def unload_all(self) -> None:
        with adapter_runtime_lock():
            unload_all_adapters()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass
        self._publish_runtime_update()

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
        adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
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
            adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
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
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "loading"
            job["message"] = "Loading model weights onto GPU if available"
            job["updated_at"] = _now()
            model_id = job["model_id"]
            self._runtime_version += 1
            self._condition.notify_all()

        try:
            model = model_registry.get(model_id)
            adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
            local_files_only = model_cache_service.cache_status(model_id)["cached"]
            adapter.load_tokenizer(local_files_only=local_files_only)
            adapter.load_model(local_files_only=local_files_only)
            device = adapter.model_device() or "unknown device"
            with self._lock:
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
            with self._lock:
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
            adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
            if adapter.is_model_loaded():
                loaded_model_ids.append(model["id"])
        return loaded_model_ids


model_runtime_service = ModelRuntimeService()


def _runtime_memory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
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
