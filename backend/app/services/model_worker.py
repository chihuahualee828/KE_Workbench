from __future__ import annotations

import contextlib
import traceback
from typing import Any


class _WorkerLogStream:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def write(self, value: str) -> int:
        if value:
            self._connection.send({"type": "log", "text": value})
        return len(value)

    def flush(self) -> None:
        return None


def model_worker_entry(connection: Any) -> None:
    """Run model-owning commands in a disposable CUDA process."""
    import app.services.model_runtime_service as runtime_module

    runtime = runtime_module.ModelRuntimeService()
    runtime_module.model_runtime_service = runtime
    connection.send({"type": "ready"})

    try:
        while True:
            command = connection.recv()
            action = command.get("action")
            if action == "shutdown":
                connection.send({"type": "result", "ok": True, "result": {"stopped": True}})
                break
            try:
                with contextlib.redirect_stdout(_WorkerLogStream(connection)), contextlib.redirect_stderr(
                    _WorkerLogStream(connection)
                ):
                    result = _dispatch(runtime, action, command.get("payload") or {})
            except Exception as exc:
                connection.send(
                    {
                        "type": "result",
                        "ok": False,
                        "error_type": "value_error" if isinstance(exc, ValueError) else "runtime_error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            else:
                connection.send({"type": "result", "ok": True, "result": result})
    finally:
        runtime.unload_all()
        connection.close()


def _dispatch(runtime: Any, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    if action == "load":
        job = runtime.start_load(payload["model_id"])
        completed = job
        while completed["status"] in {"queued", "loading"}:
            completed = runtime.wait_job(job["job_id"], timeout=120)
            if completed is None:
                raise RuntimeError("Model load job disappeared before completion.")
        if completed is None or completed["status"] != "completed":
            raise RuntimeError((completed or job).get("message", "Model load failed."))
        return runtime.runtime_status(payload["model_id"])
    if action == "status":
        return runtime.runtime_status(payload["model_id"])
    if action == "sync_patches":
        runtime.sync_loaded_model_to_active_patches(payload["model_id"])
        return {"model_id": payload["model_id"], "synced": True}
    if action == "export_model":
        return runtime.export_active_model(payload["model_id"], payload["output_path"])
    if action == "export_model_archive":
        archive_path, temporary_root = runtime.export_active_model_archive(payload["model_id"])
        return {"archive_path": str(archive_path), "temporary_root": str(temporary_root)}
    if action == "generate":
        from app.ml.inference.generate import generate_text

        return generate_text(**payload)
    if action == "logit_lens":
        from app.ml.lens.logit_lens import run_logit_lens

        return run_logit_lens(**payload)
    if action == "causal_trace":
        from app.ml.tracing.causal_trace import run_causal_trace

        return run_causal_trace(**payload)
    if action == "rome_edit":
        from app.ml.editing.rome_wrapper import run_rome_edit

        return run_rome_edit(**payload)
    if action == "r_rome_edit":
        from app.ml.editing.r_rome_wrapper import run_r_rome_edit

        return run_r_rome_edit(**payload)
    if action == "rome_sequential_edits":
        from app.ml.editing.rome_wrapper import run_rome_sequential_edits

        return run_rome_sequential_edits(**payload)
    if action == "memit_edit":
        from app.ml.editing.memit_wrapper import run_memit_edit

        return run_memit_edit(**payload)
    if action == "kn_edit":
        from app.ml.editing.kn_wrapper import run_kn_edit

        return run_kn_edit(**payload)
    if action == "prepare_mom2":
        from app.ml.editing.easyedit_runtime import prepare_mom2_statistics

        return prepare_mom2_statistics(**payload)
    raise ValueError(f"Unsupported model worker action: {action}")
