from __future__ import annotations

import contextlib
import json
import re
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.db.schemas import KnEditRequest, MemitEditRequest, Mom2StatisticsRequest, RRomeEditRequest, RomeEditRequest, RomeSequentialEditsRequest, JobResponse, PatchListResponse
from app.ml.editing.easyedit_runtime import (
    EasyEditRuntimeError,
    EasyEditUnsupportedError,
    get_easyedit_default_layers,
    import_mom2_statistics,
    list_mom2_statistics,
    get_mom2_statistics_status,
    prepare_mom2_statistics,
    require_mom2_statistics,
    delete_mom2_statistics,
)
from app.ml.editing.memit_wrapper import run_memit_edit
from app.ml.editing.kn_wrapper import run_kn_edit
from app.ml.editing.rome_wrapper import run_rome_edit, run_rome_sequential_edits
from app.ml.editing.r_rome_wrapper import run_r_rome_edit
from app.services.experiment_service import experiment_service
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service
from app.services.patch_service import patch_service

router = APIRouter(tags=["editing"])
_edit_job_lock = threading.Lock()
_active_edit_job_id: str | None = None
_active_edit_job_lock = threading.Lock()


class PatchExportRequest(BaseModel):
    output_path: str


class EditDefaultLayersRequest(BaseModel):
    model_id: str
    method: str


@router.post("/edit/default-layers")
def edit_default_layers(request: EditDefaultLayersRequest) -> dict:
    try:
        return get_easyedit_default_layers(request.model_id, request.method)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/edit/rome", response_model=JobResponse)
def rome_edit(request: RomeEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _require_mom2_statistics(request.model_id, "rome", [request.target_layer] if request.target_layer is not None else None, request.mom2_adjustment)
    _prepare_edit_patch_stack(request.model_id, request.sequential_edit)
    experiment = experiment_service.create("rome_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("ROME", job["id"], run_rome_edit, request.model_dump())
    return _job_response(job)


@router.post("/edit/r-rome", response_model=JobResponse)
def r_rome_edit(request: RRomeEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _require_mom2_statistics(
        request.model_id,
        "r_rome",
        [request.target_layer] if request.target_layer is not None else None,
        request.mom2_adjustment,
    )
    _prepare_edit_patch_stack(request.model_id, request.sequential_edit)
    experiment = experiment_service.create("r_rome_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("R-ROME", job["id"], run_r_rome_edit, request.model_dump())
    return _job_response(job)


@router.post("/edit/rome/sequential", response_model=JobResponse)
def rome_sequential_edits(request: RomeSequentialEditsRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _require_mom2_statistics(request.model_id, "rome", [request.target_layer], request.mom2_adjustment)
    experiment = experiment_service.create("rome_sequential_edits", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("ROME sequential series", job["id"], run_rome_sequential_edits, request.model_dump())
    return _job_response(job)


@router.post("/edit/kn", response_model=JobResponse)
def kn_edit(request: KnEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _prepare_edit_patch_stack(request.model_id, request.sequential_edit)
    experiment = experiment_service.create("kn_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("KN", job["id"], run_kn_edit, request.model_dump())
    return _job_response(job)


@router.post("/edit/memit", response_model=JobResponse)
def memit_edit(request: MemitEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _require_mom2_statistics(request.model_id, "memit", request.layers, request.mom2_adjustment)
    _prepare_edit_patch_stack(request.model_id, request.sequential_edit)
    experiment = experiment_service.create("memit_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("MEMIT", job["id"], run_memit_edit, request.model_dump())
    return _job_response(job)


@router.get("/edit/jobs/{job_id}", response_model=JobResponse)
def get_edit_job(job_id: str) -> JobResponse:
    job = experiment_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Edit job not found")
    return _job_response(job)


@router.get("/edit/jobs/{job_id}/wait", response_model=JobResponse)
def wait_edit_job(
    job_id: str,
    after_log_count: int = Query(default=0, ge=0),
    timeout: float = Query(default=30.0, ge=1.0, le=120.0),
) -> JobResponse:
    job = experiment_service.wait_job(job_id, after_log_count=after_log_count, timeout=timeout)
    if job is None:
        raise HTTPException(status_code=404, detail="Edit job not found")
    return _job_response(job, after_log_count=after_log_count)


@router.get("/edit/jobs/{job_id}/events")
def stream_edit_job(job_id: str) -> StreamingResponse:
    def events():
        after_log_count = 0
        for job in experiment_service.job_events(job_id):
            response = _job_response(job, after_log_count=after_log_count)
            after_log_count = response.log_count
            yield f"event: job\ndata: {response.model_dump_json()}\n\n"
        yield "event: complete\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/edit/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_edit_job(job_id: str) -> JobResponse:
    current_job = experiment_service.get_job(job_id)
    was_running = current_job is not None and current_job.get("status") == "running"
    job = experiment_service.cancel_job(job_id, "Edit canceled by user. The loaded model was unloaded.")
    if job is None:
        raise HTTPException(status_code=404, detail="Edit job not found")
    with _active_edit_job_lock:
        is_active = _active_edit_job_id == job_id
    if was_running and job.get("status") == "canceled" and is_active:
        threading.Thread(target=_stop_canceled_edit_worker, args=(job_id,), daemon=True).start()
    return _job_response(job)


def _stop_canceled_edit_worker(job_id: str) -> None:
    """Release the worker after responding to a cancellation request."""
    try:
        model_runtime_service.cancel_active_operation("Edit canceled by user.")
    except Exception as exc:
        experiment_service.append_job_log(job_id, f"Model-worker shutdown warning: {exc}")


@router.post("/edit/mom2/status")
def mom2_statistics_status(request: Mom2StatisticsRequest) -> dict:
    try:
        return get_mom2_statistics_status(request.model_id, request.method, request.layers)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/edit/mom2/prepare", response_model=JobResponse)
def prepare_mom2(request: Mom2StatisticsRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    experiment = experiment_service.create("prepare_mom2_statistics", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("MOM2 statistics", job["id"], prepare_mom2_statistics, request.model_dump())
    return _job_response(job)


@router.post("/edit/mom2/import")
def import_mom2(
    model_id: str = Form(...),
    method: str = Form(...),
    dataset: str = Form(...),
    layers: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict:
    try:
        if method not in {"rome", "r_rome", "memit"}:
            raise ValueError("Method must be rome, r_rome, or memit.")
        parsed_layers = json.loads(layers)
        if not isinstance(parsed_layers, list) or not all(isinstance(layer, int) for layer in parsed_layers):
            raise ValueError("Layers must be a JSON array of integers.")
        return import_mom2_statistics(
            model_id,
            method,
            dataset,
            parsed_layers,
            [(file.filename or "upload.npz", file.file) for file in files],
        )
    except (EasyEditUnsupportedError, EasyEditRuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/edit/mom2/cache")
def list_mom2_cache(request: Mom2StatisticsRequest) -> dict:
    try:
        return list_mom2_statistics(request.model_id, request.method)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/edit/mom2/cache")
def delete_mom2_cache(request: Mom2StatisticsRequest) -> dict:
    if _edit_job_lock.locked():
        raise HTTPException(status_code=409, detail="Wait for the active edit or statistics job before deleting cached statistics.")
    try:
        return delete_mom2_statistics(request.model_id, request.method, request.layers or [])
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_loaded_model(model_id: str) -> None:
    try:
        model_runtime_service.require_loaded_runtime_bundle(model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _require_mom2_statistics(model_id: str, method: str, layers: list[int] | None, mom2_adjustment: bool) -> None:
    try:
        require_mom2_statistics(model_id, method, layers, mom2_adjustment)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _prepare_edit_patch_stack(model_id: str, continue_from_active_patches: bool) -> None:
    """Synchronously reset reloadable patches before a fresh edit job starts."""
    if continue_from_active_patches:
        return

    previous_active_state = patch_service.active_patches_by_model()
    patch_service.deactivate_runtime_patches_for_model(model_id)
    try:
        model_runtime_service.sync_loaded_model_to_active_patches(model_id)
    except Exception as exc:
        patch_service.restore_active_patches(previous_active_state)
        try:
            model_runtime_service.sync_loaded_model_to_active_patches(model_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=503,
            detail="Could not reset active patches before starting the edit.",
        ) from exc


def _start_edit_thread(method_label: str, job_id: str, runner, payload: dict) -> None:
    thread = threading.Thread(
        target=_run_edit_job,
        args=(method_label, job_id, runner, payload),
        daemon=True,
    )
    thread.start()


def _run_edit_job(method_label: str, job_id: str, runner, payload: dict) -> None:
    global _active_edit_job_id
    try:
        with _edit_job_lock:
            current_job = experiment_service.get_job(job_id)
            if current_job is None or current_job.get("status") != "running":
                return
            experiment_service.append_job_log(job_id, f"Starting {method_label} edit")
            experiment_service.set_job_progress(job_id, 0.05)
            with _active_edit_job_lock:
                _active_edit_job_id = job_id
            try:
                with _capture_job_output(job_id):
                    result = runner(**payload)
                experiment_service.complete_job(job_id, result)
            finally:
                with _active_edit_job_lock:
                    if _active_edit_job_id == job_id:
                        _active_edit_job_id = None
    except EasyEditUnsupportedError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except EasyEditRuntimeError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except Exception as exc:
        experiment_service.fail_job(job_id, str(exc))
@contextlib.contextmanager
def _capture_job_output(job_id: str):
    stdout = _JobLogStream(job_id, sys.stdout)
    stderr = _JobLogStream(job_id, sys.stderr)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()


class _JobLogStream:
    """Capture third-party edit output for the job record without terminal teeing."""

    def __init__(self, job_id: str, terminal_stream) -> None:
        self.job_id = job_id
        self.terminal_stream = terminal_stream
        self.buffer = ""

    def write(self, text: str) -> int:
        if settings.edit_log_to_terminal:
            self.terminal_stream.write(text)
            self.terminal_stream.flush()
        self.buffer += text.replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if settings.edit_log_to_terminal:
            self.terminal_stream.flush()
        if self.buffer.strip():
            self._emit(self.buffer)
            self.buffer = ""

    def isatty(self) -> bool:
        return settings.edit_log_to_terminal and bool(getattr(self.terminal_stream, "isatty", lambda: False)())

    def _emit(self, line: str) -> None:
        clean = line.strip()
        if clean:
            series_progress = _sequential_edit_progress(clean)
            completed, total = series_progress if series_progress is not None else (None, None)
            edit_started = _easyedit_edit_started(clean)
            experiment_service.append_job_log(
                self.job_id,
                clean,
                progress=(completed / total if completed is not None and total else 0.1 if edit_started else None),
                progress_current=completed,
                progress_total=total,
            )


def _job_response(job: dict, after_log_count: int | None = None) -> JobResponse:
    total_log_count = int(job.get("log_count", len(job.get("logs", []))))
    retained_logs = job.get("logs", [])
    retained_start = total_log_count - len(retained_logs)
    logs = retained_logs
    log_start_count = retained_start
    if after_log_count is not None:
        log_start_count = max(min(after_log_count, total_log_count), retained_start)
        logs = retained_logs[log_start_count - retained_start :]
    return JobResponse(
        job_id=job["id"],
        experiment_id=job["experiment_id"],
        status=job["status"],
        progress=job.get("progress", 0.0),
        progress_current=job.get("progress_current"),
        progress_total=job.get("progress_total"),
        logs=logs,
        log_count=total_log_count,
        log_start_count=log_start_count,
        result=_job_result(job),
    )


def _sequential_edit_progress(log_line: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"Created (?:R-)?ROME sequential edit (\d+)/(\d+)\.", log_line)
    if match is None:
        return None
    completed, total = (int(value) for value in match.groups())
    return (completed, total) if total else None


def _easyedit_edit_started(log_line: str) -> bool:
    return log_line == "Starting EasyEdit algorithm."


def _job_result(job: dict) -> dict | None:
    if job.get("status") != "completed":
        return None
    experiment = experiment_service.get_experiment(job["experiment_id"])
    result_path = experiment.get("result_path") if experiment else None
    if not result_path:
        return None
    path = Path(result_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/patches", response_model=PatchListResponse)
def list_patches() -> PatchListResponse:
    return PatchListResponse(patches=patch_service.list_patches())


@router.post("/patches/{patch_id}/export")
def export_patch(patch_id: str, request: PatchExportRequest) -> dict:
    try:
        return patch_service.export_patch_bundle(patch_id, request.output_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc


@router.post("/patches/{patch_id}/load")
def load_patch(patch_id: str) -> dict:
    try:
        patch = patch_service.get_patch(patch_id)
        model_id = patch["base_model"]
        if not model_runtime_service.runtime_status(model_id)["loaded"]:
            raise HTTPException(
                status_code=409,
                detail=f"Load {model_id} before loading this saved patch.",
            )
        previous_active_state = patch_service.active_patches_by_model()
        state = patch_service.load_saved_patch(patch_id)
        try:
            model_runtime_service.sync_loaded_model_to_active_patches(model_id)
        except Exception:
            patch_service.restore_active_patches(previous_active_state)
            model_runtime_service.sync_loaded_model_to_active_patches(model_id)
            raise
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/patches/{patch_id}/discard")
def discard_patch(patch_id: str) -> dict:
    try:
        state = patch_service.discard_active_patch(patch_id)
        model_runtime_service.sync_loaded_model_to_active_patches(state["model_id"])
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/patches/{patch_id}")
def delete_patch(patch_id: str) -> dict:
    try:
        return patch_service.delete_patch_artifact(patch_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/patches")
def delete_all_patches() -> dict:
    state = patch_service.deactivate_all_patches()
    try:
        for model_id in state["model_ids"]:
            model_runtime_service.sync_loaded_model_to_active_patches(model_id)
    except Exception:
        patch_service.restore_active_patches(state["active_patches_by_model"])
        for model_id in state["model_ids"]:
            model_runtime_service.sync_loaded_model_to_active_patches(model_id)
        raise
    return patch_service.delete_all_patch_artifacts()


@router.post("/patches/{patch_id}/revert-to")
def revert_to_patch(patch_id: str) -> dict:
    try:
        state = patch_service.revert_to_patch(patch_id)
        model_runtime_service.sync_loaded_model_to_active_patches(state["model_id"])
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
