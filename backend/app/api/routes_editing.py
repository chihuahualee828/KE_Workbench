from __future__ import annotations

import contextlib
import json
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.schemas import MemitEditRequest, RomeEditRequest, JobResponse, PatchListResponse
from app.ml.editing.easyedit_runtime import EasyEditRuntimeError, EasyEditUnsupportedError
from app.ml.editing.memit_wrapper import run_memit_edit
from app.ml.editing.rome_wrapper import run_rome_edit
from app.services.experiment_service import experiment_service
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service
from app.services.patch_service import patch_service

router = APIRouter(tags=["editing"])
_edit_job_lock = threading.Lock()


class PatchExportRequest(BaseModel):
    output_path: str


@router.post("/edit/rome", response_model=JobResponse)
def rome_edit(request: RomeEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    experiment = experiment_service.create("rome_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("ROME", job["id"], run_rome_edit, request.model_dump())
    return _job_response(job)


@router.post("/edit/memit", response_model=JobResponse)
def memit_edit(request: MemitEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
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
    return _job_response(job)


def _require_loaded_model(model_id: str) -> None:
    try:
        model_runtime_service.require_loaded_runtime_bundle(model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _start_edit_thread(method_label: str, job_id: str, runner, payload: dict) -> None:
    thread = threading.Thread(
        target=_run_edit_job,
        args=(method_label, job_id, runner, payload),
        daemon=True,
    )
    thread.start()


def _run_edit_job(method_label: str, job_id: str, runner, payload: dict) -> None:
    experiment_service.append_job_log(job_id, f"Starting {method_label} edit")
    if _edit_job_lock.locked():
        experiment_service.append_job_log(job_id, "Waiting for active edit job to finish")
    try:
        with _edit_job_lock:
            with _capture_job_output(job_id):
                result = runner(**payload)
    except EasyEditUnsupportedError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except EasyEditRuntimeError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except Exception as exc:
        experiment_service.fail_job(job_id, str(exc))
    else:
        experiment_service.complete_job(job_id, result)


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
    def __init__(self, job_id: str, original) -> None:
        self.job_id = job_id
        self.original = original
        self.buffer = ""

    def write(self, text: str) -> int:
        self.original.write(text)
        self.original.flush()
        self.buffer += text.replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        if self.buffer.strip():
            self._emit(self.buffer)
            self.buffer = ""

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())

    def _emit(self, line: str) -> None:
        clean = line.strip()
        if clean:
            experiment_service.append_job_log(self.job_id, clean)


def _job_response(job: dict) -> JobResponse:
    return JobResponse(
        job_id=job["id"],
        experiment_id=job["experiment_id"],
        status=job["status"],
        progress=job.get("progress", 0.0),
        logs=job.get("logs", []),
        result=_job_result(job),
    )


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
