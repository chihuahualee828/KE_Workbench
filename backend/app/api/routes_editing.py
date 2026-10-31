from __future__ import annotations

import contextlib
import json
import re
import sys
import threading
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Field

from app.config import settings
from app.ml.editing.edit_cancellation import run_cancellable_edit
from app.api.operation_guard import operation_guard
from app.db.schemas import ApiSchema, BatchWeightEditRequest, DpoEditRequest, KnEditRequest, Mom2StatisticsRequest, Mom2StatisticsPrecisionRequest, Mom2StatisticsSourceRequest, RankOneEditRequest, RankOneSeriesRequest, JobResponse, PatchListResponse, TrainingEditRequest
from app.ml.editing.easyedit_runtime import (
    EasyEditRuntimeError,
    EasyEditUnsupportedError,
    evaluate_post_edit_reliability,
    get_easyedit_default_layers,
    import_mom2_statistics,
    list_mom2_statistics,
    get_mom2_statistics_status,
    prepare_mom2_statistics,
    require_mom2_statistics,
    set_mom2_statistics_source,
    set_mom2_statistics_precision,
    delete_mom2_statistics,
)
from app.ml.editing.methods import BATCH_WEIGHT_METHODS
from app.ml.editing.memit_wrapper import run_memit_edit
from app.ml.editing.kn_wrapper import run_kn_edit
from app.ml.editing.rome_wrapper import run_rome_edit, run_rome_sequential_edits
from app.ml.editing.lora_modules import get_lora_modules
from app.ml.editing.training_hparams import get_training_defaults, validate_training_hparams
from app.ml.editing.training_wrapper import run_training_edit
from app.services.experiment_service import experiment_service
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service
from app.services.patch_service import patch_service

router = APIRouter(tags=["editing"])
_edit_job_lock = threading.Lock()
_EDIT_METHODS = {"ROME": "rome", "R-ROME": "r_rome", "ROME sequential series": "rome",
                 "R-ROME sequential series": "r_rome", "KN": "kn", "MEMIT": "memit",
                 "FT": "ft", "LORA": "lora", "DPO": "dpo",
                 "EMMET": "emmet", "MEMIT-FE": "memit_fe", "PMET": "pmet",
                 "EAMET": "eamet", "ALPHAEDIT": "alphaedit", "SPHERE": "sphere"}


class PatchExportRequest(ApiSchema):
    output_path: str


class PatchDeletionPreviewRequest(ApiSchema):
    patch_ids: list[str] = Field(min_length=1, max_length=10000)


class PatchBatchDeleteRequest(PatchDeletionPreviewRequest):
    preview_token: str = Field(min_length=64, max_length=64)


class EditDefaultLayersRequest(ApiSchema):
    model_id: str
    method: str


class LoraModulesRequest(ApiSchema):
    model_id: str


@router.post("/edit/lora/modules")
@operation_guard.protect()
def lora_modules(request: LoraModulesRequest) -> dict:
    try:
        return get_lora_modules(request.model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc



@router.post("/edit/training-defaults")
def training_defaults(request: EditDefaultLayersRequest) -> dict:
    try:
        return get_training_defaults(request.model_id, request.method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc



@router.post("/edit/default-layers")
def edit_default_layers(request: EditDefaultLayersRequest) -> dict:
    try:
        return get_easyedit_default_layers(request.model_id, request.method)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _start_rank_one_job(
    method: str, request: RankOneEditRequest | RankOneSeriesRequest, *, series: bool = False,
) -> JobResponse:
    _require_loaded_model(request.model_id)
    layers = [request.target_layer] if request.target_layer is not None else None
    _require_mom2_statistics(request.model_id, method, layers, request.mom2_adjustment)
    kind = f"{method}_sequential_edits" if series else f"{method}_edit"
    experiment = experiment_service.create(kind, request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    label = method.upper().replace("_", "-") + (" sequential series" if series else "")
    runner = run_rome_sequential_edits if series else run_rome_edit
    _start_edit_thread(label, job["id"], partial(runner, method=method), request.model_dump())
    return _job_response(job)


@router.post("/edit/rome", response_model=JobResponse)
@operation_guard.protect()
def rome_edit(request: RankOneEditRequest) -> JobResponse:
    return _start_rank_one_job("rome", request)


@router.post("/edit/r-rome", response_model=JobResponse)
@operation_guard.protect()
def r_rome_edit(request: RankOneEditRequest) -> JobResponse:
    return _start_rank_one_job("r_rome", request)


@router.post("/edit/rome/sequential", response_model=JobResponse)
@operation_guard.protect()
def rome_sequential_edits(request: RankOneSeriesRequest) -> JobResponse:
    return _start_rank_one_job("rome", request, series=True)


@router.post("/edit/r-rome/sequential", response_model=JobResponse)
@operation_guard.protect()
def r_rome_sequential_edits(request: RankOneSeriesRequest) -> JobResponse:
    return _start_rank_one_job("r_rome", request, series=True)


@router.post("/edit/kn", response_model=JobResponse)
@operation_guard.protect()
def kn_edit(request: KnEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    experiment = experiment_service.create("kn_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("KN", job["id"], run_kn_edit, request.model_dump())
    return _job_response(job)


@router.post("/edit/memit", response_model=JobResponse)
@operation_guard.protect()
def memit_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("memit", request)


def _start_batch_weight_edit(method: str, request: BatchWeightEditRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    _require_mom2_statistics(request.model_id, method, request.layers, request.mom2_adjustment)
    experiment = experiment_service.create(f"{method}_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread(method.upper().replace("_", "-"), job["id"],
                       partial(run_memit_edit, method=method), request.model_dump())
    return _job_response(job)


@router.post("/edit/emmet", response_model=JobResponse)
@operation_guard.protect()
def emmet_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("emmet", request)


@router.post("/edit/memit-fe", response_model=JobResponse)
@operation_guard.protect()
def memit_fe_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("memit_fe", request)


@router.post("/edit/pmet", response_model=JobResponse)
@operation_guard.protect()
def pmet_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("pmet", request)


@router.post("/edit/eamet", response_model=JobResponse)
@operation_guard.protect()
def eamet_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("eamet", request)


@router.post("/edit/alphaedit", response_model=JobResponse)
@operation_guard.protect()
def alphaedit_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("alphaedit", request)


@router.post("/edit/sphere", response_model=JobResponse)
@operation_guard.protect()
def sphere_edit(request: BatchWeightEditRequest) -> JobResponse:
    return _start_batch_weight_edit("sphere", request)


@router.post("/edit/dpo", response_model=JobResponse)
@operation_guard.protect()
def dpo_edit(request: DpoEditRequest) -> JobResponse:
    return _start_training_edit("dpo", request)


@router.post("/edit/ft", response_model=JobResponse)
@operation_guard.protect()
def ft_edit(request: TrainingEditRequest) -> JobResponse:
    return _start_training_edit("ft", request)



@router.post("/edit/lora", response_model=JobResponse)
@operation_guard.protect()
def lora_edit(request: TrainingEditRequest) -> JobResponse:
    return _start_training_edit("lora", request)



def _start_training_edit(method: str, request: TrainingEditRequest) -> JobResponse:
    try:
        validate_training_hparams(method, request.hparams)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _require_loaded_model(request.model_id)
    experiment = experiment_service.create(f"{method}_edit", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread(method.upper(), job["id"], partial(run_training_edit, method=method), request.model_dump())
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
def stream_edit_job(job_id: str, include_result: bool = True) -> StreamingResponse:
    def events():
        after_log_count = 0
        for job in experiment_service.job_events(job_id):
            response = _job_response(job, after_log_count=after_log_count, include_result=include_result)
            after_log_count = response.log_count
            yield f"event: job\ndata: {response.model_dump_json()}\n\n"
        yield "event: complete\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/edit/jobs/{job_id}/cancel", response_model=JobResponse)
@operation_guard.protect()
def cancel_edit_job(job_id: str) -> JobResponse:
    job = experiment_service.request_edit_cancellation(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Edit job not found")
    return _job_response(job)


@router.post("/edit/mom2/status")
def mom2_statistics_status(request: Mom2StatisticsRequest) -> dict:
    try:
        return get_mom2_statistics_status(
            request.model_id, request.method, request.layers, request.stats_source, request.mom2_dtype
        )
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/edit/mom2/source")
@operation_guard.protect(patch_change=True)
def select_mom2_statistics_source(request: Mom2StatisticsSourceRequest) -> dict:
    try:
        return set_mom2_statistics_source(request.model_id, request.stats_source)
    except (EasyEditUnsupportedError, EasyEditRuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/edit/mom2/precision")
@operation_guard.protect(patch_change=True)
def select_mom2_statistics_precision(request: Mom2StatisticsPrecisionRequest) -> dict:
    try:
        return set_mom2_statistics_precision(request.model_id, request.method, request.mom2_dtype)
    except (EasyEditUnsupportedError, EasyEditRuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/edit/mom2/prepare", response_model=JobResponse)
@operation_guard.protect()
def prepare_mom2(request: Mom2StatisticsRequest) -> JobResponse:
    _require_loaded_model(request.model_id)
    try:
        status = get_mom2_statistics_status(request.model_id, request.method, request.layers, request.stats_source, request.mom2_dtype)
        if status["mom2_dtype"] not in status["available_mom2_dtypes"]:
            raise EasyEditRuntimeError("FP64 second-moment computation is not supported on MPS. Select FP32 or FP16.")
    except (EasyEditUnsupportedError, EasyEditRuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    experiment = experiment_service.create("prepare_mom2_statistics", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    _start_edit_thread("MOM2 statistics", job["id"], prepare_mom2_statistics, request.model_dump())
    return _job_response(job)


@router.post("/edit/mom2/import")
@operation_guard.protect(patch_change=True)
def import_mom2(
    source_id: str = Form(..., alias="model_id"),
    method: str = Form(...),
    dataset: str = Form(...),
    layers: str = Form(...),
    stats_source: str | None = Form(default=None),
    mom2_dtype: str | None = Form(default=None),
    files: list[UploadFile] = File(...),
) -> dict:
    try:
        if method not in {"rome", "r_rome", *BATCH_WEIGHT_METHODS}:
            raise ValueError("Method must support second-moment statistics.")
        parsed_layers = json.loads(layers)
        if not isinstance(parsed_layers, list) or not all(isinstance(layer, int) for layer in parsed_layers):
            raise ValueError("Layers must be a JSON array of integers.")
        return import_mom2_statistics(
            source_id,
            method,
            dataset,
            parsed_layers,
            [(file.filename or "upload.npz", file.file) for file in files],
            stats_source,
            mom2_dtype,
        )
    except (EasyEditUnsupportedError, EasyEditRuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/edit/mom2/cache")
def list_mom2_cache(request: Mom2StatisticsRequest) -> dict:
    try:
        return list_mom2_statistics(request.model_id, request.method, request.stats_source, request.mom2_dtype)
    except EasyEditUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EasyEditRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/edit/mom2/cache")
@operation_guard.protect(patch_change=True)
def delete_mom2_cache(request: Mom2StatisticsRequest) -> dict:
    if _edit_job_lock.locked():
        raise HTTPException(status_code=409, detail="Wait for the active edit or statistics job before deleting cached statistics.")
    try:
        return delete_mom2_statistics(
            request.model_id,
            request.method,
            request.layers or [],
            request.stats_source,
            request.mom2_dtype,
        )
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


def _start_edit_thread(method_label: str, job_id: str, runner, payload: dict) -> None:
    if method_label in _EDIT_METHODS:
        payload = {**payload, "generate_evaluation": False}
    operation_guard.start_thread(
        target=_run_edit_job,
        args=(method_label, job_id, runner, payload),
    )


def _run_edit_job(method_label: str, job_id: str, runner, payload: dict) -> None:
    try:
        with _edit_job_lock, model_runtime_service.exclusive_runtime_access():
            current_job = experiment_service.get_job(job_id)
            if current_job is None or current_job.get("status") != "running":
                return
            experiment_service.append_job_log(job_id, f"Starting {method_label} edit")
            experiment_service.set_job_progress(job_id, 0.05)
            with _capture_job_output(job_id):
                result = run_cancellable_edit(
                    job_id, runner,
                    on_patch_state_changed=(lambda: experiment_service.notify_job_patch_state(job_id, payload["model_id"]))
                    if payload.get("model_id") else None,
                    **payload,
                )
            if result.get("_edit_canceled"):
                experiment_service.cancel_job(job_id, message=None)
            else:
                evaluation_job = None
                patch = None
                if result.get("patch_id") and payload.get("model_id") and method_label in _EDIT_METHODS:
                    patch = patch_service.get_patch(result["patch_id"])
                    experiment = experiment_service.create("evaluation", payload["model_id"], {
                        "source": "post_edit", "method": _EDIT_METHODS[method_label],
                        "edit_job_id": job_id, "patch_id": result["patch_id"],
                        "edit_count": len(patch["edits"]),
                        "active_patch_ids": patch_service.active_patches_for_model(payload["model_id"]),
                    })
                    evaluation_job = experiment_service.create_job(experiment["id"])
                    result = {**result, "post_edit_evaluation_job_id": evaluation_job["id"],
                              "post_edit_evaluation_experiment_id": experiment["id"]}
                experiment_service.complete_job(job_id, result)
                if evaluation_job is not None:
                    _run_post_edit_evaluation(evaluation_job["id"], patch, result, payload)
    except EasyEditUnsupportedError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except EasyEditRuntimeError as exc:
        experiment_service.fail_job(job_id, str(exc))
    except Exception as exc:
        experiment_service.fail_job(job_id, str(exc))


def _run_post_edit_evaluation(job_id: str, patch: dict, edit_result: dict, payload: dict) -> None:
    # The edit transaction has already committed. Evaluation never owns rollback.
    try:
        def report(current: int, total: int) -> None:
            experiment_service.append_job_log(
                job_id, f"Post-edit evaluation {current}/{total}.",
                progress=current / total if total else 0.0,
                progress_current=current, progress_total=total, progress_phase="post_edit_evaluation",
            )

        if experiment_service.get_job(job_id)["status"] != "running":
            return
        report(0, len(patch["edits"]))
        with _capture_job_output(job_id):
            result = evaluate_post_edit_reliability(
                model_id=payload["model_id"], method=patch["method"], edits=patch["edits"],
                edit_result=edit_result,
                single_batch=len(patch["edits"]) <= payload.get("batch_size", 1),
                cancel_job_id=job_id, progress_callback=report,
            )
        experiment_service.complete_job(job_id, {**result, "patch_id": patch["patch_id"]})
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
            evaluation = re.fullmatch(r"Post-edit evaluation (\d+)/(\d+)\.", clean)
            if evaluation:
                series_progress = tuple(int(value) for value in evaluation.groups())
            completed, total = series_progress if series_progress is not None else (None, None)
            edit_started = _easyedit_edit_started(clean)
            phase = "post_edit_evaluation" if evaluation else "editing" if edit_started else None
            batch = re.fullmatch(r"(Applying|Completed) edit batch (\d+)/(\d+)\.", clean)
            if batch:
                action, index, count = batch.groups()
                if 1 <= int(index) <= int(count):
                    completed = int(index) - (1 if action == "Applying" else 0)
                    total = int(count)
                    phase = "batch_editing" if action == "Applying" else "batch_completed"
            experiment_service.append_job_log(
                self.job_id,
                clean,
                progress=(completed / total if completed is not None and total else 0.1 if edit_started else None),
                progress_current=completed,
                progress_total=total,
                progress_phase=phase,
            )


def _job_response(job: dict, after_log_count: int | None = None, include_result: bool = True) -> JobResponse:
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
        cancel_requested=bool(job.get("cancel_requested")),
        progress=job.get("progress", 0.0),
        progress_current=job.get("progress_current"),
        progress_total=job.get("progress_total"),
        progress_phase=job.get("progress_phase"),
        patch_state=job.get("patch_state"),
        logs=logs,
        log_count=total_log_count,
        log_start_count=log_start_count,
        result_item_count=int(job.get("result_item_count") or 0),
        result_page_size=job.get("result_page_size"),
        result=_job_result(job) if include_result else job.get("partial_result"),
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
    return PatchListResponse(patches=patch_service.list_patch_summaries())


@router.get("/patches/page")
def list_patch_page(offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200),
                    model_id: str | None = None) -> dict:
    return patch_service.list_patch_page(offset, limit, model_id)


@router.get("/patches/active")
def list_active_patches() -> dict:
    active = patch_service.active_patches_by_model()
    items = []
    for model_id, patch_ids in active.items():
        for index, patch_id in enumerate(patch_ids):
            patch_service.get_patch_edits_page(patch_id, 0, 0)
            from app.storage.catalog import catalog
            with catalog() as db:
                row = db.execute("SELECT summary FROM records WHERE kind='patch' AND id=?", (patch_id,)).fetchone()
            if row:
                patch = json.loads(row[0])
                items.append({
                    **patch, "active": True, "active_index": index,
                    "tensor_size_bytes": patch_service.tensor_size_bytes(patch),
                })
    return {"patches": items}


@router.get("/patches/{patch_id}/edits")
def get_patch_edits(
    patch_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    try:
        return patch_service.get_patch_edits_page(patch_id, offset, limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc


@router.post("/patches/{patch_id}/export")
def export_patch(patch_id: str, request: PatchExportRequest) -> dict:
    try:
        return patch_service.export_patch_bundle(patch_id, request.output_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc


@router.post("/patches/{patch_id}/load")
@operation_guard.protect(patch_change=True)
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
@operation_guard.protect(patch_change=True)
def discard_patch(patch_id: str) -> dict:
    try:
        state = patch_service.discard_active_patch(patch_id)
        model_runtime_service.sync_loaded_model_to_active_patches(state["model_id"])
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/patches/deletion-preview")
def preview_patch_deletion(request: PatchDeletionPreviewRequest) -> dict:
    with operation_guard.patch_deletion():
        try:
            return patch_service.preview_patch_deletion(request.patch_ids)
        except FileNotFoundError as exc:
            raise HTTPException(404, "A selected patch no longer exists. Update the selection.") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.post("/patches/bulk-delete")
def delete_patch_batch(request: PatchBatchDeleteRequest) -> dict:
    with operation_guard.patch_deletion():
        try:
            return patch_service.delete_patch_batch(request.patch_ids, request.preview_token)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, "Could not remove the selected patch files. Refresh history before retrying.") from exc


@router.delete("/patches/{patch_id}")
def delete_patch(patch_id: str) -> dict:
    with operation_guard.patch_deletion() as busy:
        try:
            deletion = patch_service.prepare_patch_artifact_deletion(patch_id, require_inactive=busy)
            try:
                for model_id in deletion["affected_model_ids"]:
                    model_runtime_service.sync_loaded_model_to_active_patches(model_id)
            except Exception:
                patch_service.restore_patch_deletion_state(deletion)
                for model_id in deletion["affected_model_ids"]:
                    try:
                        model_runtime_service.sync_loaded_model_to_active_patches(model_id)
                    except Exception:
                        pass
                raise
            return patch_service.finalize_patch_artifact_deletion(deletion)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Patch not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/patches")
@operation_guard.protect(patch_change=True)
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
@operation_guard.protect(patch_change=True)
def revert_to_patch(patch_id: str) -> dict:
    try:
        state = patch_service.revert_to_patch(patch_id)
        model_runtime_service.sync_loaded_model_to_active_patches(state["model_id"])
        return state
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Patch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
