from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, SecretStr
from starlette.background import BackgroundTask

from app.services.model_cache_service import model_cache_service
from app.services.model_registry import model_registry
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service

router = APIRouter(tags=["models"])


class ModelCacheRequest(BaseModel):
    model_id: str


class HfTokenRequest(BaseModel):
    token: SecretStr


class LocalModelRequest(BaseModel):
    id: str | None = None
    name: str
    local_path: str
    model_type: str = "gpt2"
    n_layers: int = 0
    supports_generation: bool = True
    supports_logit_lens: bool = False
    supports_causal_trace: bool = False
    supports_rome: bool = False
    supports_r_rome: bool = False
    supports_memit: bool = False
    supports_kn: bool = False
    rome_hparams: str | None = None
    r_rome_hparams: str | None = None
    memit_hparams: str | None = None
    kn_hparams: str | None = None
    import_job_id: str | None = None


class ModelExportRequest(BaseModel):
    model_id: str
    output_path: str


class LocalFolderImportRequest(BaseModel):
    source_name: str
    total_bytes: int


@router.get("/models")
def list_models() -> list[dict]:
    return model_registry.list_models()


@router.get("/models/hf-token")
def hf_token_status() -> dict:
    return model_cache_service.hf_token_status()


@router.post("/models/hf-token")
def set_hf_token(request: HfTokenRequest) -> dict:
    try:
        return model_cache_service.set_hf_token(request.token.get_secret_value())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/hf-token")
def clear_hf_token() -> dict:
    return model_cache_service.clear_hf_token()


@router.post("/models/local")
def add_local_model(request: LocalModelRequest) -> dict:
    try:
        model = request.model_dump()
        if request.import_job_id:
            return model_cache_service.register_staged_local_model(request.import_job_id, model)
        model.pop("import_job_id", None)
        return model_registry.add_local_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/local/import")
def import_local_model(files: list[UploadFile] = File(...)) -> dict:
    try:
        return model_cache_service.stage_local_model(
            [(file.filename or "uploaded-model-file", file.file) for file in files]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/local/import/jobs")
def start_local_model_import(request: LocalFolderImportRequest) -> dict:
    try:
        return model_cache_service.start_folder_import_job(request.source_name, request.total_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/local/import/jobs/{job_id}/chunks")
def upload_local_model_import_chunk(
    job_id: str,
    relative_path: str = Form(...),
    offset: int = Form(...),
    chunk: UploadFile = File(...),
) -> dict:
    try:
        return model_cache_service.append_folder_import_chunk(job_id, relative_path, offset, chunk.file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/models/local/import/jobs/{job_id}/complete")
def complete_local_model_import(job_id: str) -> dict:
    try:
        return model_cache_service.complete_folder_import_upload(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/models/local")
def delete_local_model(request: ModelCacheRequest) -> dict:
    try:
        runtime_status = model_runtime_service.runtime_status(request.model_id)
        if runtime_status["loaded"] or runtime_status["runtime_state"] in {"queued", "loading"}:
            raise RuntimeError("Unload the model before removing its local cache.")
        return model_cache_service.delete_local_model(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/models/cache")
def list_model_cache() -> list[dict]:
    return model_cache_service.list_cache_statuses()


@router.post("/models/cache/download")
def download_model(request: ModelCacheRequest) -> dict:
    try:
        return model_cache_service.start_download(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models/cache/jobs")
def list_model_download_jobs() -> list[dict]:
    return model_cache_service.list_jobs()


@router.delete("/models/cache/jobs")
def clear_model_download_jobs() -> dict:
    return {"deleted": model_cache_service.clear_finished_jobs()}


@router.get("/models/cache/jobs/{job_id}")
def get_model_download_job(job_id: str) -> dict:
    job = model_cache_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Model download job not found")
    return job


@router.post("/models/cache/jobs/{job_id}/cancel")
def cancel_model_download_job(job_id: str) -> dict:
    try:
        return model_cache_service.cancel_download(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/models/cache")
def delete_model_cache(request: ModelCacheRequest) -> dict:
    try:
        runtime_status = model_runtime_service.runtime_status(request.model_id)
        if runtime_status["loaded"] or runtime_status["runtime_state"] in {"queued", "loading"}:
            raise RuntimeError("Unload the model before deleting its cache.")
        return model_cache_service.delete_model_cache(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/models/runtime")
def list_model_runtime() -> list[dict]:
    return model_runtime_service.list_runtime_statuses()


@router.get("/models/runtime/events")
def stream_model_runtime() -> StreamingResponse:
    def events():
        for event in model_runtime_service.runtime_status_events():
            if event["type"] == "keepalive":
                yield ": keepalive\n\n"
                continue
            yield f"event: runtime\ndata: {json.dumps(event['statuses'])}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/models/runtime/load")
def load_model(request: ModelCacheRequest) -> dict:
    try:
        return model_runtime_service.start_load(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models/runtime/jobs/{job_id}/wait")
def wait_model_runtime_job(job_id: str, timeout: float = 60.0) -> dict:
    job = model_runtime_service.wait_job(job_id, timeout=timeout)
    if job is None:
        raise HTTPException(status_code=404, detail="Model runtime job not found")
    return job


@router.get("/models/runtime/jobs/{job_id}")
def get_model_runtime_job(job_id: str) -> dict:
    job = model_runtime_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Model runtime job not found")
    return job


@router.delete("/models/runtime")
def unload_model(request: ModelCacheRequest) -> dict:
    try:
        return model_runtime_service.unload(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/export")
def export_model(request: ModelExportRequest) -> dict:
    try:
        return model_runtime_service.export_active_model(request.model_id, request.output_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/models/export/archive")
def export_model_archive(request: ModelCacheRequest) -> FileResponse:
    try:
        archive_path, temporary_root = model_runtime_service.export_active_model_archive(request.model_id)
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=archive_path.name,
            background=BackgroundTask(shutil.rmtree, temporary_root, ignore_errors=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
