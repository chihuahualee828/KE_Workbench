from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.schemas import GenerateRequest, GenerateResponse, LogitLensRequest, LogitLensResponse
from app.ml.inference.generate import generate_text
from app.ml.lens.logit_lens import run_logit_lens
from app.services.experiment_service import experiment_service
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service

router = APIRouter(tags=["inference"])


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        model_runtime_service.require_loaded_runtime_bundle(request.model_id)
        result = generate_text(**request.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    experiment = experiment_service.record_completed("generation", request.model_id, request.model_dump(), result)
    return GenerateResponse(experiment_id=experiment["id"], text=result["text"])


@router.post("/logit-lens", response_model=LogitLensResponse)
def logit_lens(request: LogitLensRequest) -> LogitLensResponse:
    try:
        model_runtime_service.require_loaded_runtime_bundle(request.model_id)
        result = run_logit_lens(**request.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    experiment = experiment_service.record_completed("logit_lens", request.model_id, request.model_dump(), result)
    return LogitLensResponse(experiment_id=experiment["id"], layers=result["layers"])
