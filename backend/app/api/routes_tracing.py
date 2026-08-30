from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.schemas import CausalTraceRequest, JobResponse
from app.ml.tracing.causal_trace import run_causal_trace
from app.services.experiment_service import experiment_service

router = APIRouter(tags=["tracing"])


@router.post("/causal-trace", response_model=JobResponse)
def causal_trace(request: CausalTraceRequest) -> JobResponse:
    experiment = experiment_service.create("causal_trace", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    try:
        result = run_causal_trace(**request.model_dump(), require_configured=True)
    except ValueError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    experiment_service.complete_job(job["id"], result)
    return JobResponse(
        job_id=job["id"],
        experiment_id=experiment["id"],
        status="completed",
        result=result,
    )
