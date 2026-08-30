from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.schemas import EvaluationRequest, JobResponse
from app.services.evaluation_service import evaluation_service
from app.services.experiment_service import experiment_service

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=JobResponse)
def evaluate(request: EvaluationRequest) -> JobResponse:
    experiment = experiment_service.create("evaluation", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    try:
        result = evaluation_service.evaluate_patch(**request.model_dump())
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
