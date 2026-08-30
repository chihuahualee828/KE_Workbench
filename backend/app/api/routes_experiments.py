from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.experiment_service import experiment_service

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def list_experiments() -> list[dict]:
    return experiment_service.list_experiments()


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    experiment = experiment_service.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

