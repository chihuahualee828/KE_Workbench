from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.experiment_service import experiment_service

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def list_experiments() -> list[dict]:
    return experiment_service.list_experiments()


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    experiment = experiment_service.get_experiment_details(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: str) -> dict:
    try:
        return experiment_service.delete_experiment(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/experiments")
def clear_experiments() -> dict[str, int]:
    try:
        return experiment_service.clear_history()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
