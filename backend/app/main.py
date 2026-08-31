from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_editing,
    routes_evaluation,
    routes_experiments,
    routes_inference,
    routes_models,
    routes_settings,
    routes_tracing,
)
from app.config import ensure_data_dirs
from app.services.experiment_service import experiment_service
from app.services.patch_service import patch_service

app = FastAPI(title="Knowledge Editing Workbench", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_data_dirs()
    experiment_service.reconcile_interrupted_jobs()
    patch_service.clear_active_patches()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(routes_models.router, prefix="/api")
app.include_router(routes_settings.router, prefix="/api")
app.include_router(routes_inference.router, prefix="/api")
app.include_router(routes_tracing.router, prefix="/api")
app.include_router(routes_editing.router, prefix="/api")
app.include_router(routes_evaluation.router, prefix="/api")
app.include_router(routes_experiments.router, prefix="/api")
