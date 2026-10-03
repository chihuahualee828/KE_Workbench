from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import Field

from app.config import settings
from app.db.schemas import ApiSchema
from app.services.model_cache_service import model_cache_service

router = APIRouter(tags=["settings"])


class ModelCacheLocationRequest(ApiSchema):
    download_dir: str = Field(default="", max_length=4096)


@router.put("/settings/model-cache")
def set_model_cache_location(request: ModelCacheLocationRequest) -> dict:
    try:
        return model_cache_service.set_cache_location(request.download_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/settings")
def get_settings() -> dict:
    easyedit_root = settings.easyedit_root
    return {
        "app_env": settings.app_env,
        "editing_runtime": settings.editing_runtime,
        "easyedit_root": str(easyedit_root),
        "easyedit_available": (easyedit_root / "easyeditor").exists(),
        "data_dir": str(settings.data_dir),
        "install_easyedit": os.getenv("INSTALL_EASYEDIT", ""),
        "hf_token": model_cache_service.hf_token_status(),
        "model_cache": model_cache_service.cache_location(),
    }
