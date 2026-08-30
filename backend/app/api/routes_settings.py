from __future__ import annotations

import os

from fastapi import APIRouter

from app.config import settings
from app.services.model_cache_service import model_cache_service

router = APIRouter(tags=["settings"])


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
    }
