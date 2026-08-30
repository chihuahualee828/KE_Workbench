from __future__ import annotations

from app.workers.celery_app import celery_app


@celery_app.task
def ping() -> str:
    return "pong"

