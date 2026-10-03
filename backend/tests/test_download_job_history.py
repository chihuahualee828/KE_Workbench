from __future__ import annotations

import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.storage.local_store import LocalStore

cache_module = importlib.import_module("app.services.model_cache_service")


@pytest.fixture
def history_service(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_module, "local_store", LocalStore(tmp_path))
    monkeypatch.setattr(cache_module, "_jobs_dir", lambda: tmp_path / "model_cache_jobs")
    monkeypatch.setattr(cache_module, "_read_hub_size_cache", lambda: {})
    monkeypatch.setattr(cache_module.model_registry, "get", lambda _: {"id": "test", "hf_id": "test"})
    service = cache_module.ModelCacheService()
    monkeypatch.setattr(service, "_hub_size_bytes", lambda _: 100)
    monkeypatch.setattr(service, "_download_ignore_patterns_for_model", lambda _: [], raising=False)
    return service


@pytest.mark.parametrize("outcome, size", [("completed", 100), ("failed", 40), ("canceled", 40)])
def test_finished_download_keeps_its_progress_after_delete_redownload_and_restart(
    history_service, monkeypatch, outcome, size
):
    service = history_service
    cache = {"size_bytes": size, "cached": outcome == "completed", "cache_state": "cached" if outcome == "completed" else "partial"}
    monkeypatch.setattr(service, "_cache_status", lambda *args, **kwargs: {**cache, "hub_size_bytes": kwargs.get("hub_size_bytes")})
    monkeypatch.setattr(service, "_delete_download_cache", lambda *_: cache.update(size_bytes=0, cached=False, cache_state="not_cached"))
    job = {"job_id": "model_download_old", "model_id": "test", "hf_id": "test", "status": "queued", "created_at": "2026-01-01", "updated_at": "2026-01-01"}
    service._jobs[job["job_id"]] = job

    def communicate():
        if outcome == "canceled":
            job["cancel_requested"] = True
        return "", "Download failed in regression test"

    process = SimpleNamespace(pid=123, poll=lambda: 1 if outcome == "failed" else 0, returncode=1 if outcome == "failed" else 0, communicate=communicate)
    monkeypatch.setattr(service, "_start_snapshot_download_process", lambda *args: process)
    service._download_model(job["job_id"])
    saved = deepcopy(service.get_job(job["job_id"]))
    assert saved["status"] == outcome
    assert saved["progress_current_bytes"] == size
    assert saved["progress_total_bytes"] == 100
    assert saved["cache"]["size_bytes"] == size
    if outcome == "canceled":
        assert cache["size_bytes"] == 0  # Snapshot predates cancellation cleanup.

    for current_size in (0, 20, 100):
        cache.update(size_bytes=current_size)
        assert service.get_job(job["job_id"]) == saved
        assert service.list_jobs() == [saved]

    # Persisted terminal records must never be reconciled against today's cache.
    restarted = cache_module.ModelCacheService()
    monkeypatch.setattr(restarted, "_cache_status", lambda *args, **kwargs: pytest.fail("History inspected live cache"))
    assert restarted.get_job(job["job_id"]) == saved
    assert restarted.list_jobs() == [saved]


@pytest.mark.parametrize("canceling", [False, True])
def test_detached_download_freezes_partial_progress(history_service, monkeypatch, canceling):
    service = history_service
    cache = {"size_bytes": 35, "cached": False, "cache_state": "partial"}
    monkeypatch.setattr(service, "_cache_status", lambda *args, **kwargs: dict(cache))
    monkeypatch.setattr(service, "_delete_download_cache", lambda *_: cache.update(size_bytes=0))
    job = {"job_id": "model_download_detached", "model_id": "test", "status": "canceling" if canceling else "downloading", "progress_total_bytes": 100}
    result = service._reconcile_persisted_job(job, detached=True)
    assert result["status"] == ("canceled" if canceling else "failed")
    assert result["progress_current_bytes"] == 35
    assert result["progress_total_bytes"] == 100
    assert result["cache"]["size_bytes"] == 35
