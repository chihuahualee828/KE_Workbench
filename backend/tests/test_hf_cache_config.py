from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import config
from app.ml.adapters import gpt2_adapter
from app.services import model_cache_service as cache
from app.services import model_cache_paths as paths
from app.storage.local_store import LocalStore


@pytest.fixture
def cache_paths(monkeypatch, tmp_path):
    root = tmp_path / "default hub"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "settings", SimpleNamespace(hf_hub_cache_dir=root))
    monkeypatch.setattr(paths, "default_hf_hub_cache_dir", lambda: root)
    manager = paths.ModelCachePaths(LocalStore(tmp_path / "data"))
    manager.remember_environment()
    monkeypatch.setattr(cache, "model_cache_paths", manager)
    monkeypatch.setattr(cache, "local_store", manager.store)
    monkeypatch.setattr(cache, "_jobs_dir", lambda: tmp_path / "data" / "model_cache_jobs")
    return manager


@pytest.fixture
def cache_service(monkeypatch, cache_paths):
    service = cache.ModelCacheService()
    monkeypatch.setattr(cache, "model_cache_service", service)
    monkeypatch.setattr(cache.model_registry, "get", lambda _: {"id": "test", "hf_id": "test/model"})
    monkeypatch.setattr(service, "_hub_size_bytes", lambda _: 100)
    return service


def complete_snapshot(root):
    repository = root / "models--test--model"
    snapshot = repository / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("a" * 40)
    return repository


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({}, "home/.cache/huggingface/hub"),
        ({"HF_HUB_CACHE": ""}, "home/.cache/huggingface/hub"),
        ({"XDG_CACHE_HOME": "~/xdg"}, "home/xdg/huggingface/hub"),
        ({"HF_HOME": "~/hf", "XDG_CACHE_HOME": "~/xdg"}, "home/hf/hub"),
        ({"HF_HUB_CACHE": "~/hub", "HF_HOME": "~/hf"}, "home/hub"),
        ({"HUGGINGFACE_HUB_CACHE": "~/legacy"}, "home/legacy"),
        ({"TRANSFORMERS_CACHE": "~/transformers"}, "home/transformers"),
        ({"HF_HUB_CACHE": "~/hub", "HUGGINGFACE_HUB_CACHE": "~/old",
          "TRANSFORMERS_CACHE": "~/transformers"}, "home/hub"),
        ({"HF_HUB_CACHE": "cache/hub"}, "backend/cache/hub"),
    ],
)
def test_cache_directory_resolution(monkeypatch, tmp_path, environment, expected):
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(config, "_backend_root", tmp_path / "backend")
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert config._resolve_hf_hub_cache_dir() == (tmp_path / expected).resolve()


def test_download_scan_and_tokenizer_share_configured_directory(monkeypatch, tmp_path, cache_paths, cache_service):
    root = tmp_path / "custom hub"
    cache_paths.set_directory(str(root))
    service = cache_service
    monkeypatch.setattr(service, "_hf_token", lambda: None)
    calls = {}

    def start_process(args, **kwargs):
        calls["env"] = kwargs["env"]
        return SimpleNamespace()

    monkeypatch.setattr(cache.subprocess, "Popen", start_process)
    service._start_snapshot_download_process("test/model", [])
    assert service._model_cache_path("test/model") == root / "models--test--model"

    def download(**kwargs):
        calls["download"] = kwargs

    def tokenizer(name, **kwargs):
        calls["tokenizer"] = kwargs
        return object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", SimpleNamespace(disable_progress_bars=lambda: None))
    for key in ("KEW_HF_REPO_ID", "KEW_HF_CACHE_DIR", "KEW_HF_IGNORE_PATTERNS"):
        monkeypatch.setenv(key, calls["env"][key])
    exec(cache.SNAPSHOT_DOWNLOAD_SCRIPT, {})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
    ))
    gpt2_adapter.GPT2Adapter("test/model").load_tokenizer(local_files_only=True)
    assert Path(calls["download"]["cache_dir"]) == root
    assert Path(calls["tokenizer"]["cache_dir"]) == root
    assert calls["tokenizer"]["local_files_only"] is True


def test_settings_persist_and_preserve_previous_directories(cache_paths, tmp_path):
    original = cache_paths.download_dir()
    first = tmp_path / "ssd" / "cache"
    second = tmp_path / "other" / "hub"
    cache_paths.set_directory(str(first))
    result = cache_paths.set_directory(str(second))
    restored = paths.ModelCachePaths(cache_paths.store)
    assert first.is_dir() and second.is_dir()
    assert restored.configuration() == result
    assert restored.roots()[:2] == [second, original]
    assert first in restored.roots()
    result = restored.set_directory("")
    assert result["source"] == "environment"
    assert restored.download_dir() == original
    assert {first, second}.issubset(restored.roots())


def test_environment_change_preserves_previously_used_cache(cache_paths, tmp_path, monkeypatch):
    original = cache_paths.download_dir()
    replacement = tmp_path / "new environment"
    monkeypatch.setattr(paths.settings, "hf_hub_cache_dir", replacement)
    restored = paths.ModelCachePaths(cache_paths.store)
    restored.remember_environment()
    assert restored.download_dir() == replacement
    assert original in restored.roots()


@pytest.mark.parametrize("invalid", ["relative/cache", "not-a-directory"])
def test_invalid_directory_does_not_change_settings(cache_paths, tmp_path, invalid):
    previous = cache_paths.configuration()
    target = invalid
    if invalid == "not-a-directory":
        target = str(tmp_path / invalid)
        Path(target).write_text("file")
    with pytest.raises(ValueError):
        cache_paths.set_directory(target)
    assert cache_paths.configuration() == previous


def test_unwritable_directory_does_not_change_settings(cache_paths, tmp_path, monkeypatch):
    previous = cache_paths.configuration()
    def denied(**kwargs):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(paths.tempfile, "TemporaryFile", denied)
    with pytest.raises(ValueError, match="not writable"):
        cache_paths.set_directory(str(tmp_path / "denied"))
    assert cache_paths.configuration() == previous


def test_existing_cache_loads_from_old_folder_after_destination_change(cache_paths, cache_service, tmp_path, monkeypatch):
    original = cache_paths.download_dir()
    repository = complete_snapshot(original)
    cache_paths.set_directory(str(tmp_path / "new hub"))
    status = cache_service.cache_status("test")
    assert status["cached"] is True
    assert status["cache_path"] == str(repository)
    calls = []
    class Model:
        def to(self, target):
            pass
        def eval(self):
            pass
        def parameters(self):
            return iter([SimpleNamespace(device="cpu")])
    def load(name, **kwargs):
        calls.append(kwargs)
        return Model()
    monkeypatch.setattr(gpt2_adapter, "model_device", lambda: "cpu")
    monkeypatch.setattr(gpt2_adapter, "_unload_other_adapters_locked", lambda _: None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    adapter = gpt2_adapter.GPT2Adapter("test/model")
    adapter.load_model(local_files_only=True)
    adapter.load_tokenizer(local_files_only=True)
    assert len(calls) == 2
    assert all(call["cache_dir"] == str(original) and call["local_files_only"] for call in calls)
    monkeypatch.setattr(cache.threading, "Thread", lambda **kwargs: pytest.fail("Redownloaded an existing cached model"))
    assert cache_service.start_download("test")["status"] == "completed"


def test_primary_cache_preferred_but_partial_cache_does_not_hide_complete_old_cache(cache_paths, cache_service, tmp_path):
    original = cache_paths.download_dir()
    old_repository = complete_snapshot(original)
    new_root = tmp_path / "new hub"
    cache_paths.set_directory(str(new_root))
    partial = new_root / "models--test--model" / "blobs" / "partial.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    assert cache_service.cache_status("test")["cache_path"] == str(old_repository)
    partial.unlink()
    new_repository = complete_snapshot(new_root)
    assert cache_service.cache_status("test")["cache_path"] == str(new_repository)


def test_download_destination_is_pinned_to_job(cache_paths, cache_service, tmp_path, monkeypatch):
    destination = tmp_path / "downloads"
    cache_paths.set_directory(str(destination))
    monkeypatch.setattr(cache.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    job = cache_service.start_download("test")
    assert job["cache_dir"] == str(destination)
    with pytest.raises(RuntimeError, match="active model downloads"):
        cache_service.set_cache_location(str(tmp_path / "blocked"))
    cache_paths.set_directory(str(tmp_path / "new destination"))
    assert cache._download_cache_root(job) == destination


def test_cache_deletion_and_cancellation_use_correct_roots(cache_paths, cache_service, tmp_path, monkeypatch):
    original = cache_paths.download_dir()
    old_repository = complete_snapshot(original)
    new_root = tmp_path / "new hub"
    cache_paths.set_directory(str(new_root))
    new_repository = complete_snapshot(new_root)
    scanned = []
    def scan_cache_dir(cache_dir):
        scanned.append(cache_dir)
        return SimpleNamespace(repos=[])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(scan_cache_dir=scan_cache_dir))
    cache_service._delete_download_cache("test/model", new_root)
    assert old_repository.exists() and not new_repository.exists()
    assert scanned == [new_root]
    complete_snapshot(new_root)
    cache_service.delete_model_cache("test")
    assert not old_repository.exists() and not new_repository.exists()
    assert cache_service.cache_status("test")["cached"] is False


def test_settings_api_applies_resets_and_rejects_busy_download(cache_paths, cache_service, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import routes_settings
    monkeypatch.setattr(routes_settings, "model_cache_service", cache_service)
    application = FastAPI()
    application.include_router(routes_settings.router, prefix="/api")
    client = TestClient(application)
    assert client.get("/api/settings").json()["model_cache"] == cache_paths.configuration()
    directory = tmp_path / "api cache"
    response = client.put("/api/settings/model-cache", json={"download_dir": str(directory)})
    assert response.status_code == 200
    assert response.json()["download_dir"] == str(directory)
    assert client.put("/api/settings/model-cache", json={"download_dir": "relative"}).status_code == 400
    cache_service._jobs["active"] = {"status": "downloading"}
    assert client.put("/api/settings/model-cache", json={"download_dir": ""}).status_code == 409
    cache_service._jobs.clear()
    response = client.put("/api/settings/model-cache", json={"download_dir": ""})
    assert response.status_code == 200
    assert response.json()["source"] == "environment"


def test_local_import_does_not_block_download_folder_changes(cache_paths, cache_service, tmp_path):
    cache_service._jobs["import"] = {"status": "uploading", "job_type": "local_import"}
    directory = tmp_path / "hub"
    assert cache_service.set_cache_location(str(directory))["download_dir"] == str(directory)
