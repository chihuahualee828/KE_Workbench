from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.editing.methods import METHODS
from app.services import model_cache_service as cache_module, model_registry as registry_module

from app.services.model_cache_service import (
    LEGACY_PYTORCH_WEIGHT_PATTERNS,
    _download_ignore_patterns,
    _selected_model_cache_size,
)


@pytest.mark.parametrize("base_id", ["gpt2-xl", "llama-7b", "qwen2.5-7b", None])
def test_import_preserves_all_method_metadata_from_preview_to_registration(monkeypatch, tmp_path, base_id):
    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(cache_module, "settings", settings)
    monkeypatch.setattr(registry_module, "settings", settings)
    monkeypatch.setattr(cache_module.local_store, "root", tmp_path)
    monkeypatch.setattr(cache_module.model_cache_paths, "remember_environment", lambda: None)
    registry = registry_module.ModelRegistry()
    monkeypatch.setattr(cache_module, "model_registry", registry)
    service = cache_module.ModelCacheService()
    base = registry.get(base_id) if base_id else {}
    config = {
        "model_type": base.get("model_type", "gpt2"),
        "num_hidden_layers": base.get("n_layers", 2),
        "_name_or_path": base.get("hf_id", "custom-tiny"),
    }
    job = service.stage_local_model([
        ("config.json", BytesIO(json.dumps(config).encode())),
        ("model.safetensors", BytesIO(b"placeholder weights")),
        ("tokenizer.json", BytesIO(b"{}")),
    ])
    preview = job["preview"]
    for spec in METHODS.values():
        assert preview[spec["support_key"]] is bool(base.get(spec["support_key"]))
        assert preview[spec["hparams_key"]] == base.get("easyedit", {}).get(spec["hparams_key"])

    # Browser-supplied flags and paths cannot override inspected capabilities.
    stored = service.register_staged_local_model(job["job_id"], {
        "id": preview["id"],
        **{spec["support_key"]: not preview[spec["support_key"]] for spec in METHODS.values()},
        **{spec["hparams_key"]: "untrusted.yaml" for spec in METHODS.values()},
    })
    direct = registry.add_imported_local_model(
        model_id="local-direct", local_path=Path(stored["local_path"]), metadata=job["metadata"],
    )
    for model in (stored, registry.get(stored["id"]), direct):
        for spec in METHODS.values():
            assert model[spec["support_key"]] == preview[spec["support_key"]]
            assert model["easyedit"].get(spec["hparams_key"]) == preview[spec["hparams_key"]]


def _link_snapshot_file(snapshot: Path, blobs: Path, name: str, size: int) -> None:
    blob = blobs / name.replace("/", "_")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"x" * size)
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(blob, target)


def test_safetensors_policy_skips_duplicate_pytorch_weights(tmp_path: Path) -> None:
    filenames = ["config.json", "model.safetensors", "pytorch_model.bin"]

    patterns = _download_ignore_patterns(filenames)

    assert set(LEGACY_PYTORCH_WEIGHT_PATTERNS).issubset(patterns)

    cache_root = tmp_path / "models--example"
    snapshot = cache_root / "snapshots" / "revision"
    blobs = cache_root / "blobs"
    _link_snapshot_file(snapshot, blobs, "config.json", 10)
    _link_snapshot_file(snapshot, blobs, "model.safetensors", 100)
    _link_snapshot_file(snapshot, blobs, "pytorch_model.bin", 200)

    assert _selected_model_cache_size(cache_root, [snapshot]) == 110


def test_bin_only_policy_keeps_required_pytorch_weights(tmp_path: Path) -> None:
    filenames = ["config.json", "pytorch_model.bin"]

    patterns = _download_ignore_patterns(filenames)

    assert not set(LEGACY_PYTORCH_WEIGHT_PATTERNS).intersection(patterns)

    cache_root = tmp_path / "models--example"
    snapshot = cache_root / "snapshots" / "revision"
    blobs = cache_root / "blobs"
    _link_snapshot_file(snapshot, blobs, "config.json", 10)
    _link_snapshot_file(snapshot, blobs, "pytorch_model.bin", 200)

    assert _selected_model_cache_size(cache_root, [snapshot]) == 210
