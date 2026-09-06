from __future__ import annotations

import os
from pathlib import Path

from app.services.model_cache_service import (
    LEGACY_PYTORCH_WEIGHT_PATTERNS,
    _download_ignore_patterns,
    _selected_model_cache_size,
)


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
