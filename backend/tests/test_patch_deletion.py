from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import routes_editing
from app.api.operation_guard import operation_guard

from app.services.patch_service import PatchService
from app.storage.local_store import local_store


def _create_patch(service: PatchService, patch_id: str, parents: list[str] | None = None) -> None:
    service.create_patch(
        model_id="demo-model",
        method="rome",
        edits=[],
        modified_layers=[1],
        delta_paths=[],
        patch_id=patch_id,
        extra_metadata={"sequential_edit": bool(parents), "parent_patch_ids": parents or []},
    )


def test_deleting_a_patch_removes_transitive_descendants_and_deactivates_them(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    _create_patch(service, "patch_root")
    _create_patch(service, "patch_child", ["patch_root"])
    _create_patch(service, "patch_grandchild", ["patch_child"])
    _create_patch(service, "patch_branch", ["patch_root"])
    _create_patch(service, "patch_independent")
    service.restore_active_patches({"demo-model": ["patch_root", "patch_child", "patch_grandchild"]})

    deletion = service.prepare_patch_artifact_deletion("patch_root")

    assert deletion["deleted_patch_ids"] == ["patch_branch", "patch_child", "patch_grandchild", "patch_root"]
    assert deletion["affected_model_ids"] == ["demo-model"]
    assert service.active_patches_for_model("demo-model") == []

    result = service.finalize_patch_artifact_deletion(deletion)

    assert result["deleted_patch_count"] == 4
    assert not (tmp_path / "patches" / "patch_root").exists()
    assert not (tmp_path / "patches" / "patch_child").exists()
    assert not (tmp_path / "patches" / "patch_grandchild").exists()
    assert not (tmp_path / "patches" / "patch_branch").exists()
    assert (tmp_path / "patches" / "patch_independent" / "patch.json").exists()


def test_unrelated_library_deletion_during_job_preserves_active_stack(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    monkeypatch.setattr(routes_editing, "patch_service", service)
    for patch_id, parents in [("patch_active", []), ("patch_library", []), ("patch_child", ["patch_library"])]:
        _create_patch(service, patch_id, parents)
    service.restore_active_patches({"demo-model": ["patch_active"]})
    before = (tmp_path / "model_state/active_patches.json").read_bytes()
    monkeypatch.setattr(routes_editing.model_runtime_service, "sync_loaded_model_to_active_patches",
                        lambda _: pytest.fail("Deleting unrelated artifacts must not touch the model"))
    with operation_guard.reserve():
        result = routes_editing.delete_patch("patch_library")
    assert result["deleted_patch_ids"] == ["patch_child", "patch_library"]
    assert (tmp_path / "model_state/active_patches.json").read_bytes() == before
    assert (tmp_path / "patches/patch_active/patch.json").exists()
    assert not (tmp_path / "patches/patch_library").exists()
    assert not (tmp_path / "patches/patch_child").exists()


@pytest.mark.parametrize("target", ["patch_root", "patch_active"])
def test_busy_deletion_rejects_active_patch_and_inactive_ancestor_before_mutation(monkeypatch, tmp_path, target):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    monkeypatch.setattr(routes_editing, "patch_service", service)
    _create_patch(service, "patch_root")
    _create_patch(service, "patch_active", ["patch_root"])
    service.restore_active_patches({"demo-model": ["patch_active"]})
    before = (tmp_path / "model_state/active_patches.json").read_bytes()
    with operation_guard.reserve():
        with pytest.raises(HTTPException) as error:
            routes_editing.delete_patch(target)
        assert error.value.status_code == 409
    assert (tmp_path / "model_state/active_patches.json").read_bytes() == before
    assert (tmp_path / "patches/patch_root/patch.json").exists()
    assert (tmp_path / "patches/patch_active/patch.json").exists()


def test_library_deletion_cannot_overlap_loading_a_patch(monkeypatch):
    monkeypatch.setattr(routes_editing.patch_service, "prepare_patch_artifact_deletion",
                        lambda *_args, **_kwargs: pytest.fail("Must reject before planning deletion"))
    with operation_guard.reserve(patch_change=True):
        with pytest.raises(HTTPException) as error:
            routes_editing.delete_patch("patch_library")
        assert error.value.status_code == 409
