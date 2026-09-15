from __future__ import annotations

from pathlib import Path

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
