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


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    monkeypatch.setattr(routes_editing, "patch_service", service)
    for patch_id, parents in [
        ("patch_a", []), ("patch_b", ["patch_a"]), ("patch_c", ["patch_b"]),
        ("patch_d", ["patch_a"]), ("patch_other", []),
        ("patch_join", ["patch_b", "patch_other"]),
    ]:
        _create_patch(service, patch_id, parents)
    return service


def test_batch_preview_deduplicates_and_does_not_change_active_state(library, tmp_path):
    library.restore_active_patches({"demo-model": ["patch_a", "patch_b", "patch_c"]})
    before = (tmp_path / "model_state/active_patches.json").read_bytes()
    plan = library.preview_patch_deletion(["patch_b", "patch_a", "patch_b"])
    assert plan["selected_patch_ids"] == ["patch_a", "patch_b"]
    assert plan["dependent_patch_ids"] == ["patch_c", "patch_d", "patch_join"]
    assert plan["blocked_patch_ids"] == ["patch_a", "patch_b", "patch_c"]
    entries = {patch["patch_id"]: patch for patch in plan["patches"]}
    assert entries["patch_c"]["included_by"] == ["patch_a", "patch_b"]
    assert "patch_other" not in entries  # Never follow dependencies upstream.
    assert library.preview_patch_deletion(["patch_a", "patch_b"]) == plan
    assert (tmp_path / "model_state/active_patches.json").read_bytes() == before
    assert len(list((tmp_path / "patches").glob("*/patch.json"))) == 6


def test_batch_delete_uses_reviewed_union_and_preserves_unrelated_active_model(library, tmp_path, monkeypatch):
    library.restore_active_patches({"demo-model": ["patch_other"]})
    before = (tmp_path / "model_state/active_patches.json").read_bytes()
    monkeypatch.setattr(routes_editing.model_runtime_service, "sync_loaded_model_to_active_patches",
                        lambda _: pytest.fail("Inactive deletion must not touch runtime weights"))
    request = routes_editing.PatchDeletionPreviewRequest(patch_ids=["patch_b", "patch_d", "patch_c"])
    with operation_guard.reserve():
        plan = routes_editing.preview_patch_deletion(request)
        result = routes_editing.delete_patch_batch(routes_editing.PatchBatchDeleteRequest(
            patch_ids=request.patch_ids, preview_token=plan["preview_token"],
        ))
    assert result["deleted_patch_ids"] == ["patch_b", "patch_c", "patch_d", "patch_join"]
    assert result["deleted_patch_count"] == 4
    assert result["cleanup_pending"] is False
    assert {path.parent.name for path in (tmp_path / "patches").glob("*/patch.json")} == {"patch_a", "patch_other"}
    assert (tmp_path / "model_state/active_patches.json").read_bytes() == before
    assert not list((tmp_path / "patch_deletions").iterdir())


@pytest.mark.parametrize("busy", [False, True])
def test_batch_blocks_inactive_ancestor_of_active_patch_before_any_deletion(library, tmp_path, busy):
    from contextlib import nullcontext

    library.restore_active_patches({"demo-model": ["patch_c"]})
    plan = library.preview_patch_deletion(["patch_a", "patch_other"])
    with operation_guard.reserve() if busy else nullcontext():
        with pytest.raises(HTTPException) as error:
            routes_editing.delete_patch_batch(routes_editing.PatchBatchDeleteRequest(
                patch_ids=plan["selected_patch_ids"], preview_token=plan["preview_token"],
            ))
    assert error.value.status_code == 409
    assert "Discard" in error.value.detail
    assert len(list((tmp_path / "patches").glob("*/patch.json"))) == 6


@pytest.mark.parametrize("change", ["new_child", "metadata", "missing", "active"])
def test_batch_rechecks_the_review_before_mutating(library, tmp_path, change):
    plan = library.preview_patch_deletion(["patch_b"])
    if change == "new_child":
        _create_patch(library, "patch_late", ["patch_c"])
    elif change == "metadata":
        _create_patch(library, "patch_b", ["patch_a", "patch_other"])
    elif change == "missing":
        (tmp_path / "patches/patch_b/patch.json").unlink()
    else:
        library.restore_active_patches({"demo-model": ["patch_c"]})
    before = {path: path.read_bytes() for path in (tmp_path / "patches").glob("*/patch.json")}
    with pytest.raises(ValueError, match="changed|active"):
        library.delete_patch_batch(plan["selected_patch_ids"], plan["preview_token"])
    assert {path: path.read_bytes() for path in (tmp_path / "patches").glob("*/patch.json")} == before


def test_batch_preview_handles_multiple_parents_branches_and_cycles(library):
    _create_patch(library, "patch_cycle", ["patch_join"])
    _create_patch(library, "patch_join", ["patch_b", "patch_other", "patch_cycle"])
    plan = library.preview_patch_deletion(["patch_other"])
    assert plan["deleted_patch_ids"] == ["patch_cycle", "patch_join", "patch_other"]
    assert all(patch["included_by"] == ["patch_other"] for patch in plan["patches"] if patch["patch_id"] != "patch_other")


def test_batch_staging_failure_restores_previously_moved_patches(library, tmp_path, monkeypatch):
    plan = library.preview_patch_deletion(["patch_a"])
    replace = Path.replace

    def fail_second(source, target):
        if source == tmp_path / "patches/patch_b":
            raise OSError("simulated permission failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError, match="permission"):
        library.delete_patch_batch(plan["selected_patch_ids"], plan["preview_token"])
    assert len(list((tmp_path / "patches").glob("*/patch.json"))) == 6
    assert not list((tmp_path / "patch_deletions").iterdir())


def test_batch_cleanup_failure_reports_pending_files_without_resurrecting_a_partial_chain(library, tmp_path, monkeypatch):
    import shutil

    plan = library.preview_patch_deletion(["patch_b"])
    monkeypatch.setattr(shutil, "rmtree", lambda *_: (_ for _ in ()).throw(OSError("cleanup failed")))
    result = library.delete_patch_batch(plan["selected_patch_ids"], plan["preview_token"])
    assert result["cleanup_pending"] is True
    assert not (tmp_path / "patches/patch_b").exists()
    assert not (tmp_path / "patches/patch_c").exists()
    assert len(list((tmp_path / "patch_deletions").glob("*/patch_b/patch.json"))) == 1


def test_batch_api_validation_and_patch_change_guard(library):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(routes_editing.router, prefix="/api")
    client = TestClient(app)
    assert client.post("/api/patches/deletion-preview", json={"patch_ids": []}).status_code == 422
    assert client.post("/api/patches/deletion-preview", json={"patch_ids": ["../patch_a"]}).status_code == 404
    plan = client.post("/api/patches/deletion-preview", json={"patch_ids": ["patch_c"]}).json()
    payload = {"patch_ids": ["patch_c"], "preview_token": plan["preview_token"]}
    with operation_guard.reserve(patch_change=True):
        assert client.post("/api/patches/bulk-delete", json=payload).status_code == 409
        assert client.post("/api/patches/deletion-preview", json={"patch_ids": ["patch_c"]}).status_code == 409
    response = client.post("/api/patches/bulk-delete", json=payload)
    assert response.status_code == 200
    assert response.json()["deleted_patch_ids"] == ["patch_c"]
