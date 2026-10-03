from pathlib import Path

import pytest

from app.services.patch_service import PatchService
from app.storage.local_store import local_store


def test_relocated_patch_loads_applies_and_reverts(monkeypatch, tmp_path):
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    patch_id = "patch_moved"
    tensor_path = service.patch_dir(patch_id) / "weight_deltas.safetensors"
    save_file({"weight": torch.ones(1, 2)}, str(tensor_path))
    service.create_patch(
        model_id="demo", method="rome", edits=[], modified_layers=[0],
        patch_id=patch_id,
        delta_paths=[f"/old/machine/data/patches/{patch_id}/{tensor_path.name}"],
    )
    # Simulate metadata written by an older installation before loading it.
    metadata_path = f"patches/{patch_id}/patch.json"
    legacy = local_store.read_json(metadata_path)
    legacy["delta_paths"] = [f"/old/machine/data/patches/{patch_id}/{tensor_path.name}"]
    local_store.write_json(metadata_path, legacy)

    service.load_saved_patch(patch_id)
    assert local_store.read_json(metadata_path)["delta_paths"] == [
        f"patches/{patch_id}/{tensor_path.name}"
    ]
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    service.sync_active_patches_to_model(model, "demo")
    torch.testing.assert_close(model.weight, torch.ones(1, 2))
    service.sync_patch_stack_to_model(model, "demo", [])
    torch.testing.assert_close(model.weight, torch.zeros(1, 2))


def test_existing_absolute_path_keeps_precedence(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path / "current")
    original = tmp_path / "old/patches/patch_same/weight_deltas.safetensors"
    relocated = local_store.root / "patches/patch_same/weight_deltas.safetensors"
    for path in (original, relocated):
        path.parent.mkdir(parents=True)
        path.touch()
    assert PatchService().resolve_delta_path(str(original)) == original


def test_missing_patch_does_not_use_another_patchs_tensor(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    (service.patch_dir("patch_other") / "weight_deltas.safetensors").touch()
    missing = "/old/data/patches/patch_missing/weight_deltas.safetensors"
    service.create_patch(
        model_id="demo", method="rome", edits=[], modified_layers=[0],
        patch_id="patch_missing", delta_paths=[missing],
    )
    assert service.resolve_delta_path(missing) == Path(missing)
    with pytest.raises(ValueError, match="missing its saved tensor delta file"):
        service.load_saved_patch("patch_missing")
    assert service.active_patches_for_model("demo") == []


def test_create_and_update_save_portable_paths(monkeypatch, tmp_path):
    root = tmp_path / "custom-storage"
    monkeypatch.setattr(local_store, "root", root)
    service = PatchService()
    path = service.patch_dir("patch_portable") / "weight_deltas.safetensors"
    path.touch()
    expected = ["patches/patch_portable/weight_deltas.safetensors"]
    created = service.create_patch(
        model_id="demo", method="rome", edits=[], modified_layers=[0],
        patch_id="patch_portable", delta_paths=[str(path)],
    )
    assert created["delta_paths"] == expected
    updated = service.update_patch(
        "patch_portable", edits=[], modified_layers=[0], metrics={},
        delta_paths=[str(path)],
    )
    assert updated["delta_paths"] == expected
    assert local_store.read_json("patches/patch_portable/patch.json")["delta_paths"] == expected

    # Moving the whole data directory must not require another metadata rewrite.
    moved = tmp_path / "moved-storage"
    root.rename(moved)
    monkeypatch.setattr(local_store, "root", moved)
    monkeypatch.chdir(tmp_path)
    assert service.resolve_delta_path(expected[0]).is_file()
    assert service.load_saved_patch("patch_portable")["active_patches"] == ["patch_portable"]


def test_legacy_data_prefix_uses_configured_root(monkeypatch, tmp_path):
    root = tmp_path / "custom-storage"
    monkeypatch.setattr(local_store, "root", root)
    service = PatchService()
    assert service.resolve_delta_path("data/patches/patch_old/delta.safetensors") == (
        root / "patches/patch_old/delta.safetensors"
    )
