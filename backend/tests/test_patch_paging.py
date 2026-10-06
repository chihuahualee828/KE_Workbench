from __future__ import annotations

from pathlib import Path

import pytest

from app.services.patch_service import PatchService
from app.storage.local_store import local_store


def test_patch_edits_are_summarized_and_paged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    edits = [
        {
            "subject": f"subject-{index}",
            "relation_prompt": "{} is located in",
            "ground_truth": "before",
            "new_object": f"after-{index}",
        }
        for index in range(5)
    ]
    service.create_patch(
        model_id="demo-model",
        method="rome",
        edits=edits,
        modified_layers=[1],
        delta_paths=[],
        patch_id="patch_paged",
    )

    summary = service.list_patch_summaries()[0]
    page = service.get_patch_edits_page("patch_paged", offset=2, limit=2)

    assert summary["edits"] == []
    assert summary["edit_count"] == 5
    assert page["total"] == 5
    assert [edit["subject"] for edit in page["items"]] == ["subject-2", "subject-3"]


def test_patch_summaries_measure_current_safetensors_size(monkeypatch, tmp_path: Path) -> None:
    from app.api import routes_editing

    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    patch_id = "patch_sized"
    patch_dir = service.patch_dir(patch_id)
    first = patch_dir / "weight_deltas.safetensors"
    second = patch_dir / "extra.safetensors"
    first.write_bytes(b"a" * 12)
    second.write_bytes(b"b" * 8)
    service.create_patch(
        model_id="demo-model", method="rome", edits=[], modified_layers=[1],
        delta_paths=[str(first), str(second), str(first)], patch_id=patch_id,
    )
    # Legacy absolute paths must still work after the data directory moves.
    metadata = local_store.read_json(f"patches/{patch_id}/patch.json")
    metadata["delta_paths"][0] = f"/old/data/patches/{patch_id}/{first.name}"
    local_store.write_json(f"patches/{patch_id}/patch.json", metadata)
    local_store.write_json("model_state/active_patches.json", {"models": {"demo-model": [patch_id]}})

    assert service.list_patch_summaries()[0]["tensor_size_bytes"] == 20
    assert routes_editing.list_active_patches()["patches"][0]["tensor_size_bytes"] == 20

    # Replacing a cumulative delta must update its size even if metadata is unchanged.
    first.write_bytes(b"a" * 24)
    assert service.list_patch_page()["items"][0]["tensor_size_bytes"] == 32
    assert routes_editing.list_active_patches()["patches"][0]["tensor_size_bytes"] == 32


@pytest.mark.parametrize("delta_paths", [
    [], ["metadata_delta.json"], ["missing.safetensors"],
    ["weight_deltas.safetensors", "missing.safetensors"],
])
def test_unavailable_patch_tensors_have_unknown_size(monkeypatch, tmp_path: Path, delta_paths: list[str]) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    patch_id = "patch_unavailable"
    (service.patch_dir(patch_id) / "weight_deltas.safetensors").write_bytes(b"delta")
    service.create_patch(
        model_id="demo-model", method="rome", edits=[], modified_layers=[1],
        delta_paths=[f"patches/{patch_id}/{name}" for name in delta_paths], patch_id=patch_id,
    )
    assert service.list_patch_summaries()[0]["tensor_size_bytes"] is None
