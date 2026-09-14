from __future__ import annotations

from pathlib import Path

from app.services.patch_service import PatchService
from app.storage.local_store import local_store


def test_update_patch_keeps_one_cumulative_series_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    service.create_patch(
        model_id="demo-model",
        method="rome",
        edits=[{"subject": "First", "relation_prompt": "{} is", "new_object": "one"}],
        modified_layers=[17],
        metrics={"step": 1},
        delta_paths=["first.safetensors"],
        patch_id="patch_series",
        extra_metadata={"series_edit": True, "series_fact_count": 1},
    )

    updated = service.update_patch(
        "patch_series",
        edits=[
            {"subject": "First", "relation_prompt": "{} is", "new_object": "one"},
            {"subject": "Second", "relation_prompt": "{} is", "new_object": "two"},
        ],
        modified_layers=[17],
        metrics={"step": 2},
        delta_paths=["cumulative.safetensors"],
        extra_metadata={"series_edit": True, "series_fact_count": 2},
    )

    assert updated["patch_id"] == "patch_series"
    assert updated["series_fact_count"] == 2
    assert updated["delta_paths"] == ["cumulative.safetensors"]
    assert [edit["subject"] for edit in updated["edits"]] == ["First", "Second"]
    assert len(service.list_patches()) == 1
