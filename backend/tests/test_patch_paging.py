from __future__ import annotations

from pathlib import Path

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
