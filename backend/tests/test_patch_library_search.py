from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_editing
from app.services.patch_service import PatchService
from app.storage.local_store import local_store


@pytest.fixture
def library(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    monkeypatch.setattr(routes_editing, "patch_service", service)
    for index in range(63):
        service.create_patch(
            model_id="model-a" if index % 2 == 0 else "model-b",
            method="rome" if index % 3 == 0 else "memit",
            edits=[{"subject": f"subject-{index}", "relation_prompt": "{} lives in",
                    "ground_truth": "Paris", "new_object": "Tokyo",
                    "source_metadata": {"rephrase_prompt": "Straße 100%_"}}] * 2,
            modified_layers=[1], delta_paths=[], patch_id=f"patch_{index:03}",
            extra_metadata={"created_at": f"2026-01-{index // 3 + 1:02}T00:00:00+00:00",
                            "parent_patch_ids": [f"patch_{index - 1:03}"] if index else [],
                            "sequential_edit": index > 0},
        )
    return service


def test_search_matches_fact_values_across_pages_without_returning_facts(library):
    for query in ["subject-0", "LIVES IN", "subject-0 lives in", "paris", "TOKYO", "STRASSE", "100%_"]:
        page = library.list_patch_page(query=query)
        assert page["total"] > 0, query
        assert all(item["edits"] == [] and item["edit_count"] == 2 for item in page["items"])
    assert library.list_patch_page(query=" subject-0 ")["total"] == 1
    assert library.list_patch_page(query="subject-0")["items"][0]["patch_id"] == "patch_000"
    assert library.list_patch_page(query="source_metadata")["total"] == 0  # Values, not field names.
    assert library.list_patch_page(query="' OR 1=1 --")["total"] == 0


def test_filters_and_sort_are_applied_before_pagination(library):
    page = library.list_patch_page(offset=1, limit=2, model_id="model-a", method="ROME", query="Tokyo", sort="oldest")
    assert page["total"] == 11
    assert [patch["patch_id"] for patch in page["items"]] == ["patch_006", "patch_012"]
    assert page["methods"] == ["memit", "rome"]
    assert page["model_ids"] == ["model-a", "model-b"]
    oldest = library.list_patch_page(limit=None, sort="oldest")["items"]
    newest = library.list_patch_page(limit=None)["items"]
    assert [patch["patch_id"] for patch in oldest] == [f"patch_{index:03}" for index in range(63)]
    assert newest == list(reversed(oldest))
    first = library.list_patch_page(limit=50, sort="oldest")
    second = library.list_patch_page(offset=50, limit=50, sort="oldest")
    assert first["items"] + second["items"] == oldest
    assert library.list_patch_page(query="no match")["methods"] == ["memit", "rome"]


def test_selection_uses_filters_but_protects_ancestors_of_hidden_active_patches(library):
    library.restore_active_patches({"model-b": ["patch_003"]})
    result = library.selectable_patch_ids(method="rome", model_id="model-a")
    assert result["patch_ids"] == [f"patch_{index:03}" for index in reversed(range(6, 63, 6))]
    assert result["blocked_patch_count"] == 1  # patch_000 depends on a filtered-out active descendant.
    assert library.selectable_patch_ids(query="subject-62")["patch_ids"] == ["patch_062"]
    assert library.selectable_patch_ids(query="absent") == {"patch_ids": [], "blocked_patch_count": 0}
    plan = library.preview_patch_deletion(["patch_060"])
    assert plan["dependent_patch_ids"] == ["patch_061", "patch_062"]  # Cascade ignores display filters.


def test_graph_counts_include_filtered_children_and_search_tracks_changed_json(library):
    page = library.list_patch_page(query="subject-0")
    assert page["items"][0]["lineage_child_count"] == 1
    metadata = local_store.read_json("patches/patch_000/patch.json")
    metadata["edits"] = [{"subject": "replacement", "relation_prompt": "{} became", "new_object": "changed"}]
    local_store.write_json("patches/patch_000/patch.json", metadata)
    assert library.list_patch_page(query="subject-0")["total"] == 0
    assert library.list_patch_page(query="replacement became")["total"] == 1
    (local_store.root / "patches/patch_000/patch.json").unlink()
    assert library.list_patch_page(query="replacement")["total"] == 0


def test_existing_catalog_is_reindexed_for_fact_search(library):
    from app.storage.catalog import catalog, sync_collection
    library.list_patch_page()
    with catalog() as db:
        db.execute("DELETE FROM patch_fact_search")
        db.execute("UPDATE records SET signature=substr(signature, 10)")
        sync_collection(db, "patch")
        assert db.execute("SELECT count(*) FROM patch_fact_search").fetchone()[0] == 126
    assert library.list_patch_page(query="Tokyo")["total"] == 63


def test_patch_query_routes_validate_and_combine_parameters(library):
    app = FastAPI()
    app.include_router(routes_editing.router, prefix="/api")
    client = TestClient(app)
    assert client.get("/api/patches/page", params={"sort": "random"}).status_code == 422
    assert client.get("/api/patches/page", params={"query": "x" * 501}).status_code == 422
    assert client.get("/api/patches/page", params={"offset": -1}).status_code == 422
    response = client.get("/api/patches/page", params={"query": "subject-0", "sort": "oldest", "method": "rome", "model_id": "model-a"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    selection = client.get("/api/patches/selection", params={"query": "subject-0"})
    assert selection.json() == {"patch_ids": ["patch_000"], "blocked_patch_count": 0}
