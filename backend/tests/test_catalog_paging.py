from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from app.api import routes_editing
from app.services.experiment_service import ExperimentService
from app.services.patch_service import PatchService
from app.storage.local_store import local_store
from app.storage.catalog import catalog_path


def make_patch(service, count=120, patch_id="patch_test", **metadata):
    return service.create_patch(
        model_id="demo", method="rome", modified_layers=[1], patch_id=patch_id,
        edits=[{"subject": f"Fact {i}", "relation_prompt": "{} is", "new_object": str(i),
                "source_metadata": {"locality_prompt": "Other question"} if i % 2 else {}}
               for i in range(count)], delta_paths=[],
        metrics={"large": "x" * 100_000}, extra_metadata=metadata,
    )


def test_patch_pages_are_bounded_and_reuse_index(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    make_patch(service, series_metrics=[{"large": "x" * 100_000}])
    summary = service.list_patch_summaries()[0]
    assert summary["edit_count"] == 120
    assert "metrics" not in summary and "series_metrics" not in summary
    assert len(json.dumps(summary)) < 1000
    original = local_store.read_json

    def guarded(path):
        assert path != "patches/patch_test/patch.json", "A warm page must not parse the whole patch"
        return original(path)

    monkeypatch.setattr(local_store, "read_json", guarded)
    assert service.get_patch_edits_page("patch_test", 100, 10)["items"][0]["subject"] == "Fact 100"
    assert service.list_patch_page(limit=1)["total"] == 1


def test_index_tracks_worker_updates_deletions_and_rebuilds(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    patch = make_patch(service, count=3)
    assert service.get_patch_edits_page("patch_test", 0, 10)["total"] == 3
    patch["edits"].append({"subject": "New", "relation_prompt": "{} is", "new_object": "new"})
    local_store.write_json("patches/patch_test/patch.json", patch)
    assert service.get_patch_edits_page("patch_test", 0, 10)["total"] == 4
    catalog_path().unlink()
    assert service.get_patch_edits_page("patch_test", 3, 1)["items"][0]["subject"] == "New"
    (tmp_path / "patches/patch_test/patch.json").unlink()
    assert service.list_patch_page()["total"] == 0


def test_active_fact_pages_preserve_dedup_order_counts_and_search(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    make_patch(service, count=120, patch_id="patch_old")
    make_patch(service, count=10, patch_id="patch_new")
    monkeypatch.setattr(service, "active_patches_for_model", lambda _: ["patch_old", "patch_new"])
    expected = service.active_evaluation_facts("demo")
    assert [fact["key"] for fact in expected] == ([f"patch_new:{i}" for i in range(10)] + [f"patch_old:{i}" for i in range(10, 120)])
    page = service.active_evaluation_facts_page("demo", 45, 20)
    assert page["facts"] == expected[45:65]
    assert page["total"] == 120 and page["locality_count"] == 60
    search = service.active_evaluation_facts_page("demo", 1, 1, "FACT 11")
    assert search["match_total"] == 11
    assert search["matches"] == [110]
    assert search["facts"] == []


def test_history_filter_order_and_metadata_only_details(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    records = []
    for index in range(7):
        record = service.create("evaluation", "demo" if index % 2 else "other", {"edits": ["x" * 10000]})
        job = service.create_job(record["id"])
        service.complete_job(job["id"], {"large": "x" * 100_000})
        records.append(record)
    page = service.list_experiment_page(1, 2, "demo", "completed")
    assert page["total"] == 3
    assert [item["id"] for item in page["items"]] == [records[3]["id"], records[1]["id"]]
    assert "edits" not in page["items"][0].get("input_json", {})
    original = local_store.read_json

    def guarded(path):
        assert not path.endswith("/result.json"), "Status checks must not load results"
        return original(path)

    monkeypatch.setattr(local_store, "read_json", guarded)
    details = service.get_experiment_details(records[1]["id"], include_result=False)
    assert details["job"]["status"] == "completed"
    assert "result" not in details and "logs" not in details["job"]


def test_job_summary_events_do_not_read_completed_result(monkeypatch):
    monkeypatch.setattr(routes_editing, "_job_result", lambda _: (_ for _ in ()).throw(AssertionError("Unexpected result load")))
    response = routes_editing._job_response({"id": "job_x", "experiment_id": "exp_x", "status": "completed"}, include_result=False)
    assert response.result is None


def test_concurrent_index_readers(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = PatchService()
    make_patch(service)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(lambda offset: service.get_patch_edits_page("patch_test", offset, 10), [0, 20, 50, 100]))
    assert all(len(page["items"]) == 10 for page in pages)


def test_catalog_is_outside_network_backed_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path / "network-data")
    first = catalog_path()
    assert not first.is_relative_to(local_store.root)
    monkeypatch.setattr(local_store, "root", tmp_path / "other-data")
    assert catalog_path() != first


def test_summary_stream_includes_partial_evaluation_without_loading_result(monkeypatch):
    monkeypatch.setattr(routes_editing, "_job_result", lambda _: (_ for _ in ()).throw(AssertionError("Unexpected result load")))
    response = routes_editing._job_response({
        "id": "job_eval", "experiment_id": "exp_eval", "status": "running",
        "progress_current": 2, "progress_total": 5,
        "result_item_count": 3, "result_page_size": 50,
        "partial_result": {"mean_reliability": 0.5, "completed_fact_count": 2},
    }, include_result=False)
    assert response.result == {"mean_reliability": 0.5, "completed_fact_count": 2}
    assert response.result_item_count == 3 and response.result_page_size == 50
    assert response.progress_current == 2 and response.progress_total == 5
