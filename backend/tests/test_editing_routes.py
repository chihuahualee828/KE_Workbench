from __future__ import annotations

from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest

from app.api import routes_editing
from app.db.schemas import RankOneEditRequest, RankOneSeriesRequest
from app.ml.editing.edit_cancellation import current_edit
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


@pytest.mark.parametrize("method", ["rome", "r_rome"])
@pytest.mark.parametrize("series", [False, True])
def test_rank_one_route_starts_the_correct_runner(monkeypatch, method, series) -> None:
    started: list[tuple[str, str]] = []
    selected = {}
    job = {"id": "job_demo", "experiment_id": "exp_demo", "status": "running", "progress": 0.0, "logs": ["Job started"], "log_count": 1}

    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing, "_require_mom2_statistics", lambda *_: None)
    monkeypatch.setattr(routes_editing.patch_service, "deactivate_runtime_patches_for_model",
                        lambda _: (_ for _ in ()).throw(AssertionError("Do not reset patches before the transaction")))
    monkeypatch.setattr(routes_editing.experiment_service, "create", lambda *_: {"id": "exp_demo"})
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda _: job)
    def start(label, job_id, runner, payload):
        started.append((label, job_id))
        selected.update(method=runner.keywords["method"], runner=runner.func, payload=payload)
    monkeypatch.setattr(routes_editing, "_start_edit_thread", start)

    fact = {"subject": "The Space Needle", "relation_prompt": "{} is located in", "new_object": "Paris"}
    request = (RankOneSeriesRequest(edits=[fact], target_layer=17) if series
               else RankOneEditRequest(**fact))
    endpoint = getattr(routes_editing, f"{method}_sequential_edits" if series else f"{method}_edit")
    response = endpoint(request)

    assert response.job_id == "job_demo"
    assert response.experiment_id == "exp_demo"
    label = method.upper().replace("_", "-") + (" sequential series" if series else "")
    assert started == [(label, "job_demo")]
    assert selected["method"] == method
    assert selected["runner"] is (routes_editing.run_rome_sequential_edits if series else routes_editing.run_rome_edit)
    assert selected["payload"] == request.model_dump()


def test_cancel_active_edit_requests_cleanup_without_stopping_worker(monkeypatch) -> None:
    job = {
        "id": "job_demo",
        "experiment_id": "exp_demo",
        "status": "running",
        "cancel_requested": True,
        "progress": 0.5,
        "logs": ["Job started", "Edit canceled by user."],
        "log_count": 2,
    }
    monkeypatch.setattr(routes_editing.experiment_service, "request_edit_cancellation", lambda *_: job)
    monkeypatch.setattr(routes_editing.model_runtime_service, "cancel_active_operation",
                        lambda _: (_ for _ in ()).throw(AssertionError("Must not unload")))
    response = routes_editing.cancel_edit_job("job_demo")
    assert response.status == "running"
    assert response.cancel_requested
    assert response.progress == 0.5


def test_easyedit_start_marker_advances_job_past_preparation(monkeypatch) -> None:
    emitted: list[tuple[str, str, float | None]] = []
    monkeypatch.setattr(
        routes_editing.experiment_service,
        "append_job_log",
        lambda job_id, message, **kwargs: emitted.append((job_id, message, kwargs.get("progress"))),
    )

    stream = routes_editing._JobLogStream("job_demo", None)
    stream._emit("Starting EasyEdit algorithm.")

    assert emitted == [("job_demo", "Starting EasyEdit algorithm.", 0.1)]


@pytest.mark.parametrize("label", list(routes_editing._EDIT_METHODS))
def test_background_edits_disable_inline_evaluation(monkeypatch, label):
    started = []
    monkeypatch.setattr(routes_editing.operation_guard, "start_thread", lambda **kwargs: started.append(kwargs))
    payload = {"model_id": "demo"}
    routes_editing._start_edit_thread(label, "job_demo", object(), payload)
    assert started[0]["args"][3]["generate_evaluation"] is False
    assert payload == {"model_id": "demo"}


@pytest.mark.parametrize("outcome", ["completed", "canceled", "failed"])
def test_post_edit_evaluation_has_its_own_job_and_cannot_discard_patch(monkeypatch, tmp_path, outcome):
    from app.api import routes_evaluation

    monkeypatch.setattr(local_store, "root", tmp_path)
    monkeypatch.setattr(routes_editing, "model_runtime_service", SimpleNamespace(exclusive_runtime_access=nullcontext))
    monkeypatch.setattr(routes_editing, "run_cancellable_edit", lambda _id, runner, **kwargs: runner(**{
        key: value for key, value in kwargs.items() if key != "on_patch_state_changed"
    }))
    service = routes_editing.experiment_service
    experiment = service.create("rome_edit", "demo", {})
    edit_job = service.create_job(experiment["id"])
    entered, release = threading.Event(), threading.Event()
    evaluation_id = []

    def edit(**payload):
        assert payload["generate_evaluation"] is False
        patch = patch_service.create_patch(
            "demo", "rome", [{"subject": "A", "relation_prompt": "{} is", "new_object": "B"}],
            [0], patch_id="patch_new",
        )
        patch_service.activate_patch(patch["patch_id"])
        return {"patch_id": patch["patch_id"], "metrics": [{"post": {"rewrite_acc": [1.0]}}]}

    def evaluate(**kwargs):
        assert current_edit.get() is None
        assert service.get_job(edit_job["id"])["status"] == "completed"
        assert patch_service.active_patches_for_model("demo") == ["patch_new"]
        evaluation_id.append(kwargs["cancel_job_id"])
        kwargs["progress_callback"](0, 1)
        entered.set()
        assert release.wait(5)
        if outcome == "failed":
            raise RuntimeError("Evaluation failed, not editing")
        if outcome == "canceled":
            raise RuntimeError("Evaluation canceled")
        return {"reliability": {"mean_reliability": 1.0}, "generations": []}

    monkeypatch.setattr(routes_editing, "evaluate_post_edit_reliability", evaluate)
    thread = threading.Thread(target=routes_editing._run_edit_job, args=(
        "ROME", edit_job["id"], edit, {"model_id": "demo", "generate_evaluation": False},
    ))
    thread.start()
    try:
        assert entered.wait(5)
        response = routes_editing.get_edit_job(edit_job["id"])
        assert response.status == "completed"
        assert response.result["post_edit_evaluation_job_id"] == evaluation_id[0]
        assert evaluation_id[0] != edit_job["id"]
        evaluation = routes_editing.get_edit_job(evaluation_id[0])
        assert evaluation.status == "running"
        assert evaluation.progress_phase == "post_edit_evaluation"
        assert routes_editing.cancel_edit_job(edit_job["id"]).status == "completed"
        if outcome == "canceled":
            assert routes_evaluation.cancel_evaluation_job(evaluation_id[0]).status == "canceled"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert service.get_job(edit_job["id"])["status"] == "completed"
    assert service.get_job(evaluation_id[0])["status"] == outcome
    assert patch_service.active_patches_for_model("demo") == ["patch_new"]
    assert patch_service.get_patch("patch_new")["edits"][0]["new_object"] == "B"


def test_patch_state_snapshot_is_in_the_job_event_before_edit_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = routes_editing.experiment_service
    experiment = service.create("rome_edit", "demo", {})
    job = service.create_job(experiment["id"])
    service.notify_job_patch_state(job["id"], "demo")
    first = next(service.job_events(job["id"]))
    assert routes_editing._job_response(first, include_result=False).patch_state == {
        "version": 1, "model_id": "demo", "active_patch_ids": [],
    }
    assert first["status"] == "running"
    patch_service.create_patch("demo", "rome", [], [0], patch_id="patch_checkpoint")
    patch_service.activate_patch("patch_checkpoint")
    service.notify_job_patch_state(job["id"], "demo")
    assert routes_editing.get_edit_job(job["id"]).patch_state == {
        "version": 2, "model_id": "demo", "active_patch_ids": ["patch_checkpoint"],
    }
