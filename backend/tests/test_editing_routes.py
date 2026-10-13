from __future__ import annotations

from app.api import routes_editing
from app.db.schemas import RomeEditRequest
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


def test_rome_edit_returns_the_new_job_after_starting_the_background_thread(monkeypatch) -> None:
    started: list[tuple[str, str]] = []
    job = {"id": "job_demo", "experiment_id": "exp_demo", "status": "running", "progress": 0.0, "logs": ["Job started"], "log_count": 1}

    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing, "_require_mom2_statistics", lambda *_: None)
    monkeypatch.setattr(routes_editing.patch_service, "deactivate_runtime_patches_for_model",
                        lambda _: (_ for _ in ()).throw(AssertionError("Do not reset patches before the transaction")))
    monkeypatch.setattr(routes_editing.experiment_service, "create", lambda *_: {"id": "exp_demo"})
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda _: job)
    monkeypatch.setattr(
        routes_editing,
        "_start_edit_thread",
        lambda label, job_id, _function, _payload: started.append((label, job_id)),
    )

    response = routes_editing.rome_edit(
        RomeEditRequest(
            model_id="gpt2-xl",
            subject="The Space Needle",
            relation_prompt="{} is located in",
            new_object="Paris",
        )
    )

    assert response.job_id == "job_demo"
    assert response.experiment_id == "exp_demo"
    assert started == [("ROME", "job_demo")]


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
