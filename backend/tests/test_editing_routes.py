from __future__ import annotations

from app.api import routes_editing
from app.db.schemas import RomeEditRequest


def test_rome_edit_returns_the_new_job_after_starting_the_background_thread(monkeypatch) -> None:
    started: list[tuple[str, str]] = []
    patch_stack_preparation: list[tuple[str, bool]] = []
    job = {"id": "job_demo", "experiment_id": "exp_demo", "status": "running", "progress": 0.0, "logs": ["Job started"], "log_count": 1}

    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing, "_require_mom2_statistics", lambda *_: None)
    monkeypatch.setattr(
        routes_editing,
        "_prepare_edit_patch_stack",
        lambda model_id, continue_from_active_patches: patch_stack_preparation.append((model_id, continue_from_active_patches)),
    )
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
    assert patch_stack_preparation == [("gpt2-xl", False)]


def test_preparing_a_fresh_edit_resets_patches_before_starting_the_job(monkeypatch) -> None:
    calls: list[str] = []
    previous_state = {"gpt2-xl": ["patch_active"]}

    monkeypatch.setattr(routes_editing.patch_service, "active_patches_by_model", lambda: previous_state)
    monkeypatch.setattr(
        routes_editing.patch_service,
        "deactivate_runtime_patches_for_model",
        lambda model_id: calls.append(f"deactivate:{model_id}") or {},
    )
    monkeypatch.setattr(
        routes_editing.model_runtime_service,
        "sync_loaded_model_to_active_patches",
        lambda model_id: calls.append(f"sync:{model_id}"),
    )

    routes_editing._prepare_edit_patch_stack("gpt2-xl", False)

    assert calls == ["deactivate:gpt2-xl", "sync:gpt2-xl"]


def test_cancel_active_edit_stops_the_model_worker(monkeypatch) -> None:
    job = {
        "id": "job_demo",
        "experiment_id": "exp_demo",
        "status": "canceled",
        "progress": 1.0,
        "logs": ["Job started", "Edit canceled by user."],
        "log_count": 2,
    }
    stopped: list[str] = []

    monkeypatch.setattr(
        routes_editing.experiment_service,
        "get_job",
        lambda _: {**job, "status": "running"},
    )
    monkeypatch.setattr(routes_editing.experiment_service, "cancel_job", lambda *_: job)
    monkeypatch.setattr(routes_editing, "_stop_canceled_edit_worker", lambda job_id: stopped.append(job_id))
    monkeypatch.setattr(
        routes_editing.threading,
        "Thread",
        lambda *, target, args, daemon: type("ImmediateThread", (), {"start": lambda self: target(*args)})(),
    )
    with routes_editing._active_edit_job_lock:
        routes_editing._active_edit_job_id = "job_demo"
    try:
        response = routes_editing.cancel_edit_job("job_demo")
    finally:
        with routes_editing._active_edit_job_lock:
            routes_editing._active_edit_job_id = None

    assert response.status == "canceled"
    assert stopped == ["job_demo"]
