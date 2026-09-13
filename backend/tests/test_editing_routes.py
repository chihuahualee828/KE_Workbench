from __future__ import annotations

from app.api import routes_editing


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
