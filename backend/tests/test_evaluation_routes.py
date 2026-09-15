from __future__ import annotations

from app.api import routes_evaluation


def test_cancel_evaluation_keeps_the_model_worker_running(monkeypatch) -> None:
    running_job = {
        "id": "job_demo",
        "experiment_id": "exp_demo",
        "status": "running",
        "progress": 0.5,
        "logs": ["Job started"],
        "log_count": 1,
    }
    canceled_job = {
        **running_job,
        "status": "canceled",
        "progress": 1.0,
        "logs": ["Job started"],
        "log_count": 1,
    }
    cancellation_messages: list[str | None] = []

    monkeypatch.setattr(routes_evaluation.experiment_service, "get_job", lambda _: running_job)
    monkeypatch.setattr(
        routes_evaluation.experiment_service,
        "get_experiment",
        lambda _: {"id": "exp_demo", "type": "evaluation"},
    )
    monkeypatch.setattr(
        routes_evaluation.experiment_service,
        "cancel_job",
        lambda _job_id, message: cancellation_messages.append(message) or canceled_job,
    )
    monkeypatch.setattr(
        routes_evaluation.model_runtime_service,
        "cancel_active_operation",
        lambda *_: (_ for _ in ()).throw(AssertionError("Evaluation cancellation must not unload the model.")),
    )

    response = routes_evaluation.cancel_evaluation_job("job_demo")

    assert response.status == "canceled"
    assert cancellation_messages == [None]
