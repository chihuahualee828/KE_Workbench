from __future__ import annotations

from pathlib import Path

from app.services.experiment_service import ExperimentService
from app.storage.local_store import local_store


def test_canceled_job_remains_canceled_when_the_worker_later_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    experiment = service.create("rome_edit", "gpt2-xl", {})
    job = service.create_job(experiment["id"])

    canceled = service.cancel_job(job["id"], "Edit canceled by user.")
    service.fail_job(job["id"], "The worker exited unexpectedly.")
    service.complete_job(job["id"], {"unexpected": True})

    assert canceled is not None
    assert canceled["status"] == "canceled"
    persisted_job = service.get_job(job["id"])
    persisted_experiment = service.get_experiment(experiment["id"])
    assert persisted_job is not None
    assert persisted_experiment is not None
    assert persisted_job["status"] == "canceled"
    assert persisted_experiment["status"] == "canceled"
    assert persisted_experiment["result_path"] is None


def test_delete_experiment_uses_known_job_id_without_listing_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    experiment = service.create("rome_edit", "gpt2-xl", {})
    job = service.create_job(experiment["id"])
    service.complete_job(job["id"], {"ok": True})

    monkeypatch.setattr(
        local_store,
        "list_json",
        lambda *_: (_ for _ in ()).throw(AssertionError("Known job deletion must not rescan experiment history.")),
    )

    result = service.delete_experiment(experiment["id"], job["id"])

    assert result == {"experiment_id": experiment["id"]}
    assert not (tmp_path / "experiments" / f"{experiment['id']}.json").exists()
    assert not (tmp_path / "experiments" / f"{job['id']}.json").exists()
