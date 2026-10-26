import threading

import pytest
from fastapi import HTTPException

from app.api import routes_editing, routes_evaluation
from app.api.operation_guard import OperationGuard, operation_guard


@pytest.mark.parametrize("endpoint,args", [
    (routes_editing.load_patch, ("patch_test",)),
    (routes_editing.discard_patch, ("patch_test",)),
    (routes_editing.revert_to_patch, ("patch_test",)),
    (routes_editing.delete_all_patches, ()),
])
def test_patch_routes_reject_before_reading_or_mutating_state(monkeypatch, endpoint, args):
    def unexpected(*args, **kwargs):
        raise AssertionError("Busy requests must not access patch state")
    for name in ("get_patch", "discard_active_patch", "revert_to_patch",
                 "prepare_patch_artifact_deletion", "deactivate_all_patches"):
        monkeypatch.setattr(routes_editing.patch_service, name, unexpected)
    with operation_guard.reserve():
        with pytest.raises(HTTPException) as error:
            endpoint(*args)
        assert error.value.status_code == 409


@pytest.mark.parametrize("endpoint", [
    routes_editing.rome_edit, routes_editing.r_rome_edit,
    routes_editing.rome_sequential_edits, routes_editing.r_rome_sequential_edits,
    routes_editing.kn_edit, routes_editing.memit_edit, routes_editing.prepare_mom2,
    routes_evaluation.evaluate, routes_evaluation.evaluate_reliability,
    routes_evaluation.evaluate_active_model,
])
def test_patch_change_excludes_new_jobs_before_setup(endpoint):
    # None is intentional: the guard must reject before accessing request fields
    # or resetting the patch stack for a new edit.
    with operation_guard.reserve(patch_change=True):
        with pytest.raises(HTTPException) as error:
            endpoint(None)
        assert error.value.status_code == 409


@pytest.mark.parametrize("starter,runner", [
    (lambda: routes_editing._start_edit_thread("ROME", "job_x", None, {}), "edit"),
    (lambda: routes_evaluation._start_reliability_evaluation(
        job_id="job_x", model_id="demo", method="rome", edits=[], include_generations=True), "reliability"),
    (lambda: routes_evaluation._start_active_model_evaluation(
        job_id="job_x", model_id="demo", method="rome", active_patch_ids=[],
        facts=[], edits=[], modes=[], include_generations=True, needs_edit_quality=True), "evaluation"),
])
def test_background_job_holds_reservation_after_request_returns(monkeypatch, starter, runner):
    entered, release = threading.Event(), threading.Event()
    def work(*args, **kwargs):
        entered.set()
        assert release.wait(5)
    module, name = {
        "edit": (routes_editing, "_run_edit_job"),
        "reliability": (routes_evaluation, "_run_reliability_evaluation"),
        "evaluation": (routes_evaluation, "_run_active_model_evaluation"),
    }[runner]
    monkeypatch.setattr(module, name, work)
    original_start = operation_guard.start_thread
    threads = []
    def start(**kwargs):
        thread = original_start(**kwargs)
        threads.append(thread)
        return thread
    monkeypatch.setattr(operation_guard, "start_thread", start)
    try:
        starter()
        assert entered.wait(5)
        with pytest.raises(HTTPException):
            routes_editing.load_patch("patch_test")
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    with operation_guard.reserve(patch_change=True):
        pass


def test_reservation_exists_before_thread_starts_and_cleans_up_start_failure(monkeypatch):
    guard = OperationGuard()
    class BrokenThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            with pytest.raises(HTTPException):
                with guard.reserve(patch_change=True):
                    pass
            raise RuntimeError("Cannot start thread")
    monkeypatch.setattr(threading, "Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="Cannot start"):
        guard.start_thread(target=lambda: None)
    with guard.reserve(patch_change=True):
        pass


def test_failed_patch_change_releases_reservation():
    guard = OperationGuard()
    @guard.protect(patch_change=True)
    def fail():
        raise ValueError("bad patch")
    with pytest.raises(ValueError):
        fail()
    with guard.reserve():
        pass


def test_cancel_response_does_not_release_guard_before_edit_cleanup(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    job = {"id": "job_cancel", "experiment_id": "exp_cancel", "status": "running", "logs": []}
    monkeypatch.setattr(routes_editing.experiment_service, "request_edit_cancellation", lambda *_: {**job, "cancel_requested": True})
    def edit():
        entered.set()
        assert release.wait(5)
    thread = operation_guard.start_thread(target=edit)
    try:
        assert entered.wait(5)
        response = routes_editing.cancel_edit_job(job["id"])
        assert response.status == "running"
        assert response.cancel_requested
        with pytest.raises(HTTPException) as error:
            routes_editing.load_patch("patch_test")
        assert error.value.status_code == 409
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()
    with operation_guard.reserve(patch_change=True):
        pass


def test_openapi_preserves_decorated_request_types():
    from app.main import app
    schema = app.openapi()
    body = schema["paths"]["/api/edit/rome"]["post"]["requestBody"]
    assert body["content"]["application/json"]["schema"]["$ref"].endswith("/RankOneEditRequest")
