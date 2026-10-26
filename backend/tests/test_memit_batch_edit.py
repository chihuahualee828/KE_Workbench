from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.api import routes_editing
from app.db.schemas import BatchWeightEditRequest
from app.ml.editing import memit_wrapper
import app.services.model_runtime_service as runtime_module


FACT = {"subject": "A", "relation_prompt": "{} is", "new_object": "B"}


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
def test_memit_request_rejects_invalid_batch_sizes(value):
    with pytest.raises(ValidationError):
        BatchWeightEditRequest(edits=[FACT], batch_size=value)


def test_memit_request_defaults_to_one_and_requires_facts():
    assert BatchWeightEditRequest(edits=[FACT]).batch_size == 1
    assert BatchWeightEditRequest(edits=[FACT], batch_size=20).batch_size == 20
    with pytest.raises(ValidationError):
        BatchWeightEditRequest(edits=[])


def test_memit_route_keeps_batch_size_in_job_and_runner(monkeypatch):
    recorded = {}
    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing, "_require_mom2_statistics", lambda *_: None)
    def create(kind, model_id, payload):
        recorded["input"] = payload
        return {"id": "exp_test"}
    monkeypatch.setattr(routes_editing.experiment_service, "create", create)
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda _: {
        "id": "job_test", "experiment_id": "exp_test", "status": "running", "logs": [],
    })
    monkeypatch.setattr(routes_editing, "_start_edit_thread", lambda label, job_id, runner, payload: recorded.update(runner=payload))
    response = routes_editing.memit_edit(BatchWeightEditRequest(edits=[FACT], batch_size=20))
    assert response.status == "running"
    assert recorded["input"]["batch_size"] == recorded["runner"]["batch_size"] == 20


@pytest.mark.parametrize("worker", [False, True])
def test_memit_wrapper_preserves_batch_size_and_all_facts(monkeypatch, worker):
    recorded = {}
    def execute(action, payload, **kwargs):
        recorded.update(payload)
        assert action == "memit_edit"
        return {"worker": True}
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=worker, execute=execute))
    def run(**kwargs):
        recorded.update(kwargs)
        return {"patch_id": "patch_test"}
    monkeypatch.setattr(memit_wrapper, "run_easyedit", run)
    monkeypatch.setattr(memit_wrapper, "evaluate_post_edit_reliability", lambda model, method, edits, **kwargs: {"evaluated": len(edits)})
    edits = [{**FACT, "subject": str(index), "rephrase_prompts": [f"Rephrase {index}"]} for index in range(105)]
    result = memit_wrapper.run_memit_edit(
        method="memit",
        model_id="demo", edits=edits, layers=[0], evaluation_prompts=[],
        locality_inputs=None, portability_inputs=None, sequential_edit=False,
        mom2_adjustment=True, batch_size=20,
    )
    assert recorded["batch_size"] == 20
    assert recorded["edits"] == edits
    if not worker:
        assert recorded["rephrase_prompts"] == [f"Rephrase {index}" for index in range(105)]
        assert result["edited_facts"] == result["evaluated"] == 105
