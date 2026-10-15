from __future__ import annotations

import contextlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import routes_evaluation as routes
from app.db.schemas import ActiveModelEvaluationRequest
from app.ml.editing import easyedit_runtime as runtime
from app.services.model_runtime_service import ModelNotLoadedError

UPLOADED_FACT = {
    "key": '["Atlas is in","Paris"]',
    "subject": "Atlas",
    "prompt": "Atlas is in",
    "relation_prompt": "{} is in",
    "new_object": "Paris",
    "ground_truth": "London",
}
ACTIVE_FACT = {
    **UPLOADED_FACT,
    "key": "patch_active:0",
    "subject": "Other",
    "prompt": "Other is in",
    "new_object": "Rome",
}


@pytest.fixture
def route_state(monkeypatch):
    state = SimpleNamespace(
        patches=["patch_active"], inputs=[], started=[], loaded=[], active_fact_reads=[],
    )
    monkeypatch.setattr(
        routes.model_runtime_service, "require_loaded_runtime_bundle",
        lambda model_id: state.loaded.append(model_id) or {},
    )
    monkeypatch.setattr(
        routes.patch_service, "active_patches_for_model", lambda _: list(state.patches),
    )
    monkeypatch.setattr(routes.patch_service, "get_patch", lambda _: {"method": "memit"})
    monkeypatch.setattr(
        routes.patch_service, "active_evaluation_facts",
        lambda model_id: state.active_fact_reads.append(model_id) or [deepcopy(ACTIVE_FACT)],
    )
    for name in (
        "create_patch", "update_patch", "activate_patch", "deactivate_all_patches",
        "clear_active_patches", "restore_active_patches", "sync_patch_stack_to_model",
    ):
        monkeypatch.setattr(
            routes.patch_service, name,
            lambda *_, **__: pytest.fail("Selecting evaluation facts must not mutate patches"),
        )

    def create(_kind, _model_id, payload):
        state.inputs.append(deepcopy(payload))
        return {"id": "exp_upload"}

    monkeypatch.setattr(routes.experiment_service, "create", create)
    monkeypatch.setattr(
        routes.experiment_service, "create_job", lambda _: {"id": "job_upload"},
    )
    monkeypatch.setattr(
        routes, "_start_active_model_evaluation", lambda **kwargs: state.started.append(kwargs),
    )
    return state


def request_for(facts=None, **kwargs):
    return ActiveModelEvaluationRequest(
        model_id="demo", facts=facts, evaluation_modes=["reliability"], **kwargs,
    )


def test_uploaded_facts_replace_active_fact_selection_without_changing_patch_stack(route_state):
    response = routes.evaluate_active_model(request_for(
        [UPLOADED_FACT], fact_source_id="upload_1",
    ))
    assert response.status == "running"
    assert route_state.loaded == ["demo"]
    assert route_state.active_fact_reads == []
    assert route_state.patches == ["patch_active"]
    worker = route_state.started[0]
    assert worker["facts"] == [UPLOADED_FACT]
    assert worker["edits"][0]["prompt"] == UPLOADED_FACT["prompt"]
    assert worker["edits"][0]["new_object"] == "Paris"
    assert worker["method"] == "memit"
    assert worker["active_patch_ids"] == ["patch_active"]
    metadata = route_state.inputs[0]
    assert metadata["source"] == "evaluation_page"
    assert metadata["fact_source"] == "uploaded"
    assert metadata["fact_source_id"] == "upload_1"
    assert metadata["edit_count"] == 1
    assert not any(key in metadata for key in ("facts", "edits", "evaluation_facts"))


def test_uploaded_facts_support_loaded_model_without_active_patches(route_state, monkeypatch):
    route_state.patches = []
    monkeypatch.setattr(
        routes.patch_service, "get_patch", lambda _: pytest.fail("No active patch to resolve"),
    )
    routes.evaluate_active_model(request_for([UPLOADED_FACT]))
    worker = route_state.started[0]
    assert worker["method"] == "rome"
    assert worker["active_patch_ids"] == []
    assert route_state.active_fact_reads == []


def test_uploaded_auxiliary_fields_overrides_and_exact_prompt_are_forwarded(route_state):
    fact = {
        **UPLOADED_FACT,
        "subject": " Atlas ",
        "prompt": "  Where does {this} belong?  ",
        "relation_prompt": "  ",
        "ground_truth": " ",
        "rephrase_prompt": " Original rephrase ",
        "locality_prompt": " Nearby is in ",
        "locality_ground_truth": " Rome ",
        "portability_prompt": " Atlas belongs to ",
        "portability_ground_truth": " France ",
    }
    request = ActiveModelEvaluationRequest(
        model_id="demo", facts=[fact],
        evaluation_modes=["reliability", "generalization", "locality", "portability"],
        fluency_modes=["portability"],
        evaluation_protocols={"locality": "generate-text"},
        rephrase_overrides={fact["key"]: " Override rephrase "},
    )
    routes.evaluate_active_model(request)
    worker = route_state.started[0]
    edit = worker["edits"][0]
    assert edit == {
        "subject": "Atlas", "prompt": "Where does {this} belong?",
        "relation_prompt": "Where does {this} belong?", "new_object": "Paris",
        "ground_truth": None, "rephrase_prompts": ["Override rephrase"],
        "locality_prompt": "Nearby is in", "locality_ground_truth": "Rome",
        "portability_prompt": "Atlas belongs to", "portability_ground_truth": "France",
    }
    assert worker["include_generations"] is True
    assert worker["fluency_modes"] == ["portability"]
    assert worker["evaluation_protocols"]["locality"] == "generate-text"


@pytest.mark.parametrize("rephrase, expected", [
    (" A different question ", "A different question"),
    ("  ", UPLOADED_FACT["prompt"]),
    (None, UPLOADED_FACT["prompt"]),
])
def test_uploaded_generalization_uses_rephrase_then_original_prompt(route_state, rephrase, expected):
    request = ActiveModelEvaluationRequest(
        model_id="demo", facts=[{**UPLOADED_FACT, "rephrase_prompt": rephrase}],
        evaluation_modes=["generalization"],
    )
    routes.evaluate_active_model(request)
    assert route_state.started[0]["edits"][0]["rephrase_prompts"] == [expected]


def test_uploaded_duplicate_identity_merges_auxiliary_facts_and_preserves_conflicting_targets(route_state):
    first = {**UPLOADED_FACT, "ground_truth": "", "rephrase_prompt": UPLOADED_FACT["prompt"]}
    duplicate = {
        **UPLOADED_FACT, "key": "duplicate", "rephrase_prompt": "Where is Atlas?",
        "locality_prompt": "Nearby", "locality_ground_truth": "Rome",
        "portability_prompt": "Derived", "portability_ground_truth": "France",
    }
    conflict = {**UPLOADED_FACT, "key": "conflict", "new_object": "Berlin"}
    routes.evaluate_active_model(request_for([first, duplicate, conflict]))
    facts = route_state.started[0]["facts"]
    assert len(facts) == route_state.inputs[0]["edit_count"] == 2
    assert facts[0] == {**duplicate, "key": first["key"]}
    assert facts[1] == conflict


@pytest.mark.parametrize("facts", [
    [],
    [{**UPLOADED_FACT, "subject": "  "}],
    [{**UPLOADED_FACT, "prompt": "  "}],
    [{**UPLOADED_FACT, "new_object": "  "}],
    [{**UPLOADED_FACT, "key": ""}],
    [{**UPLOADED_FACT, "subject": None}],
    [{**UPLOADED_FACT, "new_object": 123}],
    [{**UPLOADED_FACT, "rephrase_prompt": ["invalid"]}],
    [{**UPLOADED_FACT, "ground_truth": False}],
    [{key: value for key, value in UPLOADED_FACT.items() if key != "prompt"}],
])
def test_invalid_uploaded_facts_are_rejected_before_job_creation(route_state, facts):
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.post("/evaluate/active-model", json={
            "model_id": "demo", "facts": facts, "evaluation_modes": ["reliability"],
        })
    assert response.status_code == 422
    assert route_state.inputs == route_state.started == []


def test_uploaded_ground_truth_can_be_omitted(route_state):
    fact = {key: value for key, value in UPLOADED_FACT.items() if key != "ground_truth"}
    routes.evaluate_active_model(request_for([fact]))
    assert route_state.started[0]["facts"][0]["ground_truth"] == ""
    assert route_state.started[0]["edits"][0]["ground_truth"] is None


def test_omitted_upload_keeps_active_patch_evaluation_behavior(route_state):
    routes.evaluate_active_model(request_for(fact_source_id="unused_upload"))
    worker = route_state.started[0]
    assert route_state.active_fact_reads == ["demo"]
    assert worker["facts"] == [ACTIVE_FACT]
    assert "prompt" not in worker["edits"][0]
    assert worker["fact_source"] == route_state.inputs[0]["fact_source"] == "active_patches"
    assert "fact_source_id" not in route_state.inputs[0]


def test_omitted_upload_still_requires_a_patch(route_state):
    route_state.patches = []
    with pytest.raises(HTTPException) as error:
        routes.evaluate_active_model(request_for())
    assert error.value.status_code == 400
    assert route_state.inputs == route_state.started == []


def test_uploaded_evaluation_still_requires_a_loaded_model(route_state, monkeypatch):
    def require_loaded(_model_id):
        raise ModelNotLoadedError("Load the model first")

    monkeypatch.setattr(routes.model_runtime_service, "require_loaded_runtime_bundle", require_loaded)
    with pytest.raises(HTTPException) as error:
        routes.evaluate_active_model(request_for([UPLOADED_FACT]))
    assert error.value.status_code == 409
    assert route_state.inputs == route_state.started == []


def test_uploaded_runner_preserves_pipeline_progress_and_paged_result_source(route_state, monkeypatch):
    routes.evaluate_active_model(request_for(
        [UPLOADED_FACT], fact_source_id="upload_1", fluency_modes=["reliability"],
    ))
    state = SimpleNamespace(evaluations=[], progress=[], completed=[])
    monkeypatch.setattr(routes.experiment_service, "get_job", lambda _: {"status": "running"})
    monkeypatch.setattr(routes.experiment_service, "append_job_log", lambda *_, **__: None)
    monkeypatch.setattr(
        routes.experiment_service, "set_job_progress",
        lambda *args, **kwargs: state.progress.append((args, kwargs)),
    )
    monkeypatch.setattr(
        routes.experiment_service, "fail_job", lambda *args: pytest.fail(str(args)),
    )

    def evaluate(**kwargs):
        state.evaluations.append(kwargs)
        kwargs["on_progress"](1, 1)
        return {
            "model_id": "demo", "method": "memit", "active_patch_ids": ["patch_active"],
            "cases": [{"reliability_score": 0.75, "fluency": {"reliability": {"ngram_entropy": 2}}}],
            "mean_reliability": 0.75,
            "fluency_modes": ["reliability"], "mean_fluency": {"reliability": 2},
            "sample_generations": [{"prompt": UPLOADED_FACT["prompt"], "active_model": "Paris"}],
        }

    monkeypatch.setattr(routes.evaluation_service, "evaluate_easyedit_quality", evaluate)
    monkeypatch.setattr(
        routes.experiment_service, "complete_paged_evaluation_job",
        lambda *args: state.completed.append(args),
    )
    routes._run_active_model_evaluation(**route_state.started[0])
    evaluated = state.evaluations[0]
    assert evaluated["edits"][0]["prompt"] == UPLOADED_FACT["prompt"]
    assert evaluated["cancel_job_id"] == "job_upload"
    assert evaluated["fluency_modes"] == ["reliability"]
    assert evaluated["evaluation_protocols"] == {"reliability": "traditional"}
    assert state.progress[-1] == (
        ("job_upload", 1.0), {"progress_current": 1, "progress_total": 1, "progress_phase": "evaluation"},
    )
    job_id, result, overview, pages = state.completed[0]
    assert job_id == "job_upload"
    assert result["evaluation_facts"] == [UPLOADED_FACT]
    assert result["fact_source"] == overview["fact_source"] == "uploaded"
    assert result["fact_source_id"] == overview["fact_source_id"] == "upload_1"
    assert overview["active_patch_ids"] == ["patch_active"]
    assert overview["metrics"]["reliability_fluency_ngram_entropy"] == 2
    assert pages[0]["items"][0]["fact"] == UPLOADED_FACT
    assert pages[0]["items"][0]["reliability_generation"]["active_model"] == "Paris"


@pytest.mark.parametrize("supplied_prompt, expected_prompt", [
    ("Which country contains it?", "Which country contains it?"),
    ("Explain {this} with {}", "Explain {this} with {}"),
    (None, "Atlas is in"),
])
def test_runtime_uses_uploaded_exact_prompt_and_preserves_legacy_template(
    monkeypatch, supplied_prompt, expected_prompt,
):
    hparams = SimpleNamespace(model_name="demo", device="cpu", alg_name="ROME")
    model, tokenizer = object(), object()
    calls = []
    syncs = []
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {"hf_id": "demo", "supports_rome": True})
    monkeypatch.setattr(
        runtime, "_load_easyedit", lambda _: (None, SimpleNamespace(from_hparams=lambda _: hparams)),
    )
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: Path("unused"))
    monkeypatch.setattr(runtime, "_configure_loaded_tokenizer", lambda *_: None)
    monkeypatch.setattr(runtime, "_override_easyedit_device", lambda *_: None)
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", lambda _: contextlib.nullcontext({
        "model": model, "tokenizer": tokenizer, "device": "cpu",
    }))
    monkeypatch.setattr(runtime.patch_service, "active_patches_for_model", lambda _: ["patch_active"])
    monkeypatch.setattr(
        runtime.patch_service, "sync_active_patches_to_model", lambda *args: syncs.append(args),
    )
    monkeypatch.setattr(
        runtime.patch_service, "sync_patch_stack_to_model",
        lambda *_: pytest.fail("Reliability must retain the active patch stack"),
    )

    def compute(_model, _name, _hparams, _tokenizer, record, _device):
        calls.append(deepcopy(record))
        return {"rewrite_acc": [1.0]}

    monkeypatch.setitem(sys.modules, "easyeditor.evaluate", SimpleNamespace(compute_edit_quality=compute))
    monkeypatch.setitem(
        sys.modules, "easyeditor.editors.editor", SimpleNamespace(finalize_locality_metrics=lambda *_: None),
    )
    edit = {**UPLOADED_FACT}
    if supplied_prompt is not None:
        edit["prompt"] = supplied_prompt
        edit["relation_prompt"] = "{} has {literal} braces"
    else:
        edit.pop("prompt")
    result = runtime.evaluate_easyedit_quality("demo", "rome", [edit], evaluation_modes=["reliability"])
    assert calls == [{"prompt": expected_prompt, "target_new": "Paris", "ground_truth": "London"}]
    assert result["cases"][0]["prompt"] == expected_prompt
    assert result["mean_reliability"] == 1.0
    assert syncs == [(model, "demo")]


def test_compact_experiment_status_retains_uploaded_source_identity(monkeypatch, tmp_path):
    from app.services.experiment_service import ExperimentService
    from app.storage.local_store import local_store

    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    experiment = service.create("evaluation", "demo", {
        "source": "evaluation_page", "fact_source": "uploaded", "fact_source_id": "upload-1919",
        "active_patch_ids": ["patch_active"], "edit_count": 1919,
        "facts": [UPLOADED_FACT] * 1919,
    })
    job = service.create_job(experiment["id"])
    service.set_job_progress(job["id"], 0.2, 19, 1919, progress_phase="baseline")
    status = service.get_experiment_details(experiment["id"], include_result=False)
    assert status["status"] == "running"
    assert status["input_json"]["fact_source"] == "uploaded"
    assert status["input_json"]["fact_source_id"] == "upload-1919"
    assert status["input_json"]["active_patch_ids"] == ["patch_active"]
    assert "facts" not in status["input_json"]
    assert "result" not in status
    assert status["job"]["progress_current"] == 19
    assert status["job"]["progress_phase"] == "baseline"
    service.set_job_progress(job["id"], 0.2, 19, 1919, progress_phase="evaluation")
    assert service.get_experiment_details(experiment["id"], include_result=False)["job"]["progress_phase"] == "evaluation"
