from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app import config
from app.api import routes_evaluation as routes
from app.db.schemas import ActiveModelEvaluationRequest, ReliabilityEvaluationRequest
from app.services.evaluation_service import EvaluationService

FACT = {
    "key": "fact_demo",
    "subject": "A",
    "relation_prompt": "{} is",
    "prompt": "A is",
    "new_object": "B",
    "ground_truth": "C",
    "rephrase_prompt": "What is A?",
    "portability_prompt": "A's derived fact",
    "portability_ground_truth": "Z",
}


@pytest.fixture
def route_state(monkeypatch):
    state = SimpleNamespace(inputs=[], jobs=[], started=[])
    monkeypatch.setattr(config, "settings", SimpleNamespace(deepseek_api_key=None))
    monkeypatch.setattr(routes.model_runtime_service, "require_loaded_runtime_bundle", lambda _: {})
    monkeypatch.setattr(routes.patch_service, "active_patches_for_model", lambda _: ["patch_demo"])
    monkeypatch.setattr(routes.patch_service, "get_patch", lambda _: {"method": "memit"})
    monkeypatch.setattr(routes.patch_service, "active_evaluation_facts", lambda _: [FACT])

    def create(_kind, _model_id, payload):
        state.inputs.append(payload)
        return {"id": "exp_demo"}

    def create_job(experiment_id):
        state.jobs.append(experiment_id)
        return {"id": "job_demo"}

    monkeypatch.setattr(routes.experiment_service, "create", create)
    monkeypatch.setattr(routes.experiment_service, "create_job", create_job)
    monkeypatch.setattr(routes, "_start_active_model_evaluation", lambda **kwargs: state.started.append(kwargs))
    monkeypatch.setattr(routes, "_start_reliability_evaluation", lambda **kwargs: state.started.append(kwargs))
    return state


@pytest.mark.parametrize("protocols", [
    {"reliability": "unknown"},
    {"unknown": "traditional"},
    {"locality": "LLM-judge"},
])
@pytest.mark.parametrize("request_schema", [ActiveModelEvaluationRequest, ReliabilityEvaluationRequest])
def test_request_rejects_invalid_protocols(protocols, request_schema):
    kwargs = {"method": "memit", "edits": [FACT]} if request_schema is ReliabilityEvaluationRequest else {}
    with pytest.raises(ValidationError):
        request_schema(evaluation_modes=["reliability"], evaluation_protocols=protocols, **kwargs)


@pytest.mark.parametrize("metric", ["reliability", "generalization", "portability"])
@pytest.mark.parametrize("reliability_endpoint", [False, True])
def test_generate_text_is_rejected_by_both_request_schemas_and_endpoints_before_job_creation(
    route_state, metric, reliability_endpoint
):
    payload = {
        "model_id": "demo",
        "evaluation_modes": [metric],
        "evaluation_protocols": {metric: "generate-text"},
    }
    if reliability_endpoint:
        payload.update(method="memit", edits=[FACT])
    request_schema = ReliabilityEvaluationRequest if reliability_endpoint else ActiveModelEvaluationRequest
    with pytest.raises(ValidationError):
        request_schema.model_validate(payload)
    app = FastAPI()
    app.include_router(routes.router)
    endpoint = "/evaluate/reliability" if reliability_endpoint else "/evaluate/active-model"
    with TestClient(app) as client:
        response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert route_state.inputs == route_state.jobs == route_state.started == []


@pytest.mark.parametrize("reliability_endpoint", [False, True])
def test_locality_generate_text_is_accepted_by_both_endpoints_without_judge_credentials(
    route_state, monkeypatch, reliability_endpoint
):
    locality_metadata = {"locality_prompt": "An unrelated fact", "locality_ground_truth": "C"}
    monkeypatch.setattr(routes.patch_service, "active_evaluation_facts", lambda _: [{**FACT, **locality_metadata}])
    monkeypatch.setattr(
        routes, "require_deepseek_api_key",
        lambda: pytest.fail("Locality generation must not require judge credentials"),
    )
    payload = {
        "model_id": "demo",
        "evaluation_modes": ["locality"],
        "evaluation_protocols": {"locality": "generate-text"},
    }
    if reliability_endpoint:
        payload.update(method="memit", edits=[{**FACT, "source_metadata": locality_metadata}])
    request_schema = ReliabilityEvaluationRequest if reliability_endpoint else ActiveModelEvaluationRequest
    request = request_schema.model_validate(payload)
    assert request.evaluation_protocols == {"locality": "generate-text"}
    app = FastAPI()
    app.include_router(routes.router)
    endpoint = "/evaluate/reliability" if reliability_endpoint else "/evaluate/active-model"
    with TestClient(app) as client:
        response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert route_state.inputs[0]["evaluation_protocols"] == {"locality": "generate-text"}
    assert route_state.jobs == ["exp_demo"]
    assert route_state.started[0]["evaluation_protocols"] == {"locality": "generate-text"}


def test_omitted_protocols_default_to_traditional_and_portability_is_evaluated(route_state):
    response = routes.evaluate_active_model(ActiveModelEvaluationRequest(
        model_id="demo", evaluation_modes=["portability", "reliability", "portability"]
    ))
    assert response.status == "running"
    expected = {"portability": "traditional", "reliability": "traditional"}
    assert route_state.inputs[0]["evaluation_protocols"] == expected
    worker = route_state.started[0]
    assert worker["evaluation_protocols"] == expected
    assert worker["modes"] == ["portability", "reliability"]
    assert worker["needs_edit_quality"] is True
    assert worker["edits"][0]["portability_prompt"] == FACT["portability_prompt"]
    assert worker["edits"][0]["portability_ground_truth"] == "Z"


@pytest.mark.parametrize("reliability_endpoint", [False, True])
def test_judge_requires_backend_key_before_creating_job(route_state, reliability_endpoint):
    kwargs = {
        "model_id": "demo",
        "evaluation_modes": ["reliability"],
        "evaluation_protocols": {"reliability": "LLM-judge"},
    }
    request = (
        ReliabilityEvaluationRequest(method="memit", edits=[FACT], **kwargs)
        if reliability_endpoint else ActiveModelEvaluationRequest(**kwargs)
    )
    endpoint = routes.evaluate_reliability if reliability_endpoint else routes.evaluate_active_model
    with pytest.raises(HTTPException) as error:
        endpoint(request)
    assert error.value.status_code == 400
    assert "DEEPSEEK_API_KEY" in error.value.detail
    assert route_state.inputs == route_state.jobs == route_state.started == []


def test_judge_key_is_backend_only_and_disabled_judge_does_not_need_it(route_state, monkeypatch):
    routes.evaluate_active_model(ActiveModelEvaluationRequest(
        model_id="demo", evaluation_modes=["portability"],
        evaluation_protocols={"reliability": "LLM-judge", "portability": "traditional"},
    ))
    assert route_state.inputs[0]["evaluation_protocols"] == {"portability": "traditional"}
    secret = "test-private-deepseek-key"
    monkeypatch.setattr(config, "settings", SimpleNamespace(deepseek_api_key=SecretStr(secret)))
    routes.evaluate_active_model(ActiveModelEvaluationRequest(
        model_id="demo", evaluation_modes=["reliability", "generalization"],
        evaluation_protocols={"reliability": "LLM-judge"},
    ))
    assert route_state.inputs[1]["evaluation_protocols"] == {
        "reliability": "LLM-judge", "generalization": "traditional"
    }
    assert secret not in repr(route_state.inputs)
    assert secret not in repr(route_state.started)


def test_reliability_request_keeps_legacy_rephrase_metric_default(route_state):
    routes.evaluate_reliability(ReliabilityEvaluationRequest(
        model_id="demo", method="memit", edits=[{**FACT, "rephrase_prompts": ["What is A?"]}]
    ))
    assert route_state.inputs[0]["evaluation_modes"] == ["reliability", "generalization"]
    assert route_state.started[0]["evaluation_protocols"] == {
        "reliability": "traditional", "generalization": "traditional"
    }


def test_backend_key_uses_secretstr_and_does_not_appear_in_settings_repr(monkeypatch):
    secret = "test-private-deepseek-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    settings = config.Settings()
    assert isinstance(settings.deepseek_api_key, SecretStr)
    assert settings.deepseek_api_key.get_secret_value() == secret
    assert secret not in repr(settings)
    monkeypatch.setattr(config, "settings", settings)
    assert config.require_deepseek_api_key() == secret
    monkeypatch.setenv("DEEPSEEK_API_KEY", "  ")
    assert config.Settings().deepseek_api_key is None


def test_legacy_generate_text_results_keep_protocols_portability_scores_and_unavailable_scores():
    protocols = {"reliability": "generate-text", "portability": "traditional"}
    result, overview, pages = routes._build_paged_evaluation_result({
        "model_id": "demo", "mean_reliability": None, "mean_portability": 0.75,
        "evaluation_protocols": protocols,
        "cases": [{"reliability_score": None, "portability_score": 0.75}],
        "sample_generations": [{"prompt": "A is", "active_model": "B"}],
        "sample_portability_generations": [{"prompt": FACT["portability_prompt"], "active_model": "Z"}],
    }, [FACT], ["reliability", "portability"])
    assert overview["metrics"] == {"reliability_rewrite_acc": None, "portability_score": 0.75}
    assert result["evaluation_protocols"] == overview["evaluation_protocols"] == protocols
    item = pages[0]["items"][0]
    assert item["evaluation_protocols"] == protocols
    assert item["reliability_score"] is None
    assert item["portability_score"] == 0.75
    assert item["portability_generation"]["active_model"] == "Z"
    _, missing_overview, _ = routes._build_paged_evaluation_result({}, [FACT], ["portability"])
    assert missing_overview["metrics"]["portability_score"] is None


@pytest.mark.parametrize("uses_worker", [False, True])
def test_service_forwards_protocols_without_credentials(monkeypatch, uses_worker):
    import app.services.model_runtime_service as runtime_module
    from app.ml.editing import easyedit_runtime

    captured = {}

    def execute(operation, payload, **_):
        assert operation == "easyedit_edit_quality"
        captured.update(payload)
        return {"cases": []}

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"cases": []}

    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(
        uses_hard_worker=uses_worker, execute=execute
    ))
    monkeypatch.setattr(easyedit_runtime, "evaluate_easyedit_quality", evaluate)
    result = EvaluationService().evaluate_easyedit_quality(
        "demo", "memit", [FACT], evaluation_modes=["portability"],
        evaluation_protocols={"portability": "LLM-judge"},
    )
    assert result == {"cases": []}
    assert captured["evaluation_modes"] == ["portability"]
    assert captured["evaluation_protocols"] == {"portability": "LLM-judge"}
    assert not any("key" in key.lower() for key in captured)


@pytest.mark.parametrize("uses_worker", [False, True])
def test_baseline_and_evaluation_progress_cross_service_and_worker(monkeypatch, uses_worker):
    import app.services.model_runtime_service as runtime_module
    from app.ml.editing import easyedit_runtime
    from app.services.model_worker import _dispatch

    updates = []

    def evaluate(**kwargs):
        kwargs["baseline_progress_callback"](1, 2000)
        kwargs["progress_callback"](0, 2000)
        kwargs["progress_callback"](1, 2000)
        return {"cases": []}

    monkeypatch.setattr(easyedit_runtime, "evaluate_easyedit_quality", evaluate)
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(
        uses_hard_worker=uses_worker,
        execute=lambda action, payload, **kwargs: _dispatch(None, action, payload, kwargs["progress_callback"]),
    ))
    EvaluationService().evaluate_easyedit_quality(
        "demo", "memit", [FACT],
        on_baseline_progress=lambda current, total: updates.append(("baseline", current, total)),
        on_progress=lambda current, total: updates.append(("evaluation", current, total)),
    )
    assert updates == [("baseline", 1, 2000), ("evaluation", 0, 2000), ("evaluation", 1, 2000)]


def test_active_runner_forwards_enabled_modes_and_persists_protocols(monkeypatch):
    captured = {}
    completed = {}
    monkeypatch.setattr(routes.experiment_service, "get_job", lambda _: {"status": "running"})
    monkeypatch.setattr(routes.experiment_service, "append_job_log", lambda *_, **__: None)
    monkeypatch.setattr(routes.experiment_service, "set_job_progress", lambda *_, **__: None)
    monkeypatch.setattr(routes.experiment_service, "fail_job", lambda *args: pytest.fail(str(args)))

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"mean_portability": None, "cases": [{"portability_score": None}]}

    def complete(_job_id, result, overview, pages):
        completed.update(result=result, overview=overview, pages=pages)

    monkeypatch.setattr(routes.evaluation_service, "evaluate_easyedit_quality", evaluate)
    monkeypatch.setattr(routes.experiment_service, "complete_paged_evaluation_job", complete)
    routes._run_active_model_evaluation(
        job_id="job_demo", model_id="demo", method="memit", active_patch_ids=["patch_demo"],
        facts=[FACT], edits=[FACT], modes=["portability"], include_generations=False,
        needs_edit_quality=True, evaluation_protocols={"portability": "LLM-judge"},
    )
    assert captured["evaluation_modes"] == ["portability"]
    assert captured["evaluation_protocols"] == {"portability": "LLM-judge"}
    assert completed["result"]["evaluation_protocols"] == {"portability": "LLM-judge"}
    assert completed["overview"]["metrics"]["portability_score"] is None


@pytest.mark.parametrize("protocols", [
    {"reliability": "LLM-judge", "generalization": "traditional"},
    {"reliability": "LLM-judge", "generalization": "LLM-judge"},
    {"reliability": "traditional", "generalization": "LLM-judge"},
])
def test_reliability_pair_reuses_evaluator_generation_for_custom_protocols(monkeypatch, protocols):
    import app.services.evaluation_service as service_module

    monkeypatch.setattr(service_module, "generate_text", lambda *_, **__: pytest.fail("Duplicate generation"))
    service = EvaluationService()
    captured = {}
    sample = {"prompt": "A is", "clean_model": "C", "active_model": "EasyEdit generated B"}
    case = {"reliability_score": None, "evaluation_protocols": protocols}
    callbacks = []

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"cases": [case], "sample_generations": [sample], "active_patch_ids": ["patch_demo"]}

    monkeypatch.setattr(service, "evaluate_easyedit_quality", evaluate)
    pair = service.evaluate_edit_reliability_test_pair(
        "demo", "memit", FACT, include_generation=True, on_generation=callbacks.append,
        cancel_job_id="job_demo", evaluation_modes=["reliability", "generalization"],
        evaluation_protocols=protocols,
    )
    assert captured["include_generations"] is True
    assert captured["cancel_job_id"] == "job_demo"
    assert captured["evaluation_protocols"] == protocols
    assert pair == {"case": case, "generation": sample, "active_patch_ids": ["patch_demo"]}
    assert callbacks == [sample]


@pytest.mark.parametrize("protocols", [
    None,
    {"reliability": "traditional"},
    {"reliability": "traditional", "generalization": "LLM-judge"},
])
def test_traditional_reliability_pair_preserves_generate_then_score_order(monkeypatch, protocols):
    import app.services.evaluation_service as service_module

    service = EvaluationService()
    events = []

    def generate(*args, **kwargs):
        events.append("generate")
        assert kwargs["use_active_patches"] is True
        return {"text": "workbench generated B"}

    def evaluate(**kwargs):
        events.append("evaluate")
        assert kwargs["include_generations"] is False
        assert kwargs["cancel_job_id"] == "job_demo"
        return {"cases": [{"reliability_score": 1.0}]}

    monkeypatch.setattr(service_module, "generate_text", generate)
    monkeypatch.setattr(service, "evaluate_easyedit_quality", evaluate)
    pair = service.evaluate_edit_reliability_test_pair(
        "demo", "memit", FACT, on_generation=lambda _: events.append("callback"),
        cancel_job_id="job_demo", evaluation_modes=["reliability"], evaluation_protocols=protocols,
    )
    assert events == ["generate", "callback", "evaluate"]
    assert pair["generation"] == {"prompt": "A is", "active_model": "workbench generated B"}


def test_custom_reliability_pair_can_skip_generation_display(monkeypatch):
    import app.services.evaluation_service as service_module

    monkeypatch.setattr(service_module, "generate_text", lambda *_, **__: pytest.fail("Duplicate generation"))
    service = EvaluationService()

    def evaluate(**kwargs):
        assert kwargs["include_generations"] is False
        return {"cases": [{"reliability_score": None}]}

    monkeypatch.setattr(service, "evaluate_easyedit_quality", evaluate)
    pair = service.evaluate_edit_reliability_test_pair(
        "demo", "memit", FACT, include_generation=False,
        on_generation=lambda _: pytest.fail("No display generation was requested"),
        evaluation_protocols={"reliability": "LLM-judge"},
    )
    assert pair["generation"] is None
