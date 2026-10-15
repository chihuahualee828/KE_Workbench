import contextlib
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app.ml.editing import easyedit_runtime as runtime
from app.ml.editing.evaluation_reuse import matching_post


FACT = {"subject": "A", "relation_prompt": "{} is", "new_object": "B", "ground_truth": "C"}


def meta(section, *_):
    return {"result_key": f"{section}_acc", "protocol": "teacher_forcing"}


def step(fact=FACT):
    return {"requested_rewrite": {"prompt": runtime.build_prompt(fact["subject"], fact["relation_prompt"]),
                                  "target_new": fact["new_object"], "ground_truth": fact.get("ground_truth")},
            "post": {"rewrite_acc": [1.0], "metric_meta": {"rewrite": meta("rewrite")}}}


@pytest.fixture
def evaluator(monkeypatch):
    import app.services.model_runtime_service as runtime_module

    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    state = SimpleNamespace(scores=[], generations=[], fluency=[], stacks=[], returned_text=None, resource_checks=0)
    def prepare_fluency():
        state.resource_checks += 1
    monkeypatch.setattr(runtime, "ensure_fluency_tokenizer_data", prepare_fluency)
    hparams = SimpleNamespace(model_name="demo", device="cpu", alg_name="MEMIT")
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {"hf_id": "demo", "supports_memit": True})
    monkeypatch.setattr(runtime, "_load_easyedit", lambda _: (None, SimpleNamespace(from_hparams=lambda _: hparams)))
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: Path("unused"))
    monkeypatch.setattr(runtime, "_configure_loaded_tokenizer", lambda *_: None)
    monkeypatch.setattr(runtime, "_override_easyedit_device", lambda *_: None)
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", lambda _: contextlib.nullcontext({"model": object(), "tokenizer": object(), "device": "cpu"}))
    monkeypatch.setattr(runtime.patch_service, "active_patches_for_model", lambda _: ["patch_new"])
    monkeypatch.setattr(runtime.patch_service, "sync_active_patches_to_model", lambda *_: state.stacks.append("active"))
    monkeypatch.setattr(runtime.patch_service, "sync_patch_stack_to_model", lambda *_: state.stacks.append("original"))

    def fluency(**kwargs):
        state.fluency.append(kwargs["prefixes"][0])
        return {"ngram_entropy": 2.5}

    def compute(model, name, params, tok, record, device, **kwargs):
        state.scores.append((deepcopy(record), kwargs))
        result = {"rewrite_acc": [0.5], "metric_meta": {"rewrite": meta("rewrite")}, "locality": {}}
        if record.get("rephrase_prompt"):
            result["rephrase_acc"] = [0.25]
        if record.get("locality"):
            result["locality"] = {"neighborhood_output": [[1, 2]]}
        if kwargs.get("test_generation"):
            result["fluency"] = fluency(prefixes=[record["prompt"]])
        if state.returned_text:
            result["rewrite_gen_content"] = [state.returned_text]
        return result

    def finalize(metric, *_):
        metric["post"]["locality"]["neighborhood_acc"] = [1.0]

    monkeypatch.setitem(sys.modules, "easyeditor.evaluate", SimpleNamespace(compute_edit_quality=compute))
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.metric_meta", SimpleNamespace(build_lm_metric_meta=meta))
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.evaluate_utils", SimpleNamespace(test_generation_quality=fluency))
    monkeypatch.setitem(sys.modules, "easyeditor.editors.editor", SimpleNamespace(finalize_locality_metrics=finalize))
    from app.ml.inference import generate
    def generation(model_id, prompt, **kwargs):
        state.generations.append((prompt, kwargs["use_active_patches"]))
        return {"text": f"{prompt}: {kwargs['use_active_patches']}"}
    monkeypatch.setattr(generate, "generate_text", generation)
    state.hparams = hparams
    return state


@pytest.mark.parametrize("count", [1, 2])
def test_single_fact_or_batch_reuses_scores_but_runs_fluency(evaluator, count, capsys):
    result = runtime.evaluate_post_edit_reliability("demo", "memit", [FACT] * count,
        edit_result={"metrics": [step()] * count}, single_batch=True)
    assert evaluator.scores == []
    assert evaluator.fluency == ["A is"]
    assert evaluator.resource_checks == 1
    assert evaluator.generations == [("A is", False), ("A is", True)]
    assert result["reliability"]["mean_reliability"] == 1.0
    assert result["reliability"]["fluency_modes"] == ["reliability"]
    assert result["reliability"]["mean_fluency"] == {"reliability": 2.5}
    assert all(case["score_source"] == "internal_post" for case in result["reliability"]["cases"])
    assert result["generations"][0]["evaluation"]["fluency"]["reliability"]["ngram_entropy"] == 2.5
    assert f"Post-edit evaluation {count}/{count}." in capsys.readouterr().out


def test_multi_batch_scores_final_state_and_retains_step_scores(evaluator):
    result = runtime.evaluate_post_edit_reliability("demo", "memit", [FACT, {**FACT, "subject": "D"}],
        edit_result={"metrics": [step(), step({**FACT, "subject": "D"})]})
    assert len(evaluator.scores) == 2
    assert all(kwargs["test_generation"] is True for _, kwargs in evaluator.scores)
    assert result["reliability"]["mean_reliability"] == 0.5
    assert result["reliability"]["mean_fluency"] == {"reliability": 2.5}
    assert all(case["internal_reliability_score"] == 1.0 for case in result["reliability"]["cases"])
    assert all(generation["evaluation"]["fluency"]["reliability"]["ngram_entropy"] == 2.5 for generation in result["generations"])


@pytest.mark.parametrize("mismatch", ["prompt", "target_new", "protocol", "locality"])
def test_incompatible_step_is_not_reused(evaluator, mismatch):
    entry = step()
    record = dict(entry["requested_rewrite"])
    if mismatch == "protocol":
        entry["post"]["metric_meta"]["rewrite"]["protocol"] = "other"
    elif mismatch == "locality":
        record["locality"] = {"neighborhood": {"prompt": "elsewhere", "ground_truth": "C"}}
    else:
        entry["requested_rewrite"][mismatch] = "different"
    assert matching_post([entry], 0, record, evaluator.hparams, "demo") is None


def test_returned_generation_is_reused(evaluator):
    evaluator.returned_text = "EasyEdit generated answer"
    result = runtime.evaluate_easyedit_quality("demo", "memit", [FACT], include_generations=True)
    assert evaluator.generations == [("A is", False)]
    assert result["sample_generations"][0]["active_model"] == evaluator.returned_text
    assert evaluator.resource_checks == 0


def test_all_fluency_prompt_types_and_missing_prompts(evaluator):
    facts = [{**FACT, "rephrase_prompts": ["A is"], "locality_prompt": "nearby", "locality_ground_truth": "C", "portability_prompt": "derived"}, FACT]
    result = runtime.evaluate_easyedit_quality("demo", "memit", facts, include_generations=True,
        fluency_modes=["reliability", "generalization", "locality", "portability"])
    assert evaluator.fluency == ["A is", "nearby", "derived"]
    assert evaluator.resource_checks == 1
    assert len(evaluator.generations) == len(set(evaluator.generations)) == 4
    assert set(result["cases"][0]["fluency"]) == {"reliability", "generalization", "locality", "portability"}
    assert set(result["cases"][1]["fluency"]) == {"reliability"}
    assert result["mean_fluency"]["portability"] == 2.5
    assert result["cases"][0]["locality_score"] == 1.0
    assert "original" in evaluator.stacks


def test_post_edit_progress_is_exposed_in_job_response(monkeypatch):
    from app.api import routes_editing
    import io
    updates = []
    monkeypatch.setattr(routes_editing.experiment_service, "append_job_log", lambda *args, **kwargs: updates.append(kwargs))
    stream = routes_editing._JobLogStream("job_demo", io.StringIO())
    stream.write("Post-edit evaluation 0/10.\nPost-edit evaluation 3/10.\n")
    assert updates[0]["progress_current"] == 0
    assert updates[1]["progress_current"] == 3
    assert updates[1]["progress_phase"] == "post_edit_evaluation"
    response = routes_editing._job_response({"id": "job_demo", "experiment_id": "exp_demo", "status": "running", "progress_phase": "post_edit_evaluation", "progress_current": 3, "progress_total": 10})
    assert response.progress_phase == "post_edit_evaluation"
    assert response.progress_current == 3


def test_paged_results_keep_fluency_and_metric_order():
    from app.api.routes_evaluation import _build_paged_evaluation_result
    _, overview, pages = _build_paged_evaluation_result({
        "mean_reliability": 1, "mean_generalization": 0.5,
        "fluency_modes": ["reliability", "generalization"], "mean_fluency": {"reliability": 2, "generalization": 3},
        "cases": [{"fluency": {"reliability": {"ngram_entropy": 2}}}],
    }, [FACT], ["generalization", "reliability"])
    assert list(overview["metrics"]) == ["reliability_rewrite_acc", "reliability_fluency_ngram_entropy", "generalization_rephrase_acc", "generalization_fluency_ngram_entropy"]
    assert pages[0]["items"][0]["fluency"]["reliability"]["ngram_entropy"] == 2


def test_active_model_route_passes_portability_fluency(monkeypatch):
    from app.api import routes_evaluation as routes
    from app.db.schemas import ActiveModelEvaluationRequest
    fact = {**FACT, "key": "fact", "prompt": "A is", "portability_prompt": "derived", "portability_ground_truth": "Z"}
    captured = {}
    monkeypatch.setattr(routes.model_runtime_service, "require_loaded_runtime_bundle", lambda _: {})
    monkeypatch.setattr(routes.patch_service, "active_patches_for_model", lambda _: ["patch_new"])
    monkeypatch.setattr(routes.patch_service, "get_patch", lambda _: {"method": "memit"})
    monkeypatch.setattr(routes.patch_service, "active_evaluation_facts", lambda _: [fact])
    monkeypatch.setattr(routes.experiment_service, "create", lambda *args: {"id": "exp_test"})
    monkeypatch.setattr(routes.experiment_service, "create_job", lambda _: {"id": "job_test"})
    monkeypatch.setattr(routes, "_start_active_model_evaluation", lambda **kwargs: captured.update(kwargs))
    routes.evaluate_active_model(ActiveModelEvaluationRequest(model_id="demo", evaluation_modes=["portability"], fluency_modes=["portability"]))
    assert captured["fluency_modes"] == ["portability"]
    assert captured["needs_edit_quality"] is True
    assert captured["edits"][0]["portability_prompt"] == "derived"
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        routes.evaluate_active_model(ActiveModelEvaluationRequest(model_id="demo", evaluation_modes=["reliability"], fluency_modes=["portability"]))


def test_missing_nltk_data_does_not_discard_scores(evaluator, monkeypatch):
    def missing(**_):
        raise LookupError("punkt_tab missing")
    monkeypatch.setattr(sys.modules["easyeditor.evaluate.evaluate_utils"], "test_generation_quality", missing)
    result = runtime.evaluate_post_edit_reliability("demo", "memit", [FACT], edit_result={"metrics": [step()]})
    assert result["reliability"]["mean_reliability"] == 1.0
    assert "error" in result["reliability"]["cases"][0]["fluency"]["reliability"]


@pytest.mark.parametrize("reuse", [False, True])
def test_failed_automatic_fluency_setup_keeps_scores_and_generations(evaluator, monkeypatch, reuse):
    def unavailable():
        evaluator.resource_checks += 1
        raise runtime.FluencyResourceError("Could not automatically prepare NLTK data: network unavailable")
    monkeypatch.setattr(runtime, "ensure_fluency_tokenizer_data", unavailable)
    facts = [FACT, {**FACT, "subject": "D"}]
    result = runtime.evaluate_post_edit_reliability("demo", "memit", facts,
        edit_result={"metrics": [step(fact) for fact in facts]}, single_batch=reuse)
    assert evaluator.resource_checks == 1
    assert evaluator.fluency == []
    assert all(not kwargs.get("test_generation") for _, kwargs in evaluator.scores)
    assert result["reliability"]["mean_reliability"] == (1.0 if reuse else 0.5)
    assert result["reliability"]["mean_fluency"]["reliability"] is None
    assert len(result["generations"]) == 2
    assert all("network unavailable" in case["fluency"]["reliability"]["error"] for case in result["reliability"]["cases"])


def test_missing_optional_metric_prompts_do_not_prepare_nltk(evaluator):
    result = runtime.evaluate_easyedit_quality("demo", "memit", [FACT], fluency_modes=["locality", "portability"])
    assert evaluator.resource_checks == 0
    assert result["cases"][0]["fluency"] == {}


def test_phase_survives_job_summary_and_progress_resets(monkeypatch, tmp_path):
    from app.services.experiment_service import ExperimentService
    from app.storage.local_store import local_store
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    experiment = service.create("memit_edit", "demo", {})
    job = service.create_job(experiment["id"])
    service.append_job_log(job["id"], "Edited", progress=1, progress_current=5, progress_total=5)
    service.append_job_log(job["id"], "Post-edit evaluation 0/5.", progress=0, progress_current=0, progress_total=5, progress_phase="post_edit_evaluation")
    current = service.get_job(job["id"])
    assert current["status"] == "running"
    assert current["progress_current"] == 0
    assert current["progress_phase"] == "post_edit_evaluation"
