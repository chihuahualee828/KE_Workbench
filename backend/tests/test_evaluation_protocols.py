from __future__ import annotations

import ast
import contextlib
import json
import sys
import traceback
import typing
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app import config
from app.ml.editing import easyedit_runtime as runtime

TEST_API_KEY = "regression-test-deepseek-secret"
FACT = {
    "subject": "Atlas",
    "relation_prompt": "{} is in",
    "new_object": "Paris",
    "ground_truth": "London",
    "rephrase_prompts": ["Atlas can be found in"],
    "locality_prompt": "Nearby is in",
    "locality_ground_truth": "Rome",
    "portability_prompt": "Atlas belongs to",
    "portability_ground_truth": "France",
}
PROMPT = "Atlas is in"


def test_judge_failure_and_cancellation_do_not_leak_provider_traceback(protocol_evaluator, monkeypatch):
    state = protocol_evaluator
    state.failures["rewrite"] = RuntimeError(f"Provider echoed key {TEST_API_KEY}")

    def check_running(_job_id):
        if state.calls:
            raise RuntimeError("Evaluation canceled")

    monkeypatch.setattr(runtime, "_ensure_evaluation_job_running", check_running)
    with pytest.raises(RuntimeError, match="Evaluation canceled") as caught:
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], evaluation_modes=["reliability"],
            evaluation_protocols={"reliability": "LLM-judge"},
        )
    error = caught.value
    assert TEST_API_KEY not in "".join(traceback.format_exception(type(error), error, error.__traceback__))


def metric_meta(section, hparams, _model_name="demo", *_):
    mode = getattr(hparams, "evaluation_type", None) or "traditional"
    return {
        "result_key": f"{section}_acc",
        "protocol": {
            "traditional": "decoder_only_teacher_forcing",
            "LLM-judge": "free_generation_judge",
        }[mode],
    }


class StubTokenizer:
    chat_template = "test chat template"
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0
    template_error = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        if self.template_error:
            raise self.template_error
        if self.chat_template is None:
            raise ValueError("Cannot use apply_chat_template because tokenizer.chat_template is not set")
        return f"<chat>{messages[-1]['content']}"


def set_api_key(monkeypatch, key):
    values = dict(vars(config.settings))
    values["deepseek_api_key"] = key
    settings = SimpleNamespace(**values)
    monkeypatch.setattr(config, "settings", settings)
    monkeypatch.setattr(runtime, "settings", settings)


@pytest.fixture
def protocol_evaluator(monkeypatch):
    state = SimpleNamespace(
        active=True,
        calls=[],
        combined_calls=[],
        easyedit_generations=[],
        generations=[],
        finalizers=[],
        stacks=[],
        failures={},
        missing_generated_text=set(),
        render_chat=False,
        locality_outputs={False: [[1, 2]], True: [[1, 2]]},
    )
    hparams = SimpleNamespace(
        model_name="demo", device="cpu", alg_name="MEMIT", evaluation_type="traditional", api_key=None,
    )
    tokenizer = StubTokenizer()
    state.hparams = hparams
    state.tokenizer = tokenizer
    set_api_key(monkeypatch, SecretStr(TEST_API_KEY))
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {"hf_id": "demo", "supports_memit": True})
    monkeypatch.setattr(runtime, "_load_easyedit", lambda _: (None, SimpleNamespace(from_hparams=lambda _: hparams)))
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: Path("unused"))
    monkeypatch.setattr(runtime, "_configure_loaded_tokenizer", lambda *_: None)
    monkeypatch.setattr(runtime, "_override_easyedit_device", lambda *_: None)
    monkeypatch.setattr(
        runtime, "_easyedit_runtime_access",
        lambda _: contextlib.nullcontext({"model": object(), "tokenizer": tokenizer, "device": "cpu"}),
    )
    monkeypatch.setattr(runtime.patch_service, "active_patches_for_model", lambda _: ["patch_test"])
    monkeypatch.setattr(runtime, "ensure_fluency_tokenizer_data", lambda: None)

    def sync_active(*_):
        state.active = True
        state.stacks.append(True)

    def sync_stack(_model, _model_id, patch_ids):
        state.active = bool(patch_ids)
        state.stacks.append(state.active)

    monkeypatch.setattr(runtime.patch_service, "sync_active_patches_to_model", sync_active)
    monkeypatch.setattr(runtime.patch_service, "sync_patch_stack_to_model", sync_stack)

    def observe(kind, params, tok, prompt, target):
        mode = getattr(params, "evaluation_type", None) or "traditional"
        entry = {
            "kind": kind, "active": state.active, "mode": mode, "prompt": prompt, "target": target,
            "api_key": getattr(params, "api_key", None), "uses_original_hparams": params is hparams,
        }
        state.calls.append(entry)
        if mode == "LLM-judge":
            # EasyEdit itself silently falls back to exact match when this is absent.
            assert entry["api_key"] == TEST_API_KEY, "Workbench must supply the real judge credential."
        if kind in state.failures:
            raise state.failures[kind]
        if state.render_chat and mode != "traditional":
            entry["rendered_prompt"] = tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        return mode

    def generated(kind, prompt):
        return f"EasyEdit {'edited' if state.active else 'original'} {kind}: {prompt}"

    def generated_texts(_model, tok, params, prompt, _device):
        kind = {
            PROMPT: "rewrite", FACT["rephrase_prompts"][0]: "rephrase",
            FACT["portability_prompt"]: "portability",
        }[prompt]
        entry = {"kind": kind, "mode": params.evaluation_type, "active": state.active, "prompt": prompt}
        state.easyedit_generations.append(entry)
        if state.render_chat:
            entry["rendered_prompt"] = tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        return [generated(kind, prompt)]

    def rewrite(_model, _name, params, tok, prompt, target_new, device, test_rephrase=False, **_):
        section = "rephrase" if test_rephrase else "rewrite"
        mode = observe(section, params, tok, prompt, target_new)
        result = {"metric_meta": {section: metric_meta(section, params)}}
        result[f"{section}_acc"] = [{"rewrite": 0.6, "rephrase": 0.8}[section]]
        if mode == "LLM-judge" and section not in state.missing_generated_text:
            result[f"{section}_gen_content"] = [generated(section, prompt)]
        return result

    def locality(_model, _name, params, tok, locality_key, prompt, locality_ground_truth, device):
        mode = observe("locality", params, tok, prompt, locality_ground_truth)
        if mode == "traditional":
            return {f"{locality_key}_output": state.locality_outputs[state.active]}
        assert mode == "generate-text"
        return {f"{locality_key}_gen_content": [state.locality_answers[state.active]]}

    def portability(_model, _name, params, tok, portability_key, prompt, ground_truth, device):
        mode = observe("portability", params, tok, prompt, ground_truth)
        result = {f"{portability_key}_acc": [0.4]}
        if mode == "LLM-judge":
            result[f"{portability_key}_gen_content"] = [generated("portability", prompt)]
        return result

    def compute(model, name, params, tok, record, device, **kwargs):
        state.combined_calls.append({"record": deepcopy(record), "active": state.active, "kwargs": kwargs})
        result = rewrite(model, name, params, tok, record["prompt"], record["target_new"], device)
        if record.get("rephrase_prompt"):
            extra = rewrite(model, name, params, tok, record["rephrase_prompt"], record["target_new"], device, test_rephrase=True)
            result["metric_meta"].update(extra.pop("metric_meta"))
            result.update(extra)
        for section, evaluator in (("locality", locality), ("portability", portability)):
            result[section] = {}
            for key, payload in record.get(section, {}).items():
                result[section].update(evaluator(model, name, params, tok, key, payload["prompt"], payload["ground_truth"], device))
        return result

    def finalize(metric, record, params, _model_name):
        mode = getattr(params, "evaluation_type", None) or "traditional"
        state.finalizers.append({"mode": mode, "metric": deepcopy(metric), "record": deepcopy(record)})
        for key in record["locality"]:
            output_key = f"{key}_gen_content" if mode == "generate-text" else f"{key}_output"
            pre = metric["pre"]["locality"][output_key]
            post = metric["post"]["locality"][output_key]
            metric["post"]["locality"][f"{key}_acc"] = [float(before == after) for before, after in zip(pre, post)]
            metric["post"].setdefault("metric_meta", {})[f"locality.{key}"] = {
                "result_key": f"locality.{key}_acc", "protocol": "pre_post_argmax_token_match",
            }

    module = SimpleNamespace(
        compute_edit_quality=compute,
        compute_rewrite_or_rephrase_quality=rewrite,
        compute_locality_quality=locality,
        compute_portability_quality=portability,
    )
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate", module)
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.evaluate", module)
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.metric_meta", SimpleNamespace(build_lm_metric_meta=metric_meta))
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.evaluate_utils", SimpleNamespace(
        generate_texts=generated_texts, test_generation_quality=lambda **_: {"ngram_entropy": 2.5},
    ))
    monkeypatch.setitem(sys.modules, "easyeditor.editors.editor", SimpleNamespace(finalize_locality_metrics=finalize))
    from app.ml.inference import generate

    def generation(_model_id, prompt, *, use_active_patches, **_):
        state.generations.append((prompt, use_active_patches))
        return {"text": f"Workbench {'edited' if use_active_patches else 'original'}: {prompt}"}

    monkeypatch.setattr(generate, "generate_text", generation)
    return state


def test_mixed_protocols_route_each_metric_and_reuse_generated_text(protocol_evaluator, capsys, caplog):
    state = protocol_evaluator
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["reliability", "generalization", "locality"],
        evaluation_protocols={"reliability": "traditional", "generalization": "LLM-judge", "locality": "traditional"},
    )
    assert state.combined_calls == []
    assert {(call["kind"], call["mode"], call["active"]) for call in state.calls} == {
        ("rewrite", "traditional", True),
        ("rephrase", "LLM-judge", True),
        ("locality", "traditional", False), ("locality", "traditional", True),
    }
    assert state.easyedit_generations == [{
        "kind": "rephrase", "mode": "LLM-judge", "active": False, "prompt": FACT["rephrase_prompts"][0],
    }]
    assert state.generations == [(PROMPT, False), (PROMPT, True), (FACT["locality_prompt"], False), (FACT["locality_prompt"], True)]
    assert result["sample_rephrase_generations"] == [{
        "prompt": FACT["rephrase_prompts"][0],
        "clean_model": f"EasyEdit original rephrase: {FACT['rephrase_prompts'][0]}",
        "active_model": f"EasyEdit edited rephrase: {FACT['rephrase_prompts'][0]}",
    }]
    assert result["sample_locality_generations"] == [{
        "prompt": FACT["locality_prompt"],
        "clean_model": f"Workbench original: {FACT['locality_prompt']}",
        "active_model": f"Workbench edited: {FACT['locality_prompt']}",
    }]
    assert result["mean_reliability"] == 0.6
    assert result["mean_generalization"] == 0.8
    assert result["mean_locality"] == 1.0
    assert state.finalizers[0]["mode"] == "traditional"
    assert state.hparams.evaluation_type == "traditional"
    assert state.hparams.api_key is None
    assert all(not call["uses_original_hparams"] for call in state.calls)
    output = capsys.readouterr()
    assert TEST_API_KEY not in json.dumps(result)
    assert TEST_API_KEY not in output.out + output.err + caplog.text


def test_judge_scores_reuse_exact_outputs_without_workbench_generation(protocol_evaluator):
    state = protocol_evaluator
    modes = ["reliability", "generalization", "portability"]
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=modes, evaluation_protocols={mode: "LLM-judge" for mode in modes},
    )
    assert state.generations == []
    assert {call["kind"] for call in state.calls} == {"rewrite", "rephrase", "portability"}
    assert len(state.calls) == 3
    assert all(call["mode"] == "LLM-judge" and call["active"] for call in state.calls)
    assert len(state.easyedit_generations) == 3
    assert all(not call["active"] for call in state.easyedit_generations)
    for mode, score in zip(modes, (0.6, 0.8, 0.4)):
        assert result[f"mean_{mode}"] == score
        assert result["cases"][0][f"{mode}_score"] == score
    for key in ("sample_generations", "sample_rephrase_generations", "sample_portability_generations"):
        assert result[key][0]["clean_model"].startswith("EasyEdit original")
        assert result[key][0]["active_model"].startswith("EasyEdit edited")
    metrics = result["cases"][0]["metrics"]
    assert "rewrite_gen_content" in metrics
    assert "rephrase_gen_content" in metrics
    assert any(key.endswith("_gen_content") for key in metrics["portability"])
    assert metrics["rewrite_acc"] == [0.6]
    assert metrics["rephrase_acc"] == [0.8]


def test_missing_easyedit_text_does_not_trigger_replacement_generation(protocol_evaluator):
    state = protocol_evaluator
    state.missing_generated_text.add("rewrite")
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["reliability"], evaluation_protocols={"reliability": "LLM-judge"},
    )
    assert result["mean_reliability"] == 0.6
    assert result["sample_generations"] == [None]
    assert state.generations == []


@pytest.mark.parametrize("edited_output, expected", [([[1, 2]], 1.0), ([[3, 4]], 0.0)])
def test_traditional_locality_preserves_original_and_edited_output_comparison(protocol_evaluator, edited_output, expected):
    state = protocol_evaluator
    state.locality_outputs[True] = edited_output
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["locality"], evaluation_protocols={"locality": "traditional"},
    )
    assert state.generations == [(FACT["locality_prompt"], False), (FACT["locality_prompt"], True)]
    assert result["mean_locality"] == expected
    assert result["cases"][0]["locality_score"] == expected
    finalizer = state.finalizers[0]
    assert finalizer["mode"] == "traditional"
    assert finalizer["metric"]["pre"]["locality"]["neighborhood_output"] == state.locality_outputs[False]
    assert finalizer["metric"]["post"]["locality"]["neighborhood_output"] == edited_output
    assert state.active is True


@pytest.mark.parametrize("edited_answer, expected", [
    ("Preserved original answer", 1.0), ("Changed answer", 0.0),
])
def test_generated_locality_scores_preservation_and_reuses_exact_text(protocol_evaluator, edited_answer, expected):
    state = protocol_evaluator
    state.locality_answers = {False: "Preserved original answer", True: edited_answer}
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [{**FACT, "locality_ground_truth": ""}], include_generations=True,
        evaluation_modes=["locality"], evaluation_protocols={"locality": "generate-text"},
    )
    assert state.generations == []
    assert state.easyedit_generations == []
    assert result["mean_locality"] == expected
    assert result["cases"][0]["locality_score"] == expected
    assert result["sample_locality_generations"] == [{
        "prompt": FACT["locality_prompt"],
        "clean_model": "Preserved original answer",
        "active_model": edited_answer,
    }]
    assert state.finalizers[0]["mode"] == "generate-text"
    assert [(call["mode"], call["active"], call["api_key"]) for call in state.calls] == [
        ("generate-text", False, None), ("generate-text", True, None),
    ]
    assert state.active is True


@pytest.mark.parametrize("metric", ["reliability", "generalization", "portability"])
def test_generation_only_protocol_is_rejected_before_model_work(protocol_evaluator, monkeypatch, metric):
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: pytest.fail("Invalid protocol must not access a model."))
    with pytest.raises(ValueError, match="generate-text.*only.*locality"):
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], evaluation_modes=[metric],
            evaluation_protocols={metric: "generate-text"}, include_generations=True,
        )
    assert protocol_evaluator.calls == []
    assert protocol_evaluator.easyedit_generations == []
    assert protocol_evaluator.generations == []
    assert protocol_evaluator.stacks == []


@pytest.mark.parametrize("key", [None, SecretStr(" ")])
def test_judge_requires_configured_key_without_exact_match_fallback(protocol_evaluator, monkeypatch, key):
    set_api_key(monkeypatch, key)
    with pytest.raises(ValueError, match="(?i)deepseek.*key|DEEPSEEK_API_KEY"):
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], evaluation_modes=["generalization"],
            evaluation_protocols={"generalization": "LLM-judge"},
        )
    assert protocol_evaluator.calls == []
    assert protocol_evaluator.generations == []
    assert protocol_evaluator.hparams.api_key is None


def test_provider_failure_does_not_expose_judge_credential(protocol_evaluator, capsys, caplog):
    state = protocol_evaluator
    state.failures["rephrase"] = RuntimeError(f"Provider rejected credential {TEST_API_KEY}")
    with pytest.raises(RuntimeError, match="(?i)LLM judg") as error:
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], evaluation_modes=["generalization"],
            evaluation_protocols={"generalization": "LLM-judge"},
        )
    output = capsys.readouterr()
    assert TEST_API_KEY not in str(error.value)
    assert TEST_API_KEY not in output.out + output.err + caplog.text
    assert state.hparams.api_key is None


@pytest.mark.parametrize("modes, protocols, message", [
    (["locality"], {"locality": "LLM-judge"}, "(?i)locality.*LLM|LLM.*locality"),
    (["unknown"], {"reliability": "traditional"}, "(?i)unknown|unsupported|invalid"),
    (["reliability"], {"reliability": "other"}, "(?i)protocol|unsupported|invalid"),
])
def test_invalid_mode_and_unsupported_protocol_fail_before_evaluation(protocol_evaluator, modes, protocols, message):
    with pytest.raises(ValueError, match=message):
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], evaluation_modes=modes, evaluation_protocols=protocols,
        )
    assert protocol_evaluator.calls == []
    assert protocol_evaluator.generations == []


def test_selected_metrics_skip_other_prompts_in_the_record(protocol_evaluator):
    state = protocol_evaluator
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["reliability"], evaluation_protocols={"reliability": "traditional"},
    )
    assert [(call["kind"], call["active"]) for call in state.calls] == [("rewrite", True)]
    assert state.generations == [(PROMPT, False), (PROMPT, True)]
    for mode in ("generalization", "locality", "portability"):
        assert result["cases"][0].get(f"{mode}_score") is None
    for key in ("sample_rephrase_generations", "sample_locality_generations", "sample_portability_generations"):
        assert not any(result.get(key, []))


def test_no_protocol_selectors_preserves_combined_traditional_evaluation(protocol_evaluator):
    fact = {key: value for key, value in FACT.items() if not key.startswith("locality_")}
    result = runtime.evaluate_easyedit_quality("demo", "memit", [fact])
    assert len(protocol_evaluator.combined_calls) == 1
    record = protocol_evaluator.combined_calls[0]["record"]
    assert record["prompt"] == PROMPT
    assert record["rephrase_prompt"] == FACT["rephrase_prompts"][0]
    assert next(iter(record["portability"].values())) == {
        "prompt": FACT["portability_prompt"], "ground_truth": FACT["portability_ground_truth"],
    }
    assert result["mean_reliability"] == 0.6
    assert result["mean_generalization"] == 0.8
    assert result["mean_portability"] == 0.4


def test_no_protocol_selectors_reuses_compatible_internal_metrics(protocol_evaluator):
    fact = {key: value for key, value in FACT.items() if not key.startswith(("locality_", "portability_"))}
    record = {"prompt": PROMPT, "target_new": FACT["new_object"], "rephrase_prompt": FACT["rephrase_prompts"][0]}
    post = {
        "rewrite_acc": [1.0], "rephrase_acc": [0.75],
        "metric_meta": {section: metric_meta(section, protocol_evaluator.hparams) for section in ("rewrite", "rephrase")},
    }
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [fact], reusable_post_metrics=[{"requested_rewrite": record, "post": post}],
    )
    assert protocol_evaluator.combined_calls == []
    assert protocol_evaluator.calls == []
    assert result["mean_reliability"] == 1.0
    assert result["mean_generalization"] == 0.75
    assert result["cases"][0]["score_source"] == "internal_post"
    assert post["rewrite_acc"] == [1.0]


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("mode", ["traditional", "LLM-judge"])
def test_portability_runs_real_evaluator_with_prompt_and_truth(protocol_evaluator, metadata, mode):
    fact = dict(FACT)
    if metadata:
        fact["source_metadata"] = {key: fact.pop(key) for key in ("portability_prompt", "portability_ground_truth")}
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [fact], include_generations=True,
        evaluation_modes=["portability"], evaluation_protocols={"portability": mode},
    )
    calls = protocol_evaluator.calls
    assert calls
    assert all(call["kind"] == "portability" and call["mode"] == mode for call in calls)
    assert all(call["prompt"] == FACT["portability_prompt"] and call["target"] == FACT["portability_ground_truth"] for call in calls)
    assert result["mean_portability"] == 0.4
    assert result["cases"][0]["portability_score"] == 0.4
    assert any(key.endswith("_acc") for key in result["cases"][0]["metrics"]["portability"])
    if mode == "LLM-judge":
        assert protocol_evaluator.generations == []
        assert result["sample_portability_generations"][0]["clean_model"].startswith("EasyEdit original")
        assert result["sample_portability_generations"][0]["active_model"].startswith("EasyEdit edited")


@pytest.mark.parametrize("missing", ["portability_prompt", "portability_ground_truth"])
@pytest.mark.parametrize("mode", ["traditional", "LLM-judge"])
def test_missing_optional_portability_data_does_not_become_zero(protocol_evaluator, missing, mode):
    fact = {key: value for key, value in FACT.items() if key != missing}
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [fact], evaluation_modes=["portability"],
        evaluation_protocols={"portability": mode},
    )
    assert protocol_evaluator.calls == []
    assert result["mean_portability"] is None
    assert result["cases"][0]["portability_score"] is None


def test_judge_adapts_base_tokenizer_without_mutating_it(protocol_evaluator):
    state = protocol_evaluator
    state.tokenizer.chat_template = None
    state.render_chat = True
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["reliability"], evaluation_protocols={"reliability": "LLM-judge"},
    )
    assert all(call["rendered_prompt"] == PROMPT for call in state.calls)
    assert all(call["rendered_prompt"] == PROMPT for call in state.easyedit_generations)
    assert state.tokenizer.chat_template is None
    assert result["sample_generations"][0]["active_model"].startswith("EasyEdit edited")
    assert state.generations == []


def test_base_tokenizer_proxy_runs_unmodified_easyedit_judge_path(protocol_evaluator, monkeypatch):
    torch = pytest.importorskip("torch")
    evaluate_source = config.settings.easyedit_root / "easyeditor" / "evaluate" / "evaluate.py"
    generator_source = evaluate_source.with_name("evaluate_utils.py")
    if not evaluate_source.exists() or not generator_source.exists():
        pytest.skip("Requires the EasyEdit checkout (read-only)")

    def load_function(source, name, namespace):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 -- execute the installed evaluator without its unrelated imports
        return namespace[name]

    def attach_meta(result, section, meta):
        result.setdefault("metric_meta", {})[section] = meta

    state = protocol_evaluator
    state.judge_calls = []

    def judge(question, ground_truth, prediction, api_key):
        state.judge_calls.append((question, ground_truth, prediction, api_key))
        return 1.0

    namespace = {
        "typing": typing, "HyperParams": object, "AutoTokenizer": object,
        "T5ForConditionalGeneration": type("StubSeq2Seq", (), {}),
        "normalize_device": lambda device: device,
        "build_lm_metric_meta": metric_meta, "attach_metric_meta": attach_meta,
        "llm_judge": judge,
    }
    real_generator = load_function(generator_source, "generate_texts", namespace)
    load_function(generator_source, "test_prediction_acc_LLM_judge", namespace)
    real_rewrite = load_function(evaluate_source, "compute_rewrite_or_rephrase_quality", namespace)

    class TinyBatch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class TinyTokenizer(StubTokenizer):
        chat_template = None

        def __call__(self, text, **_):
            state.tokenized_prompts.append(text)
            return TinyBatch(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, tokens):
            assert tokens.tolist() == [12]
            return "Paris<eos>"

    class TinyModel:
        def generate(self, **kwargs):
            state.model_generations.append(kwargs)
            assert kwargs["tokenizer"].eos_token == "<eos>"
            return torch.tensor([[10, 11, 12]])

    tokenizer = TinyTokenizer()
    model = TinyModel()
    state.tokenized_prompts = []
    state.model_generations = []
    monkeypatch.setattr(
        runtime, "_easyedit_runtime_access",
        lambda _: contextlib.nullcontext({"model": model, "tokenizer": tokenizer, "device": "cpu"}),
    )
    monkeypatch.setattr(sys.modules["easyeditor.evaluate"], "compute_rewrite_or_rephrase_quality", real_rewrite)
    monkeypatch.setattr(sys.modules["easyeditor.evaluate.evaluate_utils"], "generate_texts", real_generator)
    result = runtime.evaluate_easyedit_quality(
        "demo", "memit", [FACT], include_generations=True,
        evaluation_modes=["reliability"], evaluation_protocols={"reliability": "LLM-judge"},
    )
    assert state.tokenized_prompts == [PROMPT, PROMPT]
    assert len(state.model_generations) == 2
    assert result["sample_generations"] == [{"prompt": PROMPT, "clean_model": "Paris", "active_model": "Paris"}]
    assert result["cases"][0]["metrics"]["rewrite_gen_content"] == ["Paris"]
    assert result["mean_reliability"] == 1.0
    assert state.judge_calls == [(PROMPT, FACT["new_object"], "Paris", TEST_API_KEY)]
    assert state.generations == []
    assert tokenizer.chat_template is None


@pytest.mark.parametrize("include_generations", [False, True])
def test_incompatible_chat_template_reports_meaningful_error(protocol_evaluator, include_generations):
    state = protocol_evaluator
    state.render_chat = True
    state.tokenizer.template_error = ValueError("Unsupported role: system")
    with pytest.raises((runtime.EasyEditRuntimeError, ValueError), match="(?i)chat.?template"):
        runtime.evaluate_easyedit_quality(
            "demo", "memit", [FACT], include_generations=include_generations, evaluation_modes=["reliability"],
            evaluation_protocols={"reliability": "LLM-judge"},
        )
    assert state.generations == []
    assert state.hparams.evaluation_type == "traditional"
