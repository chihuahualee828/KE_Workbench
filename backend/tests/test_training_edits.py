from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import ValidationError

from app.api import routes_editing
from app.db.schemas import TrainingEditRequest
from app.ml.editing import easyedit_runtime as runtime
from app.ml.editing import training_wrapper as training
from app.ml.editing.edit_cancellation import _run_local_edit, request_cancel
from app.services.patch_service import patch_service
from app.storage.local_store import local_store

FACT = {"subject": "Paris", "relation_prompt": "{} is in", "new_object": "Italy", "ground_truth": "France"}


def test_real_easyedit_post_edit_fluency(tiny_training, monkeypatch):
    """Exercise the installed fluency generator, not just the mocked scorer."""
    import nltk
    word_tokenize = nltk.word_tokenize
    # Exercise generation/entropy offline without downloading test-only assets.
    monkeypatch.setattr(nltk, "word_tokenize", lambda text: word_tokenize(text, preserve_line=True))
    from contextlib import nullcontext
    from transformers import GPT2LMHeadModel
    config = deepcopy(tiny_training.model.config)
    config.n_positions = 128  # EasyEdit's fluency test generates up to length 100.
    model = GPT2LMHeadModel(config).eval()
    model.name_or_path = "tiny-gpt2"
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", lambda _: nullcontext({
        "model": model, "tokenizer": tiny_training.tokenizer, "device": "cpu",
    }))
    monkeypatch.setattr(runtime.patch_service, "sync_active_patches_to_model", lambda *_: None)
    result = runtime.evaluate_easyedit_quality("tiny", "ft", [FACT], fluency_modes=["reliability"])
    assert result["cases"][0]["reliability_score"] is not None
    fluency = result["cases"][0]["fluency"]["reliability"]
    assert isinstance(fluency["ngram_entropy"], float)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "2"])
def test_training_batch_size_validation(batch_size):
    with pytest.raises(ValidationError):
        TrainingEditRequest(edits=[FACT], batch_size=batch_size)


@pytest.mark.parametrize("method", ["ft", "lora"])
def test_training_route_starts_job_with_batch_size(monkeypatch, method):
    captured = {}
    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing.experiment_service, "create", lambda *args: {"id": "exp_test"})
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda _: {
        "id": "job_test", "experiment_id": "exp_test", "status": "running", "logs": [],
    })
    monkeypatch.setattr(routes_editing, "_start_edit_thread", lambda label, job, runner, payload: captured.update(
        method=runner.keywords["method"], **payload))
    response = getattr(routes_editing, f"{method}_edit")(TrainingEditRequest(edits=[FACT], batch_size=20))
    assert response.status == "running"
    assert captured["batch_size"] == 20 and captured["method"] == method


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    from test_mps_integration import tiny_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = tiny_model.__wrapped__(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()
    monkeypatch.setattr(local_store, "root", tmp_path / "data")
    monkeypatch.setattr(training, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "id": "tiny", "hf_id": "gpt2-tiny", "n_layers": 2, "supports_ft": True, "supports_lora": True,
    })
    try:
        editor_cls, _ = runtime._load_easyedit("FTHyperParams")
    except runtime.EasyEditRuntimeError as exc:
        pytest.skip(f"Requires installed EasyEdit: {exc}")
    from app.config import settings
    paths = {}
    for method, directory in [("ft", "FT"), ("lora", "LoRA")]:
        config = yaml.safe_load((settings.easyedit_root / f"hparams/{directory}/gpt2-xl.yaml").read_text())
        config.update(num_steps=1, layers=[0], lr=0.01, model_parallel=False)
        paths[f"{method}_hparams"] = tmp_path / f"{method}.yaml"
        paths[f"{method}_hparams"].write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda _, key: paths[key])
    monkeypatch.setattr(runtime, "evaluate_post_edit_reliability", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "_collect_runtime_memory", lambda: None)
    monkeypatch.chdir(tmp_path)

    @contextmanager
    def access(_):
        yield {"model": model, "tokenizer": tokenizer, "device": "cpu"}
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", access)
    return SimpleNamespace(model=model, tokenizer=tokenizer, editor_cls=editor_cls)


@pytest.mark.parametrize("method", ["ft", "lora"])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_real_easyedit_batches_create_replayable_patches(tiny_training, method, batch_size):
    model = tiny_training.model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    hooks = dict(model.get_input_embeddings()._forward_hooks)
    result = training.run_training_edit(method, "tiny", [FACT] * 3, [0], batch_size=batch_size)
    assert result["edited_facts"] == 3
    assert len(result["metrics"]) == 3
    assert result["patch"]["batch_size"] == batch_size
    assert result["patch"]["method"] == method
    assert set(model.state_dict()) == set(before)
    assert not hasattr(model, "peft_config")
    assert model.config.use_cache and not model.is_gradient_checkpointing
    assert not model.training and [p.requires_grad for p in model.parameters()] == flags
    assert dict(model.get_input_embeddings()._forward_hooks) == hooks
    assert not model._forward_pre_hooks
    from safetensors.torch import load_file
    deltas = load_file(patch_service.resolve_delta_path(result["patch"]["delta_paths"][0]))
    assert deltas
    if method == "ft":
        assert any(name.endswith("bias") for name in deltas)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name] + deltas.get(name, 0))
    edited = {name: value.clone() for name, value in model.state_dict().items()}
    patch_service.deactivate_runtime_patches_for_model("tiny")
    patch_service.sync_active_patches_to_model(model, "tiny")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    patch_service.activate_patch(result["patch_id"])
    patch_service.sync_active_patches_to_model(model, "tiny")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, edited[name])


@pytest.mark.parametrize("method", ["ft", "lora"])
@pytest.mark.parametrize("failure", ["cancel", "error"])
def test_training_failure_restores_weights_and_removes_adapters(tiny_training, monkeypatch, method, failure):
    model = tiny_training.model
    old = training.run_training_edit("ft", "tiny", [FACT], [0])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    real_loader = runtime._load_easyedit
    calls = []

    def loader(name):
        cls, hparams = real_loader(name)
        def create(params):
            editor = cls.from_hparams(params)
            apply = editor.apply_algo
            def fail_after_training(*args, **kwargs):
                result = apply(*args, **kwargs)
                calls.append(len(args[2]))
                if len(calls) == 2:
                    if failure == "cancel":
                        request_cancel("job_training")
                    else:
                        raise RuntimeError("Training failed")
                return result
            editor.apply_algo = fail_after_training
            return editor
        return SimpleNamespace(from_hparams=create), hparams
    monkeypatch.setattr(runtime, "_load_easyedit", loader)
    def run():
        return training.run_training_edit(method, "tiny", [FACT] * 3, [0], batch_size=2)
    if failure == "cancel":
        assert _run_local_edit("job_training", run, {}) == {"_edit_canceled": True}
    else:
        with pytest.raises(RuntimeError, match="Training failed"):
            _run_local_edit("job_training", run, {})
    assert calls == [2, 1]
    assert patch_service.active_patches_for_model("tiny") == [old["patch_id"]]
    assert not hasattr(model, "peft_config")
    assert not model._forward_pre_hooks and not model.get_input_embeddings()._forward_hooks
    assert not model.is_gradient_checkpointing and model.config.use_cache
    assert [p.requires_grad for p in model.parameters()] == flags
    assert set(before) == set(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["ft", "lora"])
def test_training_worker_payload(monkeypatch, method):
    recorded = {}
    def execute(action, payload, **kwargs):
        recorded.update(action=action, **payload)
        return {"worker": True}
    monkeypatch.setattr(training, "model_runtime_service", SimpleNamespace(uses_hard_worker=True, execute=execute))
    assert training.run_training_edit(method, "demo", [FACT] * 5, [0], batch_size=2) == {"worker": True}
    assert recorded["action"] == f"{method}_edit"
    assert recorded["batch_size"] == 2 and len(recorded["edits"]) == 5


@pytest.mark.parametrize("method", ["ft", "lora"])
@pytest.mark.parametrize("continuing", [False, True])
def test_training_patch_lineage(tiny_training, method, continuing):
    model = tiny_training.model
    base = {name: value.clone() for name, value in model.state_dict().items()}
    old = training.run_training_edit("ft", "tiny", [FACT], [0])
    new = training.run_training_edit(method, "tiny", [FACT] * 3, [0], batch_size=2, sequential_edit=continuing)
    assert new["patch"]["parent_patch_ids"] == ([old["patch_id"]] if continuing else [])
    active = patch_service.active_patches_for_model("tiny")
    assert active == ([old["patch_id"]] if continuing else []) + [new["patch_id"]]
    from safetensors.torch import load_file
    expected = base
    for patch_id in active:
        patch = patch_service.get_patch(patch_id)
        for name, delta in load_file(patch_service.resolve_delta_path(patch["delta_paths"][0])).items():
            expected[name] = expected[name] + delta
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_training_registry_profiles_and_inherited_local_models(monkeypatch):
    from app.services import model_registry as registry
    for model in registry.SUPPORTED_MODELS:
        for method in ("ft", "lora"):
            assert model[f"supports_{method}"] == bool(model["easyedit"].get(f"{method}_hparams"))
    models = [{"id": "local-test", "source": "local", "base_model_id": "gpt2-xl"}]
    monkeypatch.setattr(registry, "_read_local_models", lambda: models)
    monkeypatch.setattr(registry, "_migrate_managed_local_models", lambda _: False)
    local = registry.ModelRegistry().list_local_models()[0]
    assert local["supports_ft"] and local["supports_lora"]
    assert local["easyedit"]["ft_hparams"] == "hparams/FT/gpt2-xl.yaml"
    assert local["easyedit"]["lora_hparams"] == "hparams/LoRA/gpt2-xl.yaml"


def test_lora_default_layers_preserve_empty_yaml(monkeypatch, tmp_path):
    config = tmp_path / "lora.yaml"
    config.write_text("layers: []\n")
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {"supports_lora": True, "n_layers": 3})
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *args: config)
    assert runtime.get_easyedit_default_layers("tiny", "lora")["layers"] == []


@pytest.mark.parametrize("method", ["ft", "lora"])
def test_training_reliability_uses_loaded_cpu_model(tiny_training, method):
    result = runtime.evaluate_easyedit_quality("tiny", method, [FACT])
    assert result["method"] == method
    assert result["metric_name"] == "rewrite_acc"
    assert len(result["cases"]) == 1
    assert 0 <= result["mean_reliability"] <= 1


@pytest.mark.parametrize("values", [
    {"lr": 0}, {"lr": float("inf")}, {"lr": True}, {"num_steps": 1.5},
    {"rank": 0}, {"lora_dropout": 1}, {"weight_decay": -1},
    {"target_modules": []}, {"target_modules": [" "]}, {"device": "cuda:9"},
    {"norm_constraint": True},
])
def test_training_hparams_reject_invalid_values(values):
    from app.ml.editing.training_hparams import TrainingHyperparameters
    with pytest.raises(ValidationError):
        TrainingHyperparameters.model_validate(values)


@pytest.mark.parametrize("method,values", [("ft", {"rank": 8}), ("lora", {"objective_optimization": "target_new"})])
def test_training_hparams_reject_wrong_method(method, values):
    from app.ml.editing.training_hparams import validate_training_hparams
    with pytest.raises(ValueError, match="Unsupported"):
        validate_training_hparams(method, values)


@pytest.mark.parametrize("method", ["ft", "lora"])
def test_real_training_hparam_overrides(tiny_training, method):
    from app.ml.editing.training_hparams import get_training_defaults
    defaults = get_training_defaults("tiny", method)
    assert defaults["hparams"]["lr"] == 0.01
    overrides = {"lr": 0.002, "num_steps": 2, "weight_decay": 0.01}
    if method == "ft":
        overrides.update(objective_optimization="target_new", norm_constraint=False)
    else:
        overrides.update(target_modules=["c_fc"], rank=2, lora_alpha=4, lora_dropout=0)
    result = training.run_training_edit(method, "tiny", [FACT] * 2, [] if method == "lora" else [0],
                                       batch_size=2, hparams=overrides)
    assert result["patch"]["training_hparams"] == overrides
    if method == "lora":
        assert result["patch"]["configured_layers"] == []
        from safetensors.torch import load_file
        deltas = load_file(patch_service.resolve_delta_path(result["patch"]["delta_paths"][0]))
        assert set(deltas) == {"transformer.h.0.mlp.c_fc.weight", "transformer.h.1.mlp.c_fc.weight"}


def test_lora_target_typo_fails_before_any_mutation(tiny_training):
    before = {name: parameter.clone() for name, parameter in tiny_training.model.state_dict().items()}
    with pytest.raises(ValueError, match="does not match"):
        training.run_training_edit("lora", "tiny", [FACT], [], hparams={"target_modules": ["c_attn", "typo"]})
    assert not hasattr(tiny_training.model, "peft_config")
    for name, parameter in tiny_training.model.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_training_worker_carries_hparam_overrides(monkeypatch):
    captured = {}
    def execute(action, payload, **kwargs):
        captured.update(payload)
        return {}
    monkeypatch.setattr(training, "model_runtime_service", SimpleNamespace(uses_hard_worker=True, execute=execute))
    training.run_training_edit("lora", "demo", [FACT], [], hparams={"lr": 0.001, "target_modules": ["q_proj"]})
    assert captured["hparams"] == {"lr": 0.001, "target_modules": ["q_proj"]}
    assert captured["layers"] == []


def test_training_defaults_warn_about_invalid_yaml_and_allow_override(tiny_training):
    path = runtime._resolve_hparams_path({}, "ft_hparams")
    config = yaml.safe_load(path.read_text())
    config["objective_optimization"] = "promptz_last"
    path.write_text(yaml.safe_dump(config))
    response = routes_editing.training_defaults(routes_editing.EditDefaultLayersRequest(model_id="tiny", method="ft"))
    assert response["hparams"]["objective_optimization"] == "promptz_last"
    assert "objective_optimization" in response["warning"]
    with pytest.raises(ValueError):
        training.run_training_edit("ft", "tiny", [FACT], [0])
    result = training.run_training_edit("ft", "tiny", [FACT], [0], hparams={"objective_optimization": "target_new"})
    assert result["patch"]["training_hparams"]["objective_optimization"] == "target_new"


@pytest.mark.parametrize("method", ["ft", "lora"])
def test_training_route_preserves_hparams_in_job_and_worker(monkeypatch, method):
    captured = {}
    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    def create(kind, model, payload):
        captured["experiment"] = payload
        return {"id": "exp_test"}
    monkeypatch.setattr(routes_editing.experiment_service, "create", create)
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda _: {
        "id": "job_test", "experiment_id": "exp_test", "status": "running", "logs": [],
    })
    monkeypatch.setattr(routes_editing, "_start_edit_thread", lambda label, job, runner, payload: captured.update(worker=payload))
    getattr(routes_editing, f"{method}_edit")(TrainingEditRequest(edits=[FACT], hparams={"lr": 0.002}))
    assert captured["experiment"]["hparams"]["lr"] == captured["worker"]["hparams"]["lr"] == 0.002
