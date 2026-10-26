"""Editing methods exercise the shared batch, statistics, and patch lifecycle."""
from contextlib import nullcontext
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from app.api import routes_editing
from app.db.schemas import BatchWeightEditRequest, DpoEditRequest, ReliabilityEvaluationRequest, TrainingEditRequest
from app.ml.editing import easyedit_runtime as runtime, memit_wrapper
from app.ml.editing.edit_cancellation import _run_local_edit, request_cancel
from app.ml.editing.methods import BATCH_WEIGHT_METHODS, COVARIANCE_MODULES, METHODS
from app.ml.editing.training_hparams import validate_training_hparams
from app.services import model_registry as registry
from app.services.patch_service import patch_service
from app.storage.local_store import local_store

FACT = {"subject": "Paris", "relation_prompt": "{} is in", "new_object": "Italy", "ground_truth": "France"}
WEIGHT_METHODS = ["emmet", "memit_fe", "pmet", "eamet", "alphaedit", "sphere"]


@pytest.mark.parametrize("method", [*BATCH_WEIGHT_METHODS, "ft", "lora", "dpo"])
def test_routes_keep_batch_size_and_method(monkeypatch, method):
    captured = {}
    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing, "_require_mom2_statistics", lambda model, method, *args: captured.update(stats=method))
    monkeypatch.setattr(routes_editing.experiment_service, "create", lambda kind, *args: {"id": kind})
    monkeypatch.setattr(routes_editing.experiment_service, "create_job", lambda exp: {
        "id": "job_test", "experiment_id": exp, "status": "running", "logs": [],
    })
    monkeypatch.setattr(routes_editing, "_start_edit_thread", lambda label, job, runner, payload:
                        captured.update(method=runner.keywords["method"], **payload))
    if method in BATCH_WEIGHT_METHODS:
        schema = BatchWeightEditRequest
    else:
        schema = DpoEditRequest if method == "dpo" else TrainingEditRequest
    request = schema(model_id="llama-7b", edits=[{**FACT, "target_neg": "France"}], batch_size=2)
    result = getattr(routes_editing, f"{method}_edit")(request)
    assert result.experiment_id == f"{method}_edit"
    assert captured["method"] == method and captured["batch_size"] == 2
    if method in BATCH_WEIGHT_METHODS:
        assert captured["stats"] == method
    assert ReliabilityEvaluationRequest(method=method, edits=[FACT]).method == method


@pytest.mark.parametrize("negative", [None, "", "  ", "Italy", " Italy "])
def test_dpo_requires_distinct_rejected_answer(negative):
    with pytest.raises(ValidationError):
        DpoEditRequest(edits=[{**FACT, "target_neg": negative}])


@pytest.mark.parametrize("hparams", [{"alpha": -0.1}, {"alpha": 1.1}, {"beta": 0}, {"beta": True}])
def test_dpo_validates_loss_settings(hparams):
    with pytest.raises(ValueError):
        validate_training_hparams("dpo", hparams)


def test_installed_profiles_load_and_select_valid_layers():
    for model in registry.SUPPORTED_MODELS:
        for method in (*BATCH_WEIGHT_METHODS, "dpo"):
            spec = runtime.METHODS[method]
            assert model[spec["support_key"]] == bool(model["easyedit"].get(spec["hparams_key"]))
            if not model[spec["support_key"]]:
                continue
            _, cls = runtime._load_easyedit(spec["hparams_class"])
            hparams = cls.from_hparams(str(runtime._resolve_hparams_path(model, spec["hparams_key"])))
            assert all(0 <= layer < model["n_layers"] for layer in hparams.layers)
            assert hparams.alg_name == {"memit_fe": "MEMIT-FE", "alphaedit": "AlphaEdit"}.get(method, method.upper())


@pytest.mark.parametrize("base", registry.SUPPORTED_MODELS, ids=lambda model: model["id"])
def test_imported_models_inherit_profiles(monkeypatch, base):
    monkeypatch.setattr(registry, "_read_local_models", lambda: [
        {"id": "local-test", "source": "local", "base_model_id": base["id"]},
    ])
    monkeypatch.setattr(registry, "_migrate_managed_local_models", lambda _: False)
    model = registry.ModelRegistry().list_local_models()[0]
    for spec in METHODS.values():
        assert model[spec["support_key"]] == base[spec["support_key"]]
        assert model["easyedit"].get(spec["hparams_key"]) == base["easyedit"].get(spec["hparams_key"])


@pytest.mark.parametrize("method", WEIGHT_METHODS)
def test_batch_worker_keeps_all_facts(monkeypatch, method):
    import app.services.model_runtime_service as runtime_module
    captured = {}
    def execute(action, payload, **kwargs):
        captured.update(action=action, **payload)
        return {"worker": True}
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=True, execute=execute))
    memit_wrapper.run_memit_edit(method, "demo", [FACT] * 5, [0], [], None, None, False, True, batch_size=2)
    assert captured["action"] == f"{method}_edit"
    assert captured["batch_size"] == 2 and len(captured["edits"]) == 5


def test_pmet_statistics_cover_every_rewrite_module(monkeypatch, tmp_path):
    profile = tmp_path / "pmet.yaml"
    profile.write_text(yaml.safe_dump({
        "layers": [0], "rewrite_module_tmp": "layers.{}.mlp",
        "rewrite_module_tmps": ["layers.{}.mlp", "layers.{}.attn"],
        "mom2_dataset": "wikipedia", "mom2_n_samples": 2, "mom2_dtype": "float32",
    }))
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "id": "tiny", "hf_id": "tiny", "supports_pmet": True, "n_layers": 2,
    })
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: profile)
    monkeypatch.setattr(runtime, "_mom2_stats_dir", lambda: tmp_path / "stats")
    monkeypatch.setattr(runtime, "_mom2_precision_device", lambda _: "cpu")
    monkeypatch.setattr(local_store, "root", tmp_path / "data")
    status = runtime.get_mom2_statistics_status("tiny", "pmet")
    assert len(status["statistics"]) == 2 and status["missing_layers"] == [0]
    from pathlib import Path
    for entry in status["statistics"]:
        path = Path(entry["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{"sample_size": 2, "mom2.count": 2, "mom2.mom2": np.eye(2, dtype="float32")})
    assert runtime.get_mom2_statistics_status("tiny", "pmet")["ready"]
    assert len(runtime.list_mom2_statistics("tiny", "pmet")["statistics"]) == 2
    result = runtime.delete_mom2_statistics("tiny", "pmet", [0])
    assert result["deleted_layers"] == [0] and not result["statistics"]


@pytest.fixture
def tiny_weight_edit(monkeypatch, tmp_path):
    from test_mps_integration import tiny_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(tiny_model.__wrapped__(tmp_path))
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
    )).eval()
    model.config._name_or_path = "llama-tiny"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_store, "root", tmp_path / "data")
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "id": "tiny", "hf_id": "llama-tiny", "n_layers": 3,
        **{f"supports_{method}": True for method in WEIGHT_METHODS},
    })
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", lambda _: nullcontext({
        "model": model, "tokenizer": tokenizer, "device": "cpu",
    }))
    monkeypatch.setattr(runtime, "_collect_runtime_memory", lambda: None)
    paths = {}
    for method in WEIGHT_METHODS:
        _, cls = runtime._load_easyedit(runtime.METHODS[method]["hparams_class"])
        profile_name = "qwen2.5-7b.yaml" if method == "eamet" else "llama3-8b.yaml" if method in {"alphaedit", "sphere"} else "llama-7b.yaml"
        source = runtime._resolve_easyedit_root() / "hparams" / {"memit_fe": "MEMIT-FE", "alphaedit": "AlphaEdit"}.get(method, method.upper()) / profile_name
        config = yaml.safe_load(source.read_text())
        # Keep a downstream attention layer so subject-token interventions can
        # affect the prediction at the end of the prompt.
        config.update(layers=[0, 1], v_loss_layer=2, v_num_grad_steps=2,
                      v_lr=0.1, mom2_n_samples=2, model_parallel=False)
        if method == "eamet":
            config.update(mom2_update_weight=[100, 100])
        path = tmp_path / f"{method}.yaml"
        path.write_text(yaml.safe_dump(config))
        paths[f"{method}_hparams"] = path
        module = importlib.import_module(COVARIANCE_MODULES[method])
        # Deterministic small covariance/context fixtures; real EasyEdit forwards,
        # gradients, matrix solves, batch dispatch, and weight writes still execute.
        monkeypatch.setattr(module, "get_context_templates", lambda *args, **kwargs: [["{}"]])
        monkeypatch.setattr(module, "get_cov", lambda *args, **kwargs: torch.eye(32) * 0.001)
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda _, key: paths[key])
    return model


def run_weight_edit(method, **kwargs):
    return runtime.run_easyedit(method, "tiny", [FACT] * 3, [0, 1], batch_size=2, **kwargs)


@pytest.mark.parametrize("method", ["alphaedit", "sphere"])
def test_projection_history_is_shared_only_within_job(tiny_weight_edit, monkeypatch, method):
    from pathlib import Path

    module = importlib.import_module(COVARIANCE_MODULES[method])
    old_projection, old_history, old_covariance = object(), object(), {}
    monkeypatch.setattr(module, "P", old_projection, raising=False)
    monkeypatch.setattr(module, "cache_c", old_history, raising=False)
    monkeypatch.setattr(module, "P_loaded", True)
    monkeypatch.setattr(module, "cache_c_new", True)
    monkeypatch.setattr(module, "COV_CACHE", old_covariance)
    loader = runtime._load_easyedit
    seen = []

    def load(name):
        cls, hparams_cls = loader(name)

        def create(hparams):
            editor = cls.from_hparams(hparams)
            apply = editor.apply_algo

            def record(*args, **kwargs):
                history = float(module.cache_c.norm()) if module.cache_c_new else None
                seen.append((module.P_loaded, history, Path(hparams.P_loc)))
                return apply(*args, **kwargs)

            editor.apply_algo = record
            return editor

        return SimpleNamespace(from_hparams=create), hparams_cls

    monkeypatch.setattr(runtime, "_load_easyedit", load)
    for target in ("Italy", "France"):
        result = runtime.run_easyedit(method, "tiny", [{**FACT, "new_object": target}] * 3,
                                      [0, 1], batch_size=2, sequential_edit=True)
        assert result["patch"]["editing_history_scope"] == "job"
        assert module.P is old_projection and module.cache_c is old_history
        assert module.COV_CACHE is old_covariance
        assert module.P_loaded and module.cache_c_new
    assert [entry[0] for entry in seen] == [False, True, False, True]
    assert seen[0][1] is None and seen[2][1] is None
    assert seen[1][1] > 0 and seen[3][1] > 0
    assert seen[0][2] == seen[1][2] and seen[0][2] != seen[2][2]
    assert all(not entry[2].exists() for entry in seen)
    assert not Path("null_space_project.pt").exists()


def test_eamet_layer_selection_keeps_profile_weights_aligned():
    from app.ml.editing.scoped_weight_edit import configure_scoped_layers

    profile = SimpleNamespace(layers=[4, 5, 6], mom2_update_weight=[10, 20, 30])
    configure_scoped_layers("eamet", profile, [6, 4])
    assert profile.mom2_update_weight == [30, 10]
    with pytest.raises(ValueError, match="EasyEdit profile"):
        configure_scoped_layers("eamet", SimpleNamespace(layers=[4], mom2_update_weight=[10]), [0])


@pytest.mark.parametrize("method", WEIGHT_METHODS)
def test_real_batch_algorithms_save_replayable_patches(tiny_weight_edit, method):
    model = tiny_weight_edit
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = run_weight_edit(method)
    assert len(result["metrics"]) == 3
    assert result["patch"]["batch_size"] == 2 and result["patch"]["method"] == method
    from safetensors.torch import load_file
    deltas = load_file(patch_service.resolve_delta_path(result["patch"]["delta_paths"][0]))
    assert set(deltas) == {f"model.layers.{i}.mlp.down_proj.weight" for i in range(2)}
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, before[name] + deltas.get(name, 0))
    patch_service.deactivate_runtime_patches_for_model("tiny")
    patch_service.sync_active_patches_to_model(model, "tiny")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    patch_service.activate_patch(result["patch_id"])
    patch_service.sync_active_patches_to_model(model, "tiny")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name] + deltas.get(name, 0))
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("method", WEIGHT_METHODS)
@pytest.mark.parametrize("failure", ["cancel", "error"])
def test_batch_failures_restore_prior_patch_stack(tiny_weight_edit, monkeypatch, method, failure):
    model = tiny_weight_edit
    previous = run_weight_edit(method)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    loader = runtime._load_easyedit
    batches = []
    module = importlib.import_module(COVARIANCE_MODULES[method])
    absent = object()
    state = {name: getattr(module, name, absent)
             for name in ("P", "P_loaded", "cache_c", "cache_c_new")}
    projection_paths = []
    def load(name):
        cls, hparams_cls = loader(name)
        def create(hparams):
            editor = cls.from_hparams(hparams)
            apply = editor.apply_algo
            def fail(*args, **kwargs):
                if method in {"alphaedit", "sphere"}:
                    from pathlib import Path
                    projection_paths.append(Path(args[3].P_loc))
                result = apply(*args, **kwargs)
                batches.append(len(args[2]))
                if len(batches) == 2:
                    if failure == "cancel":
                        request_cancel("job_new_batch")
                    else:
                        raise RuntimeError("Batch failed")
                return result
            editor.apply_algo = fail
            return editor
        return SimpleNamespace(from_hparams=create), hparams_cls
    monkeypatch.setattr(runtime, "_load_easyedit", load)
    if failure == "cancel":
        assert _run_local_edit("job_new_batch", lambda: run_weight_edit(method), {}) == {"_edit_canceled": True}
    else:
        with pytest.raises(RuntimeError, match="Batch failed"):
            _run_local_edit("job_new_batch", lambda: run_weight_edit(method), {})
    assert batches == [2, 1]
    if method in {"alphaedit", "sphere"}:
        assert all(getattr(module, name, absent) is value for name, value in state.items())
        assert all(not path.exists() for path in projection_paths)
    assert patch_service.active_patches_for_model("tiny") == [previous["patch_id"]]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
