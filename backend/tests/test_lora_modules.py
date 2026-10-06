from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

from app.api import routes_editing
from app.ml.editing.lora_modules import get_lora_modules, inspect_lora_modules
from app.ml.editing.training_wrapper import _training_targets
from app.services import model_runtime_service as runtime_module
from app.services.model_runtime_service import ModelNotLoadedError
from app.services.model_worker import _dispatch


@pytest.fixture
def model():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(vocab_size=12, n_layer=2, n_head=2, n_embd=16, n_positions=16)).eval()


def test_discovery_lists_real_paths_and_shapes_without_inference_or_mutation(model, monkeypatch):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: pytest.fail("Discovery must not run inference"))
    result = inspect_lora_modules(model, "tiny")
    groups = {group["name"]: group for group in result["groups"]}
    assert result["model_id"] == "tiny"
    assert result["module_count"] == 8
    assert set(groups) == {"c_attn", "c_proj", "c_fc", "lm_head"}
    assert len(groups["c_proj"]["modules"]) == 4  # Attention and MLP share this suffix.
    assert groups["c_attn"]["targets"] == ["c_attn"]
    assert groups["c_attn"]["modules"][0] == {
        "path": "transformer.h.0.attn.c_attn", "type": "Conv1D", "weight_shape": [16, 48],
        "available": True, "reason": None,
    }
    assert groups["lm_head"]["targets"] == []
    assert "Shares weights" in groups["lm_head"]["modules"][0]["reason"]
    assert not hasattr(model, "peft_config")
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_every_suggested_target_uses_editor_compatibility_checks(model):
    for group in inspect_lora_modules(model, "tiny")["groups"]:
        if not group["targets"]:
            continue
        _, modules = _training_targets(model, SimpleNamespace(layers=[], target_modules=group["targets"]), "lora")
        assert set(modules) == {item["path"] for item in group["modules"] if item["available"]}
    with pytest.raises(ValueError, match="Shares weights"):
        _training_targets(model, SimpleNamespace(layers=[], target_modules=["lm_head"]), "lora")


def test_mixed_suffix_uses_exact_paths_instead_of_including_tied_module():
    model = torch.nn.Module()
    model.left = torch.nn.Module()
    model.right = torch.nn.Module()
    model.left.proj = torch.nn.Linear(4, 4)
    model.right.proj = torch.nn.Linear(4, 4)
    model.alias = model.right.proj
    group = inspect_lora_modules(model, "mixed")["groups"][0]
    assert group["name"] == "proj"
    assert group["targets"] == ["left.proj"]
    _, modules = _training_targets(model, SimpleNamespace(layers=[], target_modules=group["targets"]), "lora")
    assert list(modules) == ["left.proj"]


def test_discovery_respects_live_linear_architecture():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.ModuleDict({
        "q_proj": torch.nn.Linear(4, 4), "v_proj": torch.nn.Linear(4, 4),
        "up_proj": torch.nn.Linear(4, 8), "norm": torch.nn.LayerNorm(4),
    }) for _ in range(3)])
    result = inspect_lora_modules(model, "custom")
    assert result["module_count"] == 9
    assert {group["name"] for group in result["groups"]} == {"q_proj", "v_proj", "up_proj"}


def test_service_inspects_only_existing_loaded_bundle(monkeypatch, model):
    calls = []
    @contextmanager
    def access():
        calls.append("lock")
        yield
    def require(model_id):
        calls.append(model_id)
        return {"model": model}
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(
        exclusive_runtime_access=access, require_loaded_runtime_bundle=require,
    ))
    assert get_lora_modules("tiny")["module_count"] == 8
    assert calls == ["lock", "tiny"]


def test_service_worker_delegation_and_dispatch(monkeypatch, model):
    runtime = SimpleNamespace(
        exclusive_runtime_access=nullcontext, require_loaded_runtime_bundle=lambda _: {"model": model},
    )
    def execute(action, payload, **kwargs):
        assert action == "lora_modules" and payload == {"model_id": "tiny"}
        return _dispatch(runtime, action, payload)
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=True, execute=execute))
    assert get_lora_modules("tiny")["module_count"] == 8


def test_route_requires_loaded_model(monkeypatch):
    def unavailable(_):
        raise ModelNotLoadedError("Load tiny before inspecting modules.")
    monkeypatch.setattr(routes_editing, "get_lora_modules", unavailable)
    with pytest.raises(HTTPException) as error:
        routes_editing.lora_modules(routes_editing.LoraModulesRequest(model_id="tiny"))
    assert error.value.status_code == 409


def test_discovery_rejects_unmerged_adapters(model):
    model.peft_config = {"default": {}}
    with pytest.raises(ValueError, match="merged base model"):
        inspect_lora_modules(model, "tiny")
