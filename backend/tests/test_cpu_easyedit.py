"""Run CPU edits through the real EasyEdit entry point without model downloads."""
import importlib
from contextlib import nullcontext

import pytest
import torch
import yaml

from app.ml.editing import easyedit_runtime as runtime
from app.ml.editing.edit_cancellation import _run_local_edit, request_cancel
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


@pytest.mark.parametrize("method,directory", [("rome", "ROME"), ("r_rome", "R-ROME"), ("memit", "MEMIT")])
@pytest.mark.parametrize("cancel", [False, True])
def test_cpu_edit_and_evaluation_ignore_cuda_profile(monkeypatch, tmp_path, method, directory, cancel):
    from test_mps_integration import tiny_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = runtime.METHODS[method]
    runtime._load_easyedit(spec["hparams_class"])
    path = tiny_model.__wrapped__(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    profile = runtime._resolve_easyedit_root() / "hparams" / directory / "gpt2-xl.yaml"
    config = yaml.safe_load(profile.read_text())
    config.update(device=0, layers=[0], v_loss_layer=1, v_num_grad_steps=5 if cancel else 2, v_lr=0.01,
                  mom2_adjustment=False, model_parallel=False)
    hparams_path = tmp_path / "hparams.yaml"
    hparams_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_store, "root", tmp_path / "data")
    monkeypatch.setattr(runtime, "_mom2_stats_dir", lambda: tmp_path / "stats")
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: hparams_path)
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "id": "tiny", "hf_id": "gpt2-tiny", "n_layers": 2, spec["support_key"]: True,
    })
    monkeypatch.setattr(runtime.model_runtime_service, "exclusive_runtime_access", nullcontext)
    monkeypatch.setattr(runtime.model_runtime_service, "require_loaded_runtime_bundle", lambda _: {
        "model": model, "tokenizer": tokenizer, "device": "cpu",
    })
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU editing attempted to initialize CUDA"))

    # Keep contexts and covariance small; editor setup, gradients, updates,
    # evaluation and patch persistence still run through the installed EasyEdit.
    algorithm = importlib.import_module(f"easyeditor.models.{method}.{method}_main")
    templates = [["{}"]] if method == "memit" else ["{}"]
    monkeypatch.setattr(algorithm, "get_context_templates", lambda *args, **kwargs: templates)
    if method == "memit":
        monkeypatch.setattr(algorithm, "get_cov", lambda *args, **kwargs: torch.eye(64))
    fact = {"subject": "Paris", "relation_prompt": "{} is in", "new_object": "Italy", "ground_truth": "France"}
    if cancel:
        original_weights = {name: value.clone() for name, value in model.state_dict().items()}
        gradients = [parameter.requires_grad for parameter in model.parameters()]
        original_hooks = [(module, dict(module._forward_pre_hooks), dict(module._forward_hooks)) for module in model.modules()]
        steps = []
        step = torch.optim.Adam.step

        def cancel_after_loss(optimizer, *args, **kwargs):
            result = step(optimizer, *args, **kwargs)
            steps.append(True)
            request_cancel("job_cpu_cancel")
            return result

        monkeypatch.setattr(torch.optim.Adam, "step", cancel_after_loss)
        result = _run_local_edit("job_cpu_cancel", lambda: runtime.run_easyedit(method, "tiny", [fact], [0]), {})
        assert result == {"_edit_canceled": True}
        assert len(steps) == 1  # Do not finish the remaining optimization iterations.
        assert patch_service.list_patches() == []
        assert patch_service.active_patches_for_model("tiny") == []
        for name, value in model.state_dict().items():
            assert torch.equal(value, original_weights[name])
        assert [parameter.requires_grad for parameter in model.parameters()] == gradients
        for module, before, after in original_hooks:
            assert module._forward_pre_hooks == before
            assert module._forward_hooks == after
        assert torch.isfinite(model(**tokenizer("Paris is in", return_tensors="pt")).logits).all()
        return
    original = model.transformer.h[0].mlp.c_proj.weight.detach().clone()
    result = runtime.run_easyedit(method, "tiny", [fact], [0])
    assert result["delta_count"] > 0
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert not torch.equal(model.transformer.h[0].mlp.c_proj.weight, original)
    quality = runtime.evaluate_easyedit_quality("tiny", method, [fact])
    assert quality["cases"][0]["reliability_score"] is not None
