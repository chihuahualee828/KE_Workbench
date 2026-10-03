"""Exercise unmodified EasyEdit functions with the workbench's FP32 policy.

Only activation collection is replaced with small, deterministic tensors. The
covariance inverse, MEMIT solve, device transfers, and weight updates are real.
"""
from __future__ import annotations
import __future__

import ast
import collections
import math
import runpy
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_mps_integration
import yaml

from app.config import settings
from app.ml.editing.mps_compat import configure_mps_hparams, mps_fp32

tiny_model = test_mps_integration.tiny_model

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")


def functions_from_checkout(relative_path, names, namespace):
    path = settings.easyedit_root / "easyeditor" / relative_path
    if not path.exists():
        pytest.skip("Requires the EasyEdit checkout")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)]
    if names is not None:
        assert len(tree.body) == len(names)
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)  # noqa: S102
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("family", ["rome", "r_rome"])
def test_fp64_cached_covariance_converts_before_mps_transfer(family):
    covariance = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.float64)
    calls = []

    def stats(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(mom2=SimpleNamespace(moment=lambda: covariance))

    module = functions_from_checkout(f"models/{family}/compute_u.py", {"get_inv_cov"}, {
        "torch": torch, "normalize_device": torch.device, "layer_stats": stats, "inv_mom2_cache": {},
    })
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="tiny"))
    params = SimpleNamespace(device="mps", stats_dir="unused")
    with mps_fp32("mps"):
        result = module.get_inv_cov(model, None, "layer", "test", 1, "float32", hparams=params)
    assert result.device.type == "mps" and result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), torch.inverse(covariance).float())
    with mps_fp32("mps"):
        cached = module.get_inv_cov(model, None, "layer", "test", 1, "float32", hparams=params)
    assert cached is result and len(calls) == 1


def test_memit_uses_fp32_mps_solve_apply_and_restore(monkeypatch):
    model = torch.nn.Linear(3, 2, bias=False).to("mps")
    model.config = SimpleNamespace(_name_or_path="tiny")
    key = torch.tensor([[1., 2., 3.]], device="mps")
    offset = torch.tensor([0.2, -0.1], device="mps")
    baseline = model.weight.detach().clone()
    initial_output = model(key).detach()[0]

    def copy_to_param(parameter, value):
        with torch.no_grad():
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    functions = functions_from_checkout("models/memit/memit_main.py", {
        "apply_memit_to_model", "execute_memit", "get_cov", "upd_matrix_match_shape",
    }, {
        "torch": torch, "deepcopy": deepcopy, "Path": Path,
        "normalize_device": torch.device, "copy_to_param": copy_to_param,
        "get_module_device": lambda module, _: next(module.parameters()).device,
        "nethook": SimpleNamespace(get_parameter=lambda m, _: m.weight, get_module=lambda m, _: m),
        "get_context_templates": lambda *_: [["{}"]],
        "compute_z": lambda *_: initial_output + offset,
        "compute_ks": lambda *_: key,
        "get_module_input_output_at_words": lambda *args, **kwargs: model(key).detach(),
        "COV_CACHE": {("tiny", "layer.0"): torch.eye(3)},
    })
    params = SimpleNamespace(
        device="mps", layers=[0], rewrite_module_tmp="layer.{}", layer_module_tmp="layer.{}",
        fact_token="subject_last", mom2_dataset="test", mom2_n_samples=1, mom2_dtype="float32",
        mom2_update_weight=2.,
    )
    requests = [{"prompt": "{} is in", "subject": "Paris", "target_new": "France"}]
    original_solve = torch.linalg.solve
    solves = []

    def check_solve(a, b):
        assert a.device.type == b.device.type == "mps"
        assert a.dtype == b.dtype == torch.float32
        solves.append(True)
        return original_solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", check_solve)
    with mps_fp32("mps"):
        deltas = functions.execute_memit(model, None, requests, params)
    torch.testing.assert_close(model.weight, baseline, rtol=0, atol=0)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in deltas["layer.0.weight"])
    with mps_fp32("mps"):
        functions.apply_memit_to_model(model, None, requests, params)
    assert len(solves) == 2
    key_cpu = key.cpu().double().T
    adjusted = original_solve(2 * torch.eye(3, dtype=torch.float64) + key_cpu @ key_cpu.T, key_cpu)
    expected = baseline.cpu() + (offset.cpu().double()[:, None] @ adjusted.T).float()
    torch.testing.assert_close(model.weight.cpu(), expected)
    assert model.weight.device.type == "mps"
    assert torch.isfinite(model.weight).all()


@pytest.mark.parametrize("family,hparams_dir", [("rome", "ROME"), ("r_rome", "R-ROME"), ("memit", "MEMIT")])
def test_real_edit_gradients_on_mps(tiny_model, family, hparams_dir):
    """Run the upstream algorithms with real hooks, autograd and tiny weights.

    Fixed context templates and identity covariance keep this offline. Import
    function definitions directly to avoid EasyEdit's unrelated multimodal and
    training dependencies; no editing/gradient function is replaced.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    functions_from_checkout("util/device.py", {"normalize_device"}, {"torch": torch})
    nethook = SimpleNamespace(**runpy.run_path(str(settings.easyedit_root / "easyeditor/util/nethook.py")))
    common = {
        "torch": torch, "np": np, "deepcopy": deepcopy, "Path": Path,
        "nethook": nethook, "normalize_device": torch.device,
        "get_module_device": lambda module, _: next(module.parameters()).device,
        "copy_to_param": lambda param, value: param.copy_(value.to(device=param.device, dtype=param.dtype)),
    }
    common["repr_tools"] = functions_from_checkout("models/rome/repr_tools.py", None, dict(common))
    model = AutoModelForCausalLM.from_pretrained(tiny_model).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    params = yaml.safe_load((settings.easyedit_root / f"hparams/{hparams_dir}/gpt2-xl.yaml").read_text())
    params.update(device="mps", layers=[0], v_loss_layer=1, v_num_grad_steps=2, v_lr=0.01,
                  mom2_adjustment=False, enable_random_prefix_keys=False,
                  original_implementation=False, enable_prompt_keys=True)
    if family == "memit":
        compute_z = functions_from_checkout("models/memit/compute_z.py", None, dict(common))
        common.update(compute_z=compute_z.compute_z,
                      get_module_input_output_at_words=compute_z.get_module_input_output_at_words)
        compute_ks = functions_from_checkout("models/memit/compute_ks.py", None, dict(common))
        common.update(compute_ks=compute_ks.compute_ks, CONTEXT_TEMPLATES_CACHE=[["{}"]],
                      COV_CACHE={(str(tiny_model).replace("/", "_"), "transformer.h.0.mlp.c_proj"): torch.eye(64)})
    else:
        left = functions_from_checkout(f"models/{family}/compute_u.py", None, dict(common))
        right = functions_from_checkout(f"models/{family}/compute_v.py", None, dict(common))
        common.update(compute_u=left.compute_u, compute_v=right.compute_v, CONTEXT_TEMPLATES_CACHE=["{}"])
    algorithm = functions_from_checkout(f"models/{family}/{family}_main.py", None, common)
    weight = model.transformer.h[0].mlp.c_proj.weight
    original = weight.detach().clone()
    apply = getattr(algorithm, f"apply_{family}_to_model")
    with mps_fp32("mps"):
        apply(model, tokenizer, [{"prompt": "{} is in", "subject": "Paris", "target_new": "France"}], SimpleNamespace(**params))
    assert weight.device.type == "mps" and torch.isfinite(weight).all()
    assert not torch.equal(weight, original)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("family", ["rome", "r_rome"])
def test_fp64_mom2_config_is_converted_to_fp32_on_mps(tiny_model, tmp_path, family):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    functions_from_checkout("util/device.py", {"normalize_device"}, {"torch": torch})
    root = settings.easyedit_root / "easyeditor"
    stats = runpy.run_path(str(root / "util/runningstats.py"))
    hooks = runpy.run_path(str(root / "util/nethook.py"))
    dataset = runpy.run_path(str(root / "models/rome/tok_dataset.py"))

    def tally(*args, **kwargs):
        kwargs["num_workers"] = 0  # Same setting used by the app's loader wrapper.
        return stats["tally"](*args, **kwargs)

    accumulation = []

    class CheckedMoment(stats["SecondMoment"]):
        def add(self, data):
            accumulation.append((data.device.type, data.dtype))
            return super().add(data)

    module = functions_from_checkout(f"models/{family}/layer_stats.py", {"layer_stats"}, {
        **dataset, **hooks, **stats, "torch": torch, "Path": Path,
        "normalize_device": torch.device, "tally": tally,
        "tqdm": lambda values, **kwargs: values,
        "load_dataset": lambda *_: {"train": [{"text": "Paris is in France"}, {"text": "Rome is in Italy"}]},
        "STAT_TYPES": {"mom2": CheckedMoment},
    })
    model = AutoModelForCausalLM.from_pretrained(tiny_model).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    params = SimpleNamespace(device="mps", mom2_dtype="float64")
    configure_mps_hparams(params)
    with mps_fp32("mps"):
        result = module.layer_stats(
            model, tokenizer, "transformer.h.0.mlp.c_proj", tmp_path / "stats", "wikipedia",
            ["mom2"], sample_size=2, precision=params.mom2_dtype, hparams=params,
        )
    moment = result.mom2.moment()
    assert accumulation and all(item == ("mps", torch.float32) for item in accumulation)
    # EasyEdit serializes completed statistics to CPU; accumulation stayed on MPS.
    assert moment.dtype == torch.float32
    assert moment.shape == (64, 64) and torch.isfinite(moment).all()
    assert next(model.parameters()).device.type == "mps"
    assert all(not module._forward_hooks for module in model.modules())
    reference = module.layer_stats(
        model.cpu(), tokenizer, "transformer.h.0.mlp.c_proj", tmp_path / "cpu-stats", "wikipedia",
        ["mom2"], sample_size=2, precision="float64", hparams=SimpleNamespace(device="cpu"),
    ).mom2.moment()
    torch.testing.assert_close(moment.cpu().double(), reference, rtol=0.005, atol=1e-6)


def test_kn_attribution_edit_and_undo_on_mps(tiny_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    einops = pytest.importorskip("einops")
    functions_from_checkout("util/device.py", {"normalize_device"}, {"torch": torch})
    relative = "models/kn/knowledge_neurons/knowledge_neurons"
    patch = runpy.run_path(str(settings.easyedit_root / "easyeditor" / relative / "patch.py"))
    module = functions_from_checkout(f"{relative}/knowledge_neurons.py", {"KnowledgeNeurons"}, {
        **patch, "einops": einops, "np": np, "math": math, "collections": collections,
        "partial": partial, "tqdm": lambda values, **kwargs: values,
    })
    model = AutoModelForCausalLM.from_pretrained(tiny_model).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    kn = module.KnowledgeNeurons(model, tokenizer, model_type="gpt2", device="mps")
    scores = kn.get_scores("Paris is in", "France", batch_size=2, steps=2, pbar=False)
    assert scores.device.type == "mps" and torch.isfinite(scores).all()
    assert scores.shape == (2, 64)
    weight = model.transformer.h[0].mlp.c_proj.weight
    original = weight.detach().clone()
    with torch.no_grad():
        prediction = model(**tokenizer("Paris is in", return_tensors="pt").to("mps")).logits[0, -1].argmax().item()
    target = "Italy" if prediction == tokenizer.convert_tokens_to_ids("France") else "France"
    _, undo = kn.edit_knowledge("Paris is in", target, [[0, 0]], undo_modification=False, quiet=True)
    assert not torch.equal(weight, original) and torch.isfinite(weight).all()
    undo()
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert all(not module._forward_hooks for module in model.modules())


def test_easyedit_prediction_and_locality_scoring_on_mps(tiny_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    module = functions_from_checkout("evaluate/evaluate_utils.py", {"test_prediction_acc", "slice_list"}, {
        "torch": torch, "np": np, "normalize_device": torch.device,
    })
    model = AutoModelForCausalLM.from_pretrained(tiny_model).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    params = SimpleNamespace(max_length=16)
    for locality in (False, True):
        result = module.test_prediction_acc(model, tokenizer, params, "Paris is in", "France", "mps", locality=locality)
        assert result and np.isfinite(result).all()
        expected = module.test_prediction_acc(model.cpu(), tokenizer, params, "Paris is in", "France", "cpu", locality=locality)
        assert result == expected
        model.to("mps")


def test_dtype_mode_is_scoped_and_never_offloads():
    x = torch.ones(3, device="mps")
    original_double = torch.Tensor.double
    with pytest.raises(ValueError, match="test cleanup"), mps_fp32("mps"):
        for result in (x.double(), x.to(torch.float64), x.to(dtype=torch.float64)):
            assert result.device.type == "mps" and result.dtype == torch.float32
        assert torch.ones(3).double().dtype == torch.float64
        raise ValueError("test cleanup")
    assert torch.Tensor.double is original_double
    with pytest.raises(TypeError, match="float64"):
        x.double()
    with mps_fp32("cuda:0"):
        assert torch.ones(3).double().dtype == torch.float64
