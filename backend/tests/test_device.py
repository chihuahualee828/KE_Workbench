import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.ml import device
from app.ml.adapters.gpt2_adapter import GPT2Adapter
from app.ml.editing.easyedit_runtime import _override_easyedit_device
from app.services.model_runtime_service import HardUnloadModelRuntimeService


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_priority(monkeypatch, cuda, mps, expected):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert device.model_device() == expected



@pytest.mark.parametrize("target", ["cuda:0", "mps", "cpu"])
def test_loading_preserves_cuda_policy(monkeypatch, target):
    from app.ml.adapters import gpt2_adapter

    calls = {}

    class Model:
        def to(self, target):
            calls["to"] = target

        def eval(self):
            pass

        def parameters(self):
            return iter([SimpleNamespace(device=target, dtype="fp32")])

        def buffers(self):
            return iter([])

    def load(name, **kwargs):
        calls["kwargs"] = kwargs
        return Model()

    monkeypatch.setattr(gpt2_adapter, "model_device", lambda: target)
    monkeypatch.setattr(GPT2Adapter, "_cache_directory", lambda _: "/test/cache")
    monkeypatch.setattr(gpt2_adapter, "_unload_other_adapters_locked", lambda _: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32="fp32", float64="fp64", no_grad=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)))
    adapter = GPT2Adapter("test")
    adapter.load_model(local_files_only=True)
    common = {
        "cache_dir": "/test/cache",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if target == "cuda:0":
        assert calls == {"kwargs": {**common, "device_map": {"": "cuda:0"}, "low_cpu_mem_usage": True}}
    else:
        assert calls == {"kwargs": {**common, **({"dtype": "auto"} if target == "mps" else {})}, "to": target}
    assert adapter.model_device() == target



def test_mps_downcasts_only_fp64_parameters_and_buffers(monkeypatch):
    import torch

    from app.ml.adapters import gpt2_adapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.low = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
            self.high = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            self.register_buffer("statistics", torch.ones(2, dtype=torch.float64))
            self.register_buffer("indices", torch.ones(2, dtype=torch.int64))

        def to(self, target):
            assert target == "mps"
            assert self.low.dtype == torch.float16
            assert self.high.dtype == self.statistics.dtype == torch.float32
            assert self.indices.dtype == torch.int64
            return self

    model = Model()
    low = model.low
    high = model.high
    monkeypatch.setattr(gpt2_adapter, "model_device", lambda: "mps")
    monkeypatch.setattr(gpt2_adapter, "_unload_other_adapters_locked", lambda _: None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    ))
    assert GPT2Adapter("test").load_model() is model
    assert model.low is low and model.high is high



def test_cuda_cleanup_is_unchanged(monkeypatch):
    from app.ml.adapters import gpt2_adapter

    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )))
    monkeypatch.setattr(gpt2_adapter, "release_mps_cache", lambda: pytest.fail("MPS cleanup on CUDA"))
    gpt2_adapter._collect_runtime_memory()
    assert calls == ["synchronize", "empty_cache", "ipc_collect"]


def test_cuda_easyedit_device_preserves_precision():
    hparams = SimpleNamespace(device=0, mom2_dtype="float64")
    _override_easyedit_device(hparams, "cuda:2")
    assert hparams.device == 2
    assert hparams.mom2_dtype == "float64"


@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("algorithm", [
    "ROME", "R-ROME", "MEMIT", "KN", "FT", "LoRA", "DPO", "EMMET", "MEMIT-FE",
    "PMET", "EAMET", "AlphaEdit", "SPHERE", None,
])
def test_cpu_easyedit_device_overrides_profile_for_every_method(monkeypatch, algorithm, cuda_available):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    ))
    hparams = SimpleNamespace(device=0, alg_name=algorithm, mom2_dtype="float64")
    _override_easyedit_device(hparams, "cpu")
    assert hparams.device == "cpu"
    assert hparams.mom2_dtype == "float64"


@pytest.mark.parametrize("precision,expected", [("float64", "float32"), ("double", "float32"), ("float32", "float32")])
def test_mps_easyedit_device_converts_hparams(precision, expected):
    hparams = SimpleNamespace(device=0, mom2_dtype=precision)
    _override_easyedit_device(hparams, "mps:0")
    assert hparams.device == "mps"
    assert hparams.mom2_dtype == expected


@pytest.mark.parametrize("target,precision", [("mps", "float32"), ("cuda:0", "float64"), ("cpu", "float64")])
def test_cache_profile_matches_runtime_precision(monkeypatch, tmp_path, target, precision):
    from app.ml.editing import easyedit_runtime as runtime

    path = tmp_path / "memit.yaml"
    path.write_text("layers: [0]\nrewrite_module_tmp: transformer.h.{}.mlp.c_proj\nmom2_dataset: wikipedia\nmom2_dtype: float64\nmom2_n_samples: 2\n")
    model = {"id": "demo", "hf_id": "demo", "source": "hub", "supports_memit": True}
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: model)
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: path)
    monkeypatch.setattr(runtime, "_mom2_stats_dir", lambda: tmp_path / "stats")
    monkeypatch.setattr(runtime.model_runtime_service, "runtime_status", lambda _: {
        "device": "not_loaded" if target == "mps" else target,
        "gpu": {"mps_available": target == "mps"},
    })
    profile, _ = runtime._load_mom2_cache_profile("demo", "memit", None)
    assert profile.mom2_dtype == precision
    worker_hparams = SimpleNamespace(**vars(profile))
    worker_hparams.device = 0
    worker_hparams.mom2_dtype = "float64"
    runtime._override_easyedit_device(worker_hparams, target)
    assert runtime._mom2_statistics_path(profile, None, 0) == runtime._mom2_statistics_path(worker_hparams, None, 0)


def test_worker_memory_is_process_usage_not_vram_capacity():
    runtime = HardUnloadModelRuntimeService()
    runtime._worker_memory = [1024, 2048, 8192, 123.0]
    memory = runtime._gpu_memory()
    assert memory["mps_available"]
    assert memory["allocated_bytes"] == 1024
    assert memory["driver_allocated_bytes"] == 2048
    assert memory["recommended_max_bytes"] == 8192
    assert memory["sampled_at"] == 123.0
    assert "total_bytes" not in memory
    assert "free_bytes" not in memory
