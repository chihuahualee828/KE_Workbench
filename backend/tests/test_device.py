import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.ml import device
from app.ml.adapters.gpt2_adapter import GPT2Adapter


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
    monkeypatch.setattr(gpt2_adapter, "_unload_other_adapters_locked", lambda _: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32="fp32", float64="fp64", no_grad=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)))
    adapter = GPT2Adapter("test")
    adapter.load_model(local_files_only=True)
    common = {
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
