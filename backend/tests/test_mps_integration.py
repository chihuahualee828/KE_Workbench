"""Native Apple GPU smoke tests; no downloads or user data mutations."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")


@pytest.fixture
def tiny_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(7)
    vocab = {word: i for i, word in enumerate([
        "<pad>", "<unk>", "<eos>", "Paris", "is", "in", "France", "Italy",
        "Rome", "the", "city", "of", "a", "capital", "London", ".",
    ])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_layer=2, n_head=2, n_embd=16, n_positions=64,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))
    path = tmp_path / "tiny-gpt2"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.mark.parametrize("saved_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mps_loading_preserves_checkpoint_precision(tiny_model, tmp_path, saved_dtype):
    from transformers import AutoModelForCausalLM

    from app.ml.adapters.gpt2_adapter import GPT2Adapter

    source = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=saved_dtype)
    checkpoint = tmp_path / "typed-checkpoint"
    source.save_pretrained(checkpoint)
    original_file = (checkpoint / "model.safetensors").read_bytes()
    expected = torch.float32 if saved_dtype == torch.float64 else saved_dtype
    adapter = GPT2Adapter(str(checkpoint), n_layers=2)
    try:
        model = adapter.load_model(local_files_only=True)
        assert all(p.device.type == "mps" and p.dtype == expected for p in model.parameters())
        assert model.lm_head.weight is model.transformer.wte.weight
        inputs = torch.tensor([[3, 4, 5]], device="mps")
        output = model(inputs, labels=inputs)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert (checkpoint / "model.safetensors").read_bytes() == original_file
    finally:
        adapter.unload_model()


def test_real_mps_worker_lifecycle(tiny_model, tmp_path):
    # A separate controller gives the spawned worker the same isolated DATA_DIR.
    # It also verifies that telemetry works across processes, not just in tests
    # that happen to query their own MPS allocator.
    script = r'''
import sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from app.services.model_registry import model_registry
from app.services.model_runtime_service import model_runtime_service as runtime
from app.services.patch_service import patch_service
model_registry.add_local_model({
    "id": "mps-test", "local_path": sys.argv[1], "model_type": "gpt2",
    "supports_logit_lens": True, "supports_causal_trace": True,
})
def load():
    job = runtime.start_load("mps-test")
    job = runtime.wait_job(job["job_id"], timeout=120)
    assert job["status"] == "completed", job
    status = runtime.runtime_status("mps-test")
    assert status["device"].startswith("mps"), status
    assert status["model_vram_bytes"] > 0, status
try:
    load()
    for strategy in ("greedy", "sampling", "beam"):
        result = runtime.execute("generate", {
            "model_id": "mps-test", "prompt": "Paris is in", "max_new_tokens": 3,
            "decoding_strategy": strategy, "top_k": 5, "num_beams": 2,
        })
        assert result["text"].startswith("Paris is in"), result
    lens = runtime.execute("logit_lens", {"model_id": "mps-test", "prompt": "Paris is in", "top_k": 3})
    assert len(lens["layers"]) == 3, lens
    trace = runtime.execute("causal_trace", {
        "model_id": "mps-test", "prompt": "Paris is in", "subject": "Paris", "target": "France",
        "component": "residual", "corruption": {"noise_level": 1.0}, "samples": 2,
    })
    assert trace["heatmap"], trace
    time.sleep(1.1)
    gpu = runtime.runtime_status("mps-test")["gpu"]
    assert gpu["mps_available"] and gpu["allocated_bytes"] > 0, gpu
    assert gpu["driver_allocated_bytes"] >= gpu["allocated_bytes"], gpu
    assert time.time() - gpu["sampled_at"] < 5, gpu
    runtime.export_active_model("mps-test", sys.argv[2])
    weight_name = "transformer.h.0.mlp.c_proj.weight"
    baseline = load_file(str(Path(sys.argv[2]) / "model.safetensors"))[weight_name]
    delta_path = Path(sys.argv[2]).parent / "test-delta.safetensors"
    delta = torch.full_like(baseline, 0.01)
    save_file({weight_name: delta}, str(delta_path))
    patch = patch_service.create_patch(
        "mps-test", "rome", [], [0], delta_paths=[str(delta_path)],
    )
    patch_service.activate_patch(patch["patch_id"])
    runtime.sync_loaded_model_to_active_patches("mps-test")
    patched_export = sys.argv[2] + "-patched"
    runtime.export_active_model("mps-test", patched_export)
    patched = load_file(str(Path(patched_export) / "model.safetensors"))[weight_name]
    torch.testing.assert_close(patched, baseline + delta)
    patch_service.clear_active_patches()
    runtime.sync_loaded_model_to_active_patches("mps-test")
    runtime.export_active_model("mps-test", sys.argv[2] + "-reverted")
    reverted = load_file(str(Path(sys.argv[2] + "-reverted") / "model.safetensors"))[weight_name]
    torch.testing.assert_close(reverted, baseline)
    process = runtime._process
    status = runtime.unload("mps-test")
    assert status["hard_unloaded"] and not status["loaded"], status
    assert not process.is_alive()
    assert runtime._worker_memory is None
    assert runtime.runtime_status("mps-test")["gpu"]["allocated_bytes"] == 0
    load()
    process = runtime._process
    assert runtime.cancel_active_operation("MPS lifecycle test")["worker_stopped"]
    assert not process.is_alive()
    assert not runtime.runtime_status("mps-test")["loaded"]
finally:
    runtime.shutdown()
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tiny_model), str(tmp_path / "export")],
        env={**os.environ, "DATA_DIR": str(tmp_path / "data"), "HF_HUB_OFFLINE": "1"},
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "export" / "config.json").is_file()
