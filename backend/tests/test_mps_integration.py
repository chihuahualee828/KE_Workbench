"""Native Apple GPU smoke tests; no downloads or user data mutations."""
from __future__ import annotations


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
