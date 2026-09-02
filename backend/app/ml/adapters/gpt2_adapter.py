from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any

from app.ml.adapters.base import ModelAdapter

_ADAPTERS: dict[tuple[str, int, bool, bool], "GPT2Adapter"] = {}
_ADAPTER_LOCK = RLock()


class GPT2Adapter(ModelAdapter):
    model_type = "gpt2"

    def __init__(
        self,
        model_id: str = "gpt2-xl",
        n_layers: int = 48,
        trust_remote_code: bool = False,
        generation_use_cache: bool = True,
    ) -> None:
        self.model_id = model_id
        self.n_layers = n_layers
        self.trust_remote_code = trust_remote_code
        self.generation_use_cache = generation_use_cache
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None

    def load_model(self, local_files_only: bool = False) -> Any:
        with _ADAPTER_LOCK:
            if self._model is not None:
                return self._model
            _unload_other_adapters_locked(self)
            try:
                from transformers import AutoModelForCausalLM
                import torch
            except ImportError as exc:
                raise RuntimeError("Install backend[ml] to enable real model generation.") from exc
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            load_kwargs: dict[str, Any] = {
                "local_files_only": local_files_only,
                "trust_remote_code": self.trust_remote_code,
            }
            if device.startswith("cuda"):
                try:
                    import accelerate  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "CUDA is available, but accelerate is not installed. "
                        'Run pip install -e ".[dev,ml]" from backend again.'
                    ) from exc
                load_kwargs.update(
                    {
                        "device_map": {"": device},
                        "low_cpu_mem_usage": True,
                    }
                )
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            try:
                if device == "cpu":
                    self._model.to(device)
            except Exception:
                self._model = None
                self._device = None
                raise
            self._model.eval()
            try:
                self._device = str(next(self._model.parameters()).device)
            except StopIteration:
                self._device = "unknown"
            return self._model

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def loaded_model(self) -> Any | None:
        return self._model

    def loaded_tokenizer(self) -> Any | None:
        return self._tokenizer

    def unload_model(self) -> None:
        with _ADAPTER_LOCK:
            model = self._model
            self._model = None
            self._device = None
        del model
        _collect_runtime_memory()

    def unload_tokenizer(self) -> None:
        with _ADAPTER_LOCK:
            tokenizer = self._tokenizer
            self._tokenizer = None
        del tokenizer
        _collect_runtime_memory()

    def unload_runtime(self) -> None:
        self.unload_model()
        self.unload_tokenizer()

    def model_device(self) -> str | None:
        return self._device

    def model_vram_bytes(self) -> int:
        model = self._model
        if model is None:
            return 0
        total = 0
        try:
            parameters = list(model.parameters())
            buffers = list(model.buffers())
        except Exception:
            return 0
        for tensor in [*parameters, *buffers]:
            try:
                if tensor.device.type == "cuda":
                    total += tensor.numel() * tensor.element_size()
            except Exception:
                continue
        return total

    def move_to_model_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self._device or self._device == "cpu":
            return inputs
        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def load_tokenizer(self, local_files_only: bool = False) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install backend[ml] to enable real tokenization.") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
        return self._tokenizer

    def is_tokenizer_loaded(self) -> bool:
        return self._tokenizer is not None

    def tokenize(self, text: str) -> Any:
        return self.load_tokenizer()(text, return_tensors="pt")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        local_files_only: bool = False,
        decoding_strategy: str = "greedy",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: int = 4,
    ) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install backend[ml] to enable real generation.") from exc
        with _ADAPTER_LOCK:
            tokenizer = self.load_tokenizer(local_files_only=local_files_only)
            model = self.load_model(local_files_only=local_files_only)
            inputs = self.move_to_model_device(tokenizer(prompt, return_tensors="pt"))
            generation_kwargs = _decoding_kwargs(
                decoding_strategy=decoding_strategy,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
            )
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=self.generation_use_cache,
                    **generation_kwargs,
                )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def get_residual_hook_name(self, layer: int) -> str:
        return f"transformer.h.{layer}"

    def get_mlp_input_hook_name(self, layer: int) -> str:
        return f"transformer.h.{layer}.mlp.c_fc"

    def get_mlp_output_hook_name(self, layer: int) -> str:
        return f"transformer.h.{layer}.mlp.c_proj"

    def get_attention_output_hook_name(self, layer: int) -> str:
        return f"transformer.h.{layer}.attn.c_proj"

    def get_default_rome_layer(self) -> int:
        return 17 if self.n_layers > 17 else max(0, self.n_layers - 2)

    def get_default_memit_layers(self) -> list[int]:
        return list(range(3, min(self.n_layers, 9)))

    def find_subject_token_indices(self, prompt: str, subject: str) -> list[int]:
        words = prompt.split()
        subject_words = subject.split()
        for start in range(0, max(0, len(words) - len(subject_words) + 1)):
            if words[start : start + len(subject_words)] == subject_words:
                return list(range(start, start + len(subject_words)))
        return []


def get_gpt2_adapter(
    model_id: str = "gpt2-xl",
    n_layers: int = 48,
    trust_remote_code: bool = False,
    generation_use_cache: bool = True,
) -> GPT2Adapter:
    key = (model_id, n_layers, trust_remote_code, generation_use_cache)
    with _ADAPTER_LOCK:
        adapter = _ADAPTERS.get(key)
        if adapter is None:
            adapter = GPT2Adapter(
                model_id=model_id,
                n_layers=n_layers,
                trust_remote_code=trust_remote_code,
                generation_use_cache=generation_use_cache,
            )
            _ADAPTERS[key] = adapter
        return adapter


def unload_all_adapters() -> None:
    released: list[Any] = []
    with _ADAPTER_LOCK:
        for adapter in _ADAPTERS.values():
            released.extend([adapter._model, adapter._tokenizer])
            adapter._model = None
            adapter._tokenizer = None
            adapter._device = None
    released.clear()
    _collect_runtime_memory()


def unload_other_adapters(active_adapter: GPT2Adapter) -> None:
    """Release every runtime adapter except the one about to be loaded."""
    with _ADAPTER_LOCK:
        _unload_other_adapters_locked(active_adapter)


@contextmanager
def adapter_runtime_lock():
    with _ADAPTER_LOCK:
        yield


def _unload_other_adapters_locked(active_adapter: GPT2Adapter) -> None:
    unloaded_any = False
    released: list[Any] = []
    for adapter in _ADAPTERS.values():
        if adapter is active_adapter:
            continue
        if adapter.is_model_loaded() or adapter.is_tokenizer_loaded():
            released.extend([adapter._model, adapter._tokenizer])
            adapter._model = None
            adapter._tokenizer = None
            adapter._device = None
            unloaded_any = True
    released.clear()
    if unloaded_any:
        _collect_runtime_memory()


def _collect_runtime_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except ImportError:
        pass


def _decoding_kwargs(
    *,
    decoding_strategy: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
) -> dict[str, Any]:
    if decoding_strategy == "greedy":
        return {"do_sample": False, "num_beams": 1}
    if decoding_strategy == "sampling":
        return {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": 1,
        }
    if decoding_strategy == "beam":
        return {"do_sample": False, "num_beams": num_beams, "early_stopping": True}
    raise ValueError(f"Unsupported decoding strategy: {decoding_strategy}")
