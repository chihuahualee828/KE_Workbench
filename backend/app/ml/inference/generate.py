from __future__ import annotations

from app.ml.adapters.gpt2_adapter import adapter_runtime_lock, get_gpt2_adapter
from app.services.model_cache_service import model_cache_service
from app.services.model_registry import model_registry
from app.services.patch_service import patch_service


def generate_text(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 20,
    decoding_strategy: str = "greedy",
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    num_beams: int = 4,
    patch_id: str | None = None,
    use_active_patches: bool = True,
) -> dict[str, str]:
    model = model_registry.get(model_id)
    adapter = get_gpt2_adapter(model_id=model["hf_id"], n_layers=model["n_layers"])
    local_files_only = model_cache_service.cache_status(model_id)["cached"]
    with adapter_runtime_lock():
        loaded_model = adapter.load_model(local_files_only=local_files_only)
        with patch_service.applied_to_model(
            loaded_model,
            patch_id,
            model_id=model_id,
            use_active_patches=use_active_patches,
        ):
            return {
                "text": adapter.generate(
                    prompt,
                    max_new_tokens,
                    local_files_only=local_files_only,
                    decoding_strategy=decoding_strategy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                ),
                "runtime": "transformers",
            }
