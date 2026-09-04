from __future__ import annotations

from app.ml.adapters.gpt2_adapter import get_gpt2_adapter
from app.services.model_runtime_service import model_runtime_service
from app.services.model_registry import model_registry
from app.services.patch_service import patch_service


def run_logit_lens(
    model_id: str,
    prompt: str,
    target_token: str | None = None,
    top_k: int = 10,
) -> dict:
    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "logit_lens",
            {
                "model_id": model_id,
                "prompt": prompt,
                "target_token": target_token,
                "top_k": top_k,
            },
        )
    model = model_registry.get(model_id)
    if not model.get("supports_logit_lens"):
        raise ValueError(f"{model_id} is not configured for logit lens.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install backend[ml] to run real logit lens.") from exc

    runtime_bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
    tokenizer = runtime_bundle["tokenizer"]
    loaded_model = runtime_bundle["model"]
    adapter = get_gpt2_adapter(
        model_id=model["hf_id"],
        n_layers=model["n_layers"],
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        generation_use_cache=bool(model.get("generation_use_cache", True)),
    )
    patch_service.sync_active_patches_to_model(loaded_model, model_id)
    inputs = adapter.move_to_model_device(tokenizer(prompt, return_tensors="pt"))
    target_id = _target_token_id(tokenizer, target_token) if target_token else None

    with torch.no_grad():
        outputs = loaded_model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    layers = []
    for layer_index, hidden_state in enumerate(outputs.hidden_states):
        projected = _project_hidden_state(loaded_model, hidden_state[:, -1, :])
        probabilities = torch.softmax(projected, dim=-1)
        top_values, top_indices = torch.topk(projected[0], k=top_k)
        top_tokens = [
            (tokenizer.decode([token_id]), round(float(value), 6))
            for token_id, value in zip(top_indices.tolist(), top_values.tolist())
        ]

        target_rank = None
        target_logit = None
        target_probability = None
        if target_id is not None:
            target_logit_tensor = projected[0, target_id]
            target_rank = int((projected[0] > target_logit_tensor).sum().item() + 1)
            target_logit = round(float(target_logit_tensor.item()), 6)
            target_probability = round(float(probabilities[0, target_id].item()), 8)

        layers.append(
            {
                "layer": layer_index - 1,
                "target_rank": target_rank,
                "target_logit": target_logit,
                "target_probability": target_probability,
                "top_tokens": top_tokens,
            }
        )

    return {"prompt": prompt, "layers": layers, "runtime": "transformers"}


def _project_hidden_state(model, hidden_state):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        hidden_state = model.transformer.ln_f(hidden_state)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        hidden_state = model.gpt_neox.final_layer_norm(hidden_state)
    if not hasattr(model, "lm_head"):
        raise RuntimeError("Model does not expose an lm_head for logit lens projection.")
    return model.lm_head(hidden_state)


def _target_token_id(tokenizer, target_token: str | None) -> int | None:
    if not target_token:
        return None
    token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if not token_ids:
        return None
    if len(token_ids) > 1:
        raise ValueError(
            f"Target token {target_token!r} maps to multiple tokens. Use a single tokenizer token."
        )
    return token_ids[0]
