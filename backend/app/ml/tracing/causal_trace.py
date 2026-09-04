from __future__ import annotations

from app.ml.adapters.gpt2_adapter import get_gpt2_adapter
from app.ml.tracing.activation_patch import indirect_effect
from app.ml.tracing.corruptions import describe_corruption
from app.services.model_runtime_service import model_runtime_service
from app.services.model_registry import model_registry
from app.services.patch_service import patch_service


def run_causal_trace(
    model_id: str,
    prompt: str,
    subject: str,
    target: str,
    component: str,
    corruption: dict,
    require_configured: bool = False,
) -> dict:
    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "causal_trace",
            {
                "model_id": model_id,
                "prompt": prompt,
                "subject": subject,
                "target": target,
                "component": component,
                "corruption": corruption,
                "require_configured": require_configured,
            },
        )
    model_info = model_registry.get(model_id)
    if require_configured and not model_info.get("supports_causal_trace"):
        raise ValueError(f"{model_id} is not configured for causal tracing.")
    is_gpt2_architecture = (
        model_info["hf_id"] in {"gpt2", "gpt2-medium", "gpt2-xl"}
        or model_info.get("model_type") == "gpt2"
    )
    if not is_gpt2_architecture:
        raise ValueError("Real causal tracing is currently implemented for GPT-2 family models only.")
    if component not in {"residual", "mlp"}:
        raise ValueError("Real causal tracing currently supports residual/MLP block output restoration.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install backend[ml] to run real causal tracing.") from exc

    runtime_bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
    tokenizer = runtime_bundle["tokenizer"]
    model = runtime_bundle["model"]
    adapter = get_gpt2_adapter(
        model_id=model_info["hf_id"],
        n_layers=model_info["n_layers"],
        trust_remote_code=bool(model_info.get("trust_remote_code", False)),
        generation_use_cache=bool(model_info.get("generation_use_cache", True)),
    )
    patch_service.sync_active_patches_to_model(model, model_id)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids_cpu = encoded["input_ids"]
    input_ids = adapter.move_to_model_device({"input_ids": input_ids_cpu})["input_ids"]
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids_cpu[0].tolist()]
    subject_token_indices = _find_token_subsequence(
        input_ids_cpu[0].tolist(),
        tokenizer.encode(subject, add_special_tokens=False),
    )
    target_id = _target_id(tokenizer, target)

    embedding_layer = model.transformer.wte
    clean_embeds = embedding_layer(input_ids)
    corrupted_embeds = clean_embeds.detach().clone()
    if subject_token_indices:
        noise_level = float(corruption.get("noise_level", 3.0))
        std = clean_embeds.std().detach().clamp_min(1e-6)
        for token_index in subject_token_indices:
            corrupted_embeds[:, token_index, :] += torch.randn_like(
                corrupted_embeds[:, token_index, :]
            ) * std * noise_level

    with torch.no_grad():
        clean_outputs = model(inputs_embeds=clean_embeds, output_hidden_states=True, use_cache=False)
        corrupted_outputs = model(inputs_embeds=corrupted_embeds, use_cache=False)

    clean_probability = _next_token_probability(clean_outputs.logits, target_id)
    corrupted_probability = _next_token_probability(corrupted_outputs.logits, target_id)

    heatmap = []
    for layer in range(model_info["n_layers"]):
        block = model.transformer.h[layer]
        clean_layer_state = clean_outputs.hidden_states[layer + 1].detach()

        for token_index, token in enumerate(tokens):
            handle = block.register_forward_hook(
                _restore_token_hook(clean_layer_state, token_index)
            )
            try:
                with torch.no_grad():
                    restored_outputs = model(inputs_embeds=corrupted_embeds, use_cache=False)
            finally:
                handle.remove()

            restored_probability = _next_token_probability(restored_outputs.logits, target_id)
            heatmap.append(
                {
                    "layer": layer,
                    "token_index": token_index,
                    "token": token,
                    "restored_probability": restored_probability,
                    "indirect_effect": indirect_effect(restored_probability, corrupted_probability),
                }
            )

    return {
        "clean_probability": clean_probability,
        "corrupted_probability": corrupted_probability,
        "component": component,
        "corruption": describe_corruption(corruption),
        "target": target,
        "tokens": tokens,
        "subject_token_indices": subject_token_indices,
        "heatmap": heatmap,
        "recommended_layer": _select_recommended_layer(heatmap, subject_token_indices),
        "runtime": "transformers",
    }


def _select_recommended_layer(
    heatmap: list[dict], subject_token_indices: list[int]
) -> int:
    """Choose the layer with the strongest mean positive subject-token effect."""
    if not heatmap:
        raise ValueError("Causal trace produced no layer scores.")

    subject_indices = set(subject_token_indices)
    scored_cells = (
        [cell for cell in heatmap if cell["token_index"] in subject_indices]
        if subject_indices
        else heatmap
    )
    scores: dict[int, dict[str, float | int]] = {}
    for cell in scored_cells:
        layer = int(cell["layer"])
        score = scores.setdefault(layer, {"total": 0.0, "count": 0})
        score["total"] = float(score["total"]) + max(0.0, float(cell["indirect_effect"]))
        score["count"] = int(score["count"]) + 1

    if not scores:
        raise ValueError("Causal trace produced no usable layer scores.")
    return max(
        scores,
        key=lambda layer: float(scores[layer]["total"]) / max(1, int(scores[layer]["count"])),
    )


def _restore_token_hook(clean_layer_state, token_index: int):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, token_index, :] = clean_layer_state[:, token_index, :]
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, token_index, :] = clean_layer_state[:, token_index, :]
        return hidden

    return hook


def _find_token_subsequence(tokens: list[int], subject_tokens: list[int]) -> list[int]:
    if not subject_tokens:
        return []
    for start in range(0, len(tokens) - len(subject_tokens) + 1):
        if tokens[start : start + len(subject_tokens)] == subject_tokens:
            return list(range(start, start + len(subject_tokens)))
    return []


def _target_id(tokenizer, target: str) -> int:
    candidates = [
        tokenizer.encode(f" {target}", add_special_tokens=False),
        tokenizer.encode(target, add_special_tokens=False),
    ]
    for token_ids in candidates:
        if len(token_ids) == 1:
            return token_ids[0]
    raise ValueError(f"Target {target!r} must map to one next-token ID for causal tracing.")


def _next_token_probability(logits, target_id: int) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install backend[ml] to run real causal tracing.") from exc
    probabilities = torch.softmax(logits[:, -1, :], dim=-1)
    return round(float(probabilities[0, target_id].item()), 8)
