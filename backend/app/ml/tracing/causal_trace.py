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
    samples: int = 10,
    seed: int = 1,
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
                "samples": samples,
                "seed": seed,
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
    if samples < 1:
        raise ValueError("Causal tracing needs at least one corrupted sample.")

    try:
        import numpy
        import torch
    except ImportError as exc:
        raise RuntimeError("Install backend[ml] and NumPy to run real causal tracing.") from exc

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
    subject_token_indices = _find_subject_token_indices(tokenizer, input_ids_cpu[0].tolist(), subject)
    target_id = _target_id(tokenizer, target)

    embedding_layer = model.transformer.wte
    with torch.no_grad():
        clean_embeds = embedding_layer(input_ids)
    corrupted_embeds = clean_embeds.detach().expand(samples + 1, -1, -1).clone()
    noise_level = float(corruption.get("noise_level", 3.0))
    std = clean_embeds.std().detach().clamp_min(1e-6)
    random_state = numpy.random.RandomState(seed)
    subject_noise = torch.from_numpy(
        random_state.randn(samples, len(subject_token_indices), clean_embeds.shape[-1])
    ).to(device=corrupted_embeds.device, dtype=corrupted_embeds.dtype)
    corrupted_embeds[1:, subject_token_indices, :] += subject_noise * std * noise_level

    with torch.no_grad():
        clean_outputs = model(inputs_embeds=clean_embeds, output_hidden_states=True, use_cache=False)
        corrupted_outputs = model(inputs_embeds=corrupted_embeds, use_cache=False)

    clean_probability = _next_token_probability(clean_outputs.logits, target_id)
    corrupted_probability = _mean_next_token_probability(corrupted_outputs.logits[1:], target_id)

    heatmap = []
    for layer in range(model_info["n_layers"]):
        block = model.transformer.h[layer]
        clean_layer_state = clean_outputs.hidden_states[layer + 1].detach()

        for token_index, token in enumerate(tokens):
            handle = block.register_forward_hook(_restore_token_hook(clean_layer_state, token_index))
            try:
                with torch.no_grad():
                    restored_outputs = model(inputs_embeds=corrupted_embeds, use_cache=False)
            finally:
                handle.remove()

            restored_probability = _mean_next_token_probability(restored_outputs.logits[1:], target_id)
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
        "samples": samples,
        "seed": seed,
        "target": target,
        "tokens": tokens,
        "subject_token_indices": subject_token_indices,
        "heatmap": heatmap,
        "recommended_layer": _select_recommended_layer(heatmap, subject_token_indices),
        "runtime": "transformers",
    }


def _select_recommended_layer(heatmap: list[dict], subject_token_indices: list[int]) -> int:
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
            hidden[1:, token_index, :] = clean_layer_state[0, token_index, :]
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[1:, token_index, :] = clean_layer_state[0, token_index, :]
        return hidden

    return hook


def _find_subject_token_indices(tokenizer, token_ids: list[int], subject: str) -> list[int]:
    """Match subject text to decoded tokens as ROME's causal-trace experiment does."""
    if not subject:
        raise ValueError("Subject is required for causal tracing.")
    token_text = [tokenizer.decode([token_id]) for token_id in token_ids]
    prompt_text = "".join(token_text)
    subject_start = prompt_text.find(subject)
    if subject_start < 0:
        raise ValueError(f"Subject {subject!r} does not occur in the prompt after tokenization.")
    subject_end = subject_start + len(subject)
    cursor = 0
    start_index: int | None = None
    for index, token in enumerate(token_text):
        next_cursor = cursor + len(token)
        if start_index is None and next_cursor > subject_start:
            start_index = index
        if start_index is not None and next_cursor >= subject_end:
            return list(range(start_index, index + 1))
        cursor = next_cursor
    raise ValueError(f"Subject {subject!r} could not be mapped to prompt tokens.")


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


def _mean_next_token_probability(logits, target_id: int) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install backend[ml] to run real causal tracing.") from exc
    probabilities = torch.softmax(logits[:, -1, :], dim=-1)
    return round(float(probabilities[:, target_id].mean().item()), 8)
