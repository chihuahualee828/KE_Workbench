from __future__ import annotations

from app.ml.editing.easyedit_runtime import (
    EasyEditUnsupportedError,
    build_prompt,
    run_easyedit,
)
from app.ml.editing.patch_format import edit_to_patch_entry
from app.ml.inference.generate import generate_text
from app.ml.tracing.causal_trace import run_causal_trace
from app.services.model_registry import model_registry


def run_rome_edit(
    model_id: str,
    subject: str,
    relation_prompt: str,
    new_object: str,
    ground_truth: str | None,
    target_layer: int | None,
    evaluation_prompts: list[str],
    rephrase_prompts: list[str],
    locality_inputs: dict | None,
    portability_inputs: dict | None,
    sequential_edit: bool,
) -> dict:
    edit = edit_to_patch_entry(subject, relation_prompt, new_object)
    if ground_truth is not None:
        edit["ground_truth"] = ground_truth
    all_rephrases = [*evaluation_prompts, *rephrase_prompts]

    auto_location = None
    model_info = model_registry.get(model_id)
    is_trace_compatible = model_info.get("model_type") == "gpt2" or model_info["hf_id"] in {"gpt2", "gpt2-medium", "gpt2-xl"}
    layer_source = "request" if target_layer is not None else "easyedit_hparams"
    edit_layers = [target_layer] if target_layer is not None else None
    if target_layer is None:
        if is_trace_compatible and ground_truth:
            trace = run_causal_trace(
                model_id=model_id,
                prompt=build_prompt(subject, relation_prompt),
                subject=subject,
                target=ground_truth,
                component="mlp",
                corruption={"type": "gaussian_noise", "noise_level": 3.0},
            )
            located_layer, layer_scores = _select_layer_from_trace(trace)
            edit_layers = [located_layer]
            layer_source = "causal_trace"
            auto_location = {
                "selected_layer": located_layer,
                "layer_scores": layer_scores,
                "clean_probability": trace["clean_probability"],
                "corrupted_probability": trace["corrupted_probability"],
                "subject_token_indices": trace["subject_token_indices"],
            }

    result = run_easyedit(
        method="rome",
        model_id=model_id,
        edits=[edit],
        layers=edit_layers,
        rephrase_prompts=all_rephrases,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=sequential_edit,
        layer_source=layer_source,
    )
    if auto_location is not None:
        result["auto_location"] = auto_location
    result["generations"] = _generate_evaluation_outputs(
        model_id=model_id,
        patch_id=result["patch_id"],
        prompts=evaluation_prompts,
    )
    return result


def _generate_evaluation_outputs(model_id: str, patch_id: str, prompts: list[str]) -> list[dict]:
    outputs = []
    for prompt in prompts:
        generated = generate_text(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=32,
            decoding_strategy="greedy",
            patch_id=patch_id,
        )
        outputs.append({"prompt": prompt, "text": generated["text"]})
    return outputs


def _select_layer_from_trace(trace: dict) -> tuple[int, list[dict]]:
    subject_indices = set(trace.get("subject_token_indices") or [])
    heatmap = trace.get("heatmap") or []
    if not heatmap:
        raise EasyEditUnsupportedError("Causal tracing produced no heatmap cells for layer selection.")

    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for cell in heatmap:
        if subject_indices and cell["token_index"] not in subject_indices:
            continue
        layer = int(cell["layer"])
        effect = max(0.0, float(cell["indirect_effect"]))
        scores[layer] = scores.get(layer, 0.0) + effect
        counts[layer] = counts.get(layer, 0) + 1

    if not scores:
        for cell in heatmap:
            layer = int(cell["layer"])
            effect = max(0.0, float(cell["indirect_effect"]))
            scores[layer] = scores.get(layer, 0.0) + effect
            counts[layer] = counts.get(layer, 0) + 1

    ranked = [
        {
            "layer": layer,
            "score": round(score / max(1, counts[layer]), 8),
            "cell_count": counts[layer],
        }
        for layer, score in scores.items()
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if not ranked:
        raise EasyEditUnsupportedError("Unable to choose an edit layer from causal trace results.")
    return int(ranked[0]["layer"]), ranked
