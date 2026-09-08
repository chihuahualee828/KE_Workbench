from __future__ import annotations

from app.ml.editing.easyedit_runtime import run_easyedit
from app.ml.editing.patch_format import edit_to_patch_entry
from app.ml.inference.generate import generate_text
from app.services.model_registry import model_registry


def run_r_rome_edit(
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
    mom2_adjustment: bool,
) -> dict:
    """Apply one EasyEdit R-ROME edit and persist its reversible weight delta."""
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "r_rome_edit",
            {
                "model_id": model_id,
                "subject": subject,
                "relation_prompt": relation_prompt,
                "new_object": new_object,
                "ground_truth": ground_truth,
                "target_layer": target_layer,
                "evaluation_prompts": evaluation_prompts,
                "rephrase_prompts": rephrase_prompts,
                "locality_inputs": locality_inputs,
                "portability_inputs": portability_inputs,
                "sequential_edit": sequential_edit,
                "mom2_adjustment": mom2_adjustment,
            },
            timeout=7200.0,
        )

    model_info = model_registry.get(model_id)
    if target_layer is not None and not 0 <= target_layer < int(model_info["n_layers"]):
        raise ValueError(
            f"Layer {target_layer} is outside {model_id}'s valid range 0-{int(model_info['n_layers']) - 1}."
        )

    edit = edit_to_patch_entry(subject, relation_prompt, new_object)
    if ground_truth is not None:
        edit["ground_truth"] = ground_truth
    result = run_easyedit(
        method="r_rome",
        model_id=model_id,
        edits=[edit],
        layers=[target_layer] if target_layer is not None else None,
        rephrase_prompts=[*evaluation_prompts, *rephrase_prompts],
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=sequential_edit,
        layer_source="request" if target_layer is not None else "easyedit_hparams",
        mom2_adjustment=mom2_adjustment,
    )
    result["generations"] = [
        {
            "prompt": prompt,
            "text": generate_text(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=32,
                decoding_strategy="greedy",
                patch_id=result["patch_id"],
            )["text"],
        }
        for prompt in evaluation_prompts
    ]
    return result
