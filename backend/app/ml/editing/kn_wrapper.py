from __future__ import annotations

from app.ml.editing.easyedit_runtime import run_easyedit_kn
from app.ml.editing.patch_format import edit_to_patch_entry
from app.ml.inference.generate import generate_text


def run_kn_edit(
    model_id: str,
    subject: str,
    relation_prompt: str,
    new_object: str,
    ground_truth: str,
    evaluation_prompts: list[str],
    sequential_edit: bool,
) -> dict:
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "kn_edit",
            {
                "model_id": model_id,
                "subject": subject,
                "relation_prompt": relation_prompt,
                "new_object": new_object,
                "ground_truth": ground_truth,
                "evaluation_prompts": evaluation_prompts,
                "sequential_edit": sequential_edit,
            },
            timeout=7200.0,
        )
    edit = edit_to_patch_entry(subject, relation_prompt, new_object)
    edit["ground_truth"] = ground_truth
    result = run_easyedit_kn(
        model_id=model_id,
        edit=edit,
        sequential_edit=sequential_edit,
    )
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
