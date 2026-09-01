from __future__ import annotations

from app.ml.editing.easyedit_runtime import run_easyedit
from app.ml.inference.generate import generate_text


def run_memit_edit(
    model_id: str,
    edits: list[dict],
    layers: list[int] | None,
    locality_inputs: dict | None,
    portability_inputs: dict | None,
    sequential_edit: bool,
    mom2_adjustment: bool,
) -> dict:
    normalized_edits = [
        edit.model_dump() if hasattr(edit, "model_dump") else dict(edit)
        for edit in edits[:100]
    ]
    rephrase_prompts = [
        prompt
        for edit in normalized_edits
        for prompt in edit.pop("rephrase_prompts", [])
    ]
    result = run_easyedit(
        method="memit",
        model_id=model_id,
        edits=normalized_edits,
        layers=layers,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=sequential_edit,
        mom2_adjustment=mom2_adjustment,
    )
    result["edited_facts"] = len(normalized_edits)
    result["generations"] = _generate_evaluation_outputs(
        model_id=model_id,
        patch_id=result["patch_id"],
        prompts=rephrase_prompts,
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
