from __future__ import annotations

from app.ml.editing.easyedit_runtime import run_easyedit
from app.ml.inference.generate import generate_text


def run_memit_edit(
    model_id: str,
    edits: list[dict],
    layers: list[int] | None,
    evaluation_prompts: list[str],
    locality_inputs: dict | None,
    portability_inputs: dict | None,
    sequential_edit: bool,
    mom2_adjustment: bool,
) -> dict:
    normalized_edits = [
        edit.model_dump() if hasattr(edit, "model_dump") else dict(edit)
        for edit in edits[:100]
    ]
    rephrase_prompts = _single_rephrase_per_edit(normalized_edits)
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
        prompts=evaluation_prompts,
    )
    return result


def _single_rephrase_per_edit(edits: list[dict]) -> list[str] | None:
    """EasyEdit aligns rephrase prompts positionally with edit requests."""
    values: list[str] = []
    for edit in edits:
        prompts = edit.pop("rephrase_prompts", [])
        if not prompts:
            return None
        values.append(str(prompts[0]))
    return values


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
