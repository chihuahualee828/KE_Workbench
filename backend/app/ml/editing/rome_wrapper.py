from __future__ import annotations

from app.ml.editing.easyedit_runtime import run_easyedit
from app.ml.editing.patch_format import edit_to_patch_entry
from app.ml.inference.generate import generate_text
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
    mom2_adjustment: bool,
    generate_evaluation: bool = True,
) -> dict:
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "rome_edit",
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
                "generate_evaluation": generate_evaluation,
            },
            timeout=7200.0,
        )
    edit = edit_to_patch_entry(subject, relation_prompt, new_object)
    if ground_truth is not None:
        edit["ground_truth"] = ground_truth
    all_rephrases = [*evaluation_prompts, *rephrase_prompts]

    model_info = model_registry.get(model_id)
    if target_layer is not None and not 0 <= target_layer < int(model_info["n_layers"]):
        raise ValueError(
            f"Layer {target_layer} is outside {model_id}'s valid range 0-{int(model_info['n_layers']) - 1}."
        )
    layer_source = "request" if target_layer is not None else "easyedit_hparams"
    edit_layers = [target_layer] if target_layer is not None else None

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
        mom2_adjustment=mom2_adjustment,
    )
    if generate_evaluation:
        result["generations"] = _generate_evaluation_outputs(
            model_id=model_id,
            patch_id=result["patch_id"],
            prompts=evaluation_prompts,
        )
    return result


def run_rome_sequential_edits(
    model_id: str,
    edits: list[dict],
    target_layer: int,
    evaluation_prompts: list[str],
    mom2_adjustment: bool,
) -> dict:
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "rome_sequential_edits",
            {
                "model_id": model_id,
                "edits": edits,
                "target_layer": target_layer,
                "evaluation_prompts": evaluation_prompts,
                "mom2_adjustment": mom2_adjustment,
            },
            timeout=7200.0,
        )

    if not edits:
        raise ValueError("ROME sequential series needs at least one fact.")

    patch_results: list[dict] = []
    for index, raw_edit in enumerate(edits, start=1):
        edit = raw_edit.model_dump() if hasattr(raw_edit, "model_dump") else dict(raw_edit)
        subject = str(edit.get("subject") or "").strip()
        relation_prompt = str(edit.get("relation_prompt") or "").strip()
        new_object = str(edit.get("new_object") or "").strip()
        if not subject or not relation_prompt or not new_object:
            raise ValueError(
                f"ROME sequential series fact {index} needs a subject, relation prompt, and new object."
            )
        print(f"Applying ROME sequential edit {index}/{len(edits)}.")
        result = run_rome_edit(
            model_id=model_id,
            subject=subject,
            relation_prompt=relation_prompt,
            new_object=new_object,
            ground_truth=str(edit["ground_truth"]).strip() if edit.get("ground_truth") else None,
            target_layer=target_layer,
            evaluation_prompts=[],
            rephrase_prompts=[str(prompt) for prompt in edit.get("rephrase_prompts", [])],
            locality_inputs=None,
            portability_inputs=None,
            sequential_edit=True,
            mom2_adjustment=mom2_adjustment,
            generate_evaluation=False,
        )
        patch_results.append(result)
        print(f"Created patch {result['patch_id']} for ROME sequential edit {index}/{len(edits)}.")

    final_patch_id = patch_results[-1]["patch_id"]
    return {
        "patch_id": final_patch_id,
        "patch_ids": [result["patch_id"] for result in patch_results],
        "patch_count": len(patch_results),
        "sequential_edit": True,
        "generations": _generate_evaluation_outputs(
            model_id=model_id,
            patch_id=final_patch_id,
            prompts=evaluation_prompts,
        ),
    }


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
