"""Shared single-fact and cumulative-series execution for ROME and r-ROME."""
from __future__ import annotations

from app.ml.editing.easyedit_runtime import EasyEditSeriesContext, evaluate_post_edit_reliability, run_easyedit
from app.ml.editing.methods import RANK_ONE_METHODS
from app.ml.editing.patch_format import edit_to_patch_entry
from app.services.model_registry import model_registry


def run_rome_edit(
    method: str,
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
    source_metadata: dict | None = None,
    generate_evaluation: bool = True,
    patch_mode: str = "individual",
    series_context: EasyEditSeriesContext | None = None,
) -> dict:
    if method not in RANK_ONE_METHODS:
        raise ValueError(f"Unsupported rank-one editing method: {method}")
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            f"{method}_edit",
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
                "source_metadata": source_metadata,
                "generate_evaluation": generate_evaluation,
            },
            timeout=7200.0,
        )
    edit = edit_to_patch_entry(subject, relation_prompt, new_object)
    if ground_truth is not None:
        edit["ground_truth"] = ground_truth
    if source_metadata:
        edit["source_metadata"] = source_metadata
    all_rephrases = [*evaluation_prompts, *rephrase_prompts]

    model_info = model_registry.get(model_id)
    if target_layer is not None and not 0 <= target_layer < int(model_info["n_layers"]):
        raise ValueError(
            f"Layer {target_layer} is outside {model_id}'s valid range 0-{int(model_info['n_layers']) - 1}."
        )
    layer_source = "request" if target_layer is not None else "easyedit_hparams"
    edit_layers = [target_layer] if target_layer is not None else None

    result = run_easyedit(
        method=method,
        model_id=model_id,
        edits=[edit],
        layers=edit_layers,
        rephrase_prompts=all_rephrases,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=sequential_edit,
        layer_source=layer_source,
        mom2_adjustment=mom2_adjustment,
        patch_mode=patch_mode,
        series_context=series_context,
    )
    if generate_evaluation:
        result.update(evaluate_post_edit_reliability(model_id, method, [edit], edit_result=result))
    return result


def run_rome_sequential_edits(
    method: str,
    model_id: str,
    edits: list[dict],
    target_layer: int,
    evaluation_prompts: list[str],
    mom2_adjustment: bool,
    sequential_edit: bool = True,
    generate_evaluation: bool = True,
) -> dict:
    if method not in RANK_ONE_METHODS:
        raise ValueError(f"Unsupported rank-one editing method: {method}")
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            f"{method}_sequential_edits",
            {
                "model_id": model_id,
                "edits": edits,
                "target_layer": target_layer,
                "evaluation_prompts": evaluation_prompts,
                "mom2_adjustment": mom2_adjustment,
                "sequential_edit": sequential_edit,
                "generate_evaluation": generate_evaluation,
            },
            timeout=7200.0,
        )

    method_label = method.upper().replace("_", "-")
    if not edits:
        raise ValueError(f"{method_label} sequential series needs at least one fact.")

    series_context = EasyEditSeriesContext(continue_from_active_patches=sequential_edit)
    patch_results: list[dict] = []
    for index, raw_edit in enumerate(edits, start=1):
        edit = raw_edit.model_dump() if hasattr(raw_edit, "model_dump") else dict(raw_edit)
        subject = str(edit.get("subject") or "").strip()
        relation_prompt = str(edit.get("relation_prompt") or "").strip()
        new_object = str(edit.get("new_object") or "").strip()
        if not subject or not relation_prompt or not new_object:
            raise ValueError(
                f"{method_label} sequential series fact {index} needs a subject, relation prompt, and new object."
            )
        print(f"Applying {method_label} sequential edit {index}/{len(edits)}.")
        result = run_rome_edit(
            method=method,
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
            source_metadata=edit.get("source_metadata") if isinstance(edit.get("source_metadata"), dict) else None,
            generate_evaluation=False,
            patch_mode="series",
            series_context=series_context,
        )
        patch_results.append(result)
        print(f"Created {method_label} sequential edit {index}/{len(edits)}.")
        print(f"Updated cumulative patch {result['patch_id']}.")

    final_patch_id = patch_results[-1]["patch_id"]
    result = {
        "patch_id": final_patch_id,
        "patch_ids": [final_patch_id],
        "patch_count": 1,
        "completed_fact_count": len(patch_results),
        "sequential_edit": True,
        "metrics": [entry.get("metrics", []) for entry in patch_results],
    }
    if generate_evaluation:
        result.update(evaluate_post_edit_reliability(model_id, method, edits, edit_result=result))
    return result
