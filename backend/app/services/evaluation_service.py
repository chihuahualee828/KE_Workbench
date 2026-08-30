from __future__ import annotations

from app.ml.inference.generate import generate_text
from app.services.patch_service import patch_service


class EvaluationService:
    def evaluate_patch(self, model_id: str, patch_id: str, prompts: list[str]) -> dict:
        patch = patch_service.get_patch(patch_id)
        eval_prompts = prompts or [
            edit["relation_prompt"].format(edit["subject"])
            for edit in patch.get("edits", [])
            if "{}" in edit.get("relation_prompt", "")
        ]
        if not eval_prompts:
            raise ValueError("Provide evaluation prompts or use a patch with format-able edit prompts.")

        target = patch["edits"][0]["new_object"] if patch.get("edits") else ""
        generations = []
        selected_patch_changed = 0
        current_model_changed = 0
        target_hits_clean = 0
        target_hits_selected_patch = 0
        target_hits_current_model = 0

        for prompt in eval_prompts[:10]:
            clean_model = generate_text(
                model_id,
                prompt,
                max_new_tokens=32,
                patch_id=None,
                use_active_patches=False,
            )["text"]
            selected_patch = generate_text(
                model_id,
                prompt,
                max_new_tokens=32,
                patch_id=patch_id,
                use_active_patches=False,
            )["text"]
            current_model = generate_text(
                model_id,
                prompt,
                max_new_tokens=32,
                patch_id=None,
                use_active_patches=True,
            )["text"]
            if clean_model != selected_patch:
                selected_patch_changed += 1
            if clean_model != current_model:
                current_model_changed += 1
            if target and target.lower() in clean_model.lower():
                target_hits_clean += 1
            if target and target.lower() in selected_patch.lower():
                target_hits_selected_patch += 1
            if target and target.lower() in current_model.lower():
                target_hits_current_model += 1
            generations.append(
                {
                    "prompt": prompt,
                    "clean_model": clean_model,
                    "selected_patch": selected_patch,
                    "current_model": current_model,
                }
            )

        sample_count = len(generations)
        metrics = {
            "target_string_match_clean_model": round(target_hits_clean / sample_count, 4),
            "target_string_match_selected_patch": round(target_hits_selected_patch / sample_count, 4),
            "target_string_match_current_model": round(target_hits_current_model / sample_count, 4),
            "selected_patch_changed_generation_rate": round(selected_patch_changed / sample_count, 4),
            "current_model_changed_generation_rate": round(current_model_changed / sample_count, 4),
        }
        return {
            "patch_id": patch_id,
            "model_id": model_id,
            "active_patch_ids": patch_service.active_patches_for_model(model_id),
            "metrics": metrics,
            "sample_generations": generations,
        }


evaluation_service = EvaluationService()
