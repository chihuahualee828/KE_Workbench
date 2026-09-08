from __future__ import annotations

from app.ml.inference.generate import generate_text
from app.services.patch_service import patch_service


class EvaluationService:
    def evaluate_easyedit_reliability(
        self,
        model_id: str,
        method: str,
        edits: list[dict],
        include_generations: bool = False,
        evaluation_modes: list[str] | None = None,
    ) -> dict:
        from app.ml.editing.easyedit_runtime import evaluate_easyedit_reliability
        from app.services.model_runtime_service import model_runtime_service

        if getattr(model_runtime_service, "uses_hard_worker", False):
            return model_runtime_service.execute(
                "easyedit_reliability",
                {
                    "model_id": model_id,
                    "method": method,
                    "edits": edits,
                    "include_generations": include_generations,
                },
                timeout=7200.0,
            )
        return evaluate_easyedit_reliability(
            model_id=model_id,
            method=method,
            edits=edits,
            include_generations=include_generations,
        )

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
        active_model_changed = 0
        target_hits_clean = 0
        target_hits_active_model = 0

        for prompt in eval_prompts[:10]:
            clean_model = generate_text(
                model_id,
                prompt,
                max_new_tokens=32,
                patch_id=None,
                use_active_patches=False,
            )["text"]
            active_model = generate_text(
                model_id,
                prompt,
                max_new_tokens=32,
                patch_id=None,
                use_active_patches=True,
            )["text"]
            if clean_model != active_model:
                active_model_changed += 1
            if target and target.lower() in clean_model.lower():
                target_hits_clean += 1
            if target and target.lower() in active_model.lower():
                target_hits_active_model += 1
            generations.append(
                {
                    "prompt": prompt,
                    "clean_model": clean_model,
                    "active_model": active_model,
                }
            )

        sample_count = len(generations)
        metrics = {
            "target_string_match_clean_model": round(target_hits_clean / sample_count, 4),
            "target_string_match_active_model": round(target_hits_active_model / sample_count, 4),
            "active_model_changed_generation_rate": round(active_model_changed / sample_count, 4),
        }
        return {
            "patch_id": patch_id,
            "model_id": model_id,
            "active_patch_ids": patch_service.active_patches_for_model(model_id),
            "metrics": metrics,
            "sample_generations": generations,
        }


evaluation_service = EvaluationService()
