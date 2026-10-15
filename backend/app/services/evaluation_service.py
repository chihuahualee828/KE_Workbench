from __future__ import annotations

from collections.abc import Callable

from app.ml.inference.generate import generate_text
from app.services.patch_service import patch_service


class EvaluationService:
    def evaluate_easyedit_quality(
        self,
        model_id: str,
        method: str,
        edits: list[dict],
        include_generations: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_job_id: str | None = None,
        fluency_modes: list[str] | None = None,
        evaluation_modes: list[str] | None = None,
        evaluation_protocols: dict[str, str] | None = None,
        on_baseline_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        from app.ml.editing.easyedit_runtime import evaluate_easyedit_quality
        from app.services.model_runtime_service import model_runtime_service

        def worker_progress(update: dict) -> None:
            callback = on_baseline_progress if update.get("phase") == "baseline" else on_progress
            if callback is not None:
                callback(int(update["current"]), int(update["total"]))

        if getattr(model_runtime_service, "uses_hard_worker", False):
            return model_runtime_service.execute(
                "easyedit_edit_quality",
                {
                    "model_id": model_id,
                    "method": method,
                    "edits": edits,
                    "include_generations": include_generations,
                    "cancel_job_id": cancel_job_id,
                    "fluency_modes": fluency_modes or [],
                    **({"evaluation_modes": evaluation_modes} if evaluation_modes else {}),
                    **({"evaluation_protocols": evaluation_protocols} if evaluation_protocols else {}),
                },
                timeout=7200.0,
                progress_callback=worker_progress if on_progress is not None or on_baseline_progress is not None else None,
            )
        return evaluate_easyedit_quality(
            model_id=model_id,
            method=method,
            edits=edits,
            include_generations=include_generations,
            progress_callback=on_progress,
            cancel_job_id=cancel_job_id,
            fluency_modes=fluency_modes,
            **({"evaluation_modes": evaluation_modes} if evaluation_modes else {}),
            **({"evaluation_protocols": evaluation_protocols} if evaluation_protocols else {}),
            **({"baseline_progress_callback": on_baseline_progress} if on_baseline_progress is not None else {}),
        )

    def evaluate_easyedit_reliability(
        self,
        model_id: str,
        method: str,
        edits: list[dict],
        include_generations: bool = False,
        evaluation_modes: list[str] | None = None,
        evaluation_protocols: dict[str, str] | None = None,
    ) -> dict:
        """Compatibility entry point with optional per-metric protocols."""
        return self.evaluate_easyedit_quality(
            model_id=model_id,
            method=method,
            edits=edits,
            include_generations=include_generations,
            **({"evaluation_modes": evaluation_modes} if evaluation_modes else {}),
            **({"evaluation_protocols": evaluation_protocols} if evaluation_protocols else {}),
        )

    def evaluate_edit_reliability_test_pair(
        self,
        model_id: str,
        method: str,
        edit: dict,
        include_generation: bool = True,
        on_generation: Callable[[dict], None] | None = None,
        cancel_job_id: str | None = None,
        evaluation_modes: list[str] | None = None,
        evaluation_protocols: dict[str, str] | None = None,
    ) -> dict:
        """Generate and score one fact, preserving the edit-page test order."""
        from app.ml.editing.easyedit_runtime import build_prompt

        subject = str(edit.get("subject") or "").strip()
        relation_prompt = str(edit.get("relation_prompt") or "").strip()
        target_new = str(edit.get("new_object") or "").strip()
        if not subject or not relation_prompt or not target_new:
            raise ValueError("Each reliability test fact needs a subject, relation prompt, and new object.")

        prompt = build_prompt(subject, relation_prompt)
        custom_generation_protocol = any(
            protocol != "traditional"
            for metric, protocol in (evaluation_protocols or {}).items()
            if not evaluation_modes or metric in evaluation_modes
        )
        generation = None
        if include_generation and not custom_generation_protocol:
            generation = {
                "prompt": prompt,
                "active_model": generate_text(
                    model_id,
                    prompt,
                    max_new_tokens=32,
                    patch_id=None,
                    use_active_patches=True,
                )["text"],
            }
            if on_generation is not None:
                on_generation(generation)
        result = self.evaluate_easyedit_quality(
            model_id=model_id,
            method=method,
            edits=[edit],
            include_generations=include_generation if custom_generation_protocol else False,
            cancel_job_id=cancel_job_id,
            **({"evaluation_modes": evaluation_modes} if evaluation_modes else {}),
            **({"evaluation_protocols": evaluation_protocols} if evaluation_protocols else {}),
        )
        cases = result.get("cases")
        if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
            raise RuntimeError("EasyEdit reliability evaluation returned no fact result.")
        if custom_generation_protocol and include_generation:
            samples = result.get("sample_generations")
            if isinstance(samples, list) and samples and isinstance(samples[0], dict):
                generation = samples[0]
                if on_generation is not None:
                    on_generation(generation)
        return {
            "case": cases[0],
            "generation": generation,
            "active_patch_ids": result.get("active_patch_ids", []),
        }

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
