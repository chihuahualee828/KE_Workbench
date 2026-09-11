from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.db.schemas import ActiveModelEvaluationRequest, EvaluationRequest, JobResponse, ReliabilityEvaluationRequest
from app.services.evaluation_service import evaluation_service
from app.services.experiment_service import experiment_service
from app.services.model_runtime_service import ModelNotLoadedError, model_runtime_service
from app.services.patch_service import patch_service

router = APIRouter(tags=["evaluation"])
EVALUATION_PAGE_SIZE = 50
_active_model_evaluation_lock = threading.Lock()


@router.post("/evaluate", response_model=JobResponse)
def evaluate(request: EvaluationRequest) -> JobResponse:
    try:
        model_runtime_service.require_loaded_runtime_bundle(request.model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    experiment = experiment_service.create("evaluation", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    try:
        result = evaluation_service.evaluate_patch(**request.model_dump())
    except ValueError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    experiment_service.complete_job(job["id"], result)
    return JobResponse(
        job_id=job["id"],
        experiment_id=experiment["id"],
        status="completed",
        result=result,
    )


@router.post("/evaluate/reliability", response_model=JobResponse)
def evaluate_reliability(request: ReliabilityEvaluationRequest) -> JobResponse:
    try:
        model_runtime_service.require_loaded_runtime_bundle(request.model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    experiment = experiment_service.create("evaluation", request.model_id, request.model_dump())
    job = experiment_service.create_job(experiment["id"])
    try:
        result = evaluation_service.evaluate_easyedit_reliability(**request.model_dump())
    except ValueError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        experiment_service.fail_job(job["id"], str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    experiment_service.complete_job(job["id"], result)
    return JobResponse(
        job_id=job["id"],
        experiment_id=experiment["id"],
        status="completed",
        result=result,
    )

@router.get("/evaluate/active-facts")
def list_active_evaluation_facts(
    model_id: str,
) -> dict[str, Any]:
    facts = patch_service.active_evaluation_facts(model_id)
    active_patch_ids = patch_service.active_patches_for_model(model_id)
    method = ""
    if active_patch_ids:
        method = str(patch_service.get_patch(active_patch_ids[-1]).get("method") or "")
    return {
        "model_id": model_id,
        "active_patch_ids": active_patch_ids,
        "method": method,
        "total": len(facts),
        "facts": facts,
    }


@router.post("/evaluate/active-model", response_model=JobResponse)
def evaluate_active_model(request: ActiveModelEvaluationRequest) -> JobResponse:
    try:
        model_runtime_service.require_loaded_runtime_bundle(request.model_id)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    active_patch_ids = patch_service.active_patches_for_model(request.model_id)
    if not active_patch_ids:
        raise HTTPException(status_code=400, detail="Load at least one patch before evaluating its edit facts.")
    method = str(patch_service.get_patch(active_patch_ids[-1]).get("method") or "").lower()
    if method not in {"rome", "r_rome", "memit", "kn"}:
        raise HTTPException(status_code=400, detail="The active patch method does not support this evaluation.")

    modes = list(dict.fromkeys(request.evaluation_modes))
    if not modes:
        raise HTTPException(status_code=400, detail="Select at least one evaluation metric.")
    facts = patch_service.active_evaluation_facts(request.model_id)
    needs_edit_quality = any(mode in {"reliability", "generalization", "locality"} for mode in modes)
    if needs_edit_quality and not facts:
        raise HTTPException(status_code=400, detail="The active patches do not contain any evaluable edit facts.")

    edits = []
    if needs_edit_quality:
        for fact in facts:
            rephrase = (
                request.rephrase_overrides.get(str(fact["key"]), "").strip()
                or str(fact.get("rephrase_prompt") or "").strip()
                or str(fact["prompt"])
            )
            edit = {
                "subject": fact["subject"],
                "relation_prompt": fact["relation_prompt"],
                "new_object": fact["new_object"],
                "ground_truth": fact["ground_truth"] or None,
                "rephrase_prompts": [rephrase] if "generalization" in modes else [],
            }
            locality_prompt = str(fact.get("locality_prompt") or "").strip()
            locality_ground_truth = str(fact.get("locality_ground_truth") or "").strip()
            if "locality" in modes and locality_prompt and locality_ground_truth:
                edit["locality_prompt"] = locality_prompt
                edit["locality_ground_truth"] = locality_ground_truth
            edits.append(edit)

    experiment = experiment_service.create(
        "evaluation",
        request.model_id,
        {
            "source": "evaluation_page",
            "model_id": request.model_id,
            "method": method,
            "evaluation_modes": modes,
            "active_patch_ids": active_patch_ids,
            "rephrase_override_count": len(request.rephrase_overrides),
        },
    )
    job = experiment_service.create_job(experiment["id"])
    _start_active_model_evaluation(
        job_id=job["id"],
        model_id=request.model_id,
        method=method,
        active_patch_ids=active_patch_ids,
        facts=facts,
        edits=edits,
        modes=modes,
        include_generations=request.include_generations,
        needs_edit_quality=needs_edit_quality,
    )
    return JobResponse(
        job_id=job["id"],
        experiment_id=experiment["id"],
        status="running",
        logs=job.get("logs", []),
        log_count=int(job.get("log_count", len(job.get("logs", [])))),
    )


def _start_active_model_evaluation(
    *,
    job_id: str,
    model_id: str,
    method: str,
    active_patch_ids: list[str],
    facts: list[dict[str, Any]],
    edits: list[dict[str, Any]],
    modes: list[str],
    include_generations: bool,
    needs_edit_quality: bool,
) -> None:
    threading.Thread(
        target=_run_active_model_evaluation,
        kwargs={
            "job_id": job_id,
            "model_id": model_id,
            "method": method,
            "active_patch_ids": active_patch_ids,
            "facts": facts,
            "edits": edits,
            "modes": modes,
            "include_generations": include_generations,
            "needs_edit_quality": needs_edit_quality,
        },
        daemon=True,
    ).start()


def _run_active_model_evaluation(
    *,
    job_id: str,
    model_id: str,
    method: str,
    active_patch_ids: list[str],
    facts: list[dict[str, Any]],
    edits: list[dict[str, Any]],
    modes: list[str],
    include_generations: bool,
    needs_edit_quality: bool,
) -> None:
    try:
        with _active_model_evaluation_lock:
            job = experiment_service.get_job(job_id)
            if job is None or job.get("status") != "running":
                return
            total_facts = len(edits) if needs_edit_quality else len(facts)
            experiment_service.append_job_log(
                job_id,
                "Starting evaluation",
                progress_current=0,
                progress_total=max(1, total_facts),
            )
            experiment_service.set_job_progress(job_id, 0.05)

            def update_progress(current: int, total: int) -> None:
                experiment_service.set_job_progress(
                    job_id,
                    current / total if total else 0.05,
                    progress_current=current,
                    progress_total=total,
                )

            raw_result = (
                evaluation_service.evaluate_easyedit_quality(
                    model_id=model_id,
                    method=method,
                    edits=edits,
                    include_generations=include_generations,
                    on_progress=update_progress,
                    cancel_job_id=job_id,
                )
                if needs_edit_quality
                else {
                    "model_id": model_id,
                    "method": method,
                    "active_patch_ids": active_patch_ids,
                    "cases": [],
                }
            )
            result, overview, pages = _build_paged_evaluation_result(raw_result, facts, modes)
            experiment_service.complete_paged_evaluation_job(job_id, result, overview, pages)
    except Exception as exc:
        experiment_service.fail_job(job_id, str(exc))


@router.get("/evaluate/results/{experiment_id}")
def get_evaluation_result_overview(experiment_id: str) -> dict[str, Any]:
    overview = experiment_service.get_evaluation_overview(experiment_id)
    if overview is None:
        raise HTTPException(status_code=404, detail="Paged evaluation result not found")
    return overview


@router.get("/evaluate/results/{experiment_id}/page")
def get_evaluation_result_page(
    experiment_id: str,
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    page = experiment_service.get_evaluation_page(experiment_id, offset)
    if page is None:
        raise HTTPException(status_code=404, detail="Evaluation result page not found")
    return page


@router.get("/evaluate/results/{experiment_id}/raw")
def get_evaluation_raw_result(experiment_id: str) -> dict[str, Any]:
    result = experiment_service.get_evaluation_raw_result(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    return result


def _build_paged_evaluation_result(
    raw_result: dict[str, Any],
    facts: list[dict[str, Any]],
    modes: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    reliability_generations = raw_result.get("sample_generations") if isinstance(raw_result.get("sample_generations"), list) else []
    generalization_generations = raw_result.get("sample_rephrase_generations") if isinstance(raw_result.get("sample_rephrase_generations"), list) else []
    locality_generations = raw_result.get("sample_locality_generations") if isinstance(raw_result.get("sample_locality_generations"), list) else []
    cases = raw_result.get("cases") if isinstance(raw_result.get("cases"), list) else []
    metrics: dict[str, Any] = {}
    if "reliability" in modes:
        metrics["reliability_rewrite_acc"] = raw_result.get("mean_reliability")
    if "generalization" in modes:
        metrics["generalization_rephrase_acc"] = raw_result.get("mean_generalization")
    if "locality" in modes:
        metrics["locality_preservation_acc"] = raw_result.get("mean_locality")
    if "portability" in modes:
        metrics["portability_score"] = 0

    items = []
    for index, fact in enumerate(facts):
        case = cases[index] if index < len(cases) and isinstance(cases[index], dict) else {}
        reliability_generation = reliability_generations[index] if index < len(reliability_generations) and isinstance(reliability_generations[index], dict) else None
        generalization_generation = generalization_generations[index] if index < len(generalization_generations) and isinstance(generalization_generations[index], dict) else None
        locality_generation = locality_generations[index] if index < len(locality_generations) and isinstance(locality_generations[index], dict) else None
        items.append(
            {
                "index": index,
                "fact": fact,
                "reliability_generation": reliability_generation,
                "generalization_generation": generalization_generation,
                "locality_generation": locality_generation,
                "reliability_score": case.get("reliability_score"),
                "generalization_score": case.get("generalization_score"),
                "locality_score": case.get("locality_score"),
            }
        )

    overview = {
        "model_id": raw_result.get("model_id"),
        "method": raw_result.get("method"),
        "active_patch_ids": raw_result.get("active_patch_ids", []),
        "metrics": metrics,
        "evaluation_modes": modes,
        "total": len(items),
        "available_generation_metrics": [
            metric for metric, generations in (
                ("reliability", reliability_generations),
                ("generalization", generalization_generations),
                ("locality", locality_generations),
            ) if metric in modes and any(isinstance(generation, dict) for generation in generations)
        ],
    }
    pages = [
        {
            "offset": offset,
            "limit": EVALUATION_PAGE_SIZE,
            "total": len(items),
            "items": items[offset:offset + EVALUATION_PAGE_SIZE],
        }
        for offset in range(0, len(items), EVALUATION_PAGE_SIZE)
    ]
    result = {
        **raw_result,
        **overview,
        "evaluation_facts": facts,
        "evaluation_items": items,
    }
    return result, overview, pages


# Active edit facts are compact metadata for the browser-side IndexedDB cache.
# Evaluation generations and metric payloads are intentionally served through
# separate result pages, preventing large generation text from being included
# in this active-fact snapshot.
