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
RELIABILITY_TEST_PAGE_SIZE = 25
_active_model_evaluation_lock = threading.Lock()
_reliability_evaluation_lock = threading.Lock()


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

    experiment = experiment_service.create(
        "evaluation",
        request.model_id,
        {
            "source": "edit_page",
            "model_id": request.model_id,
            "method": request.method,
            "edit_count": len(request.edits),
            "include_generations": request.include_generations,
        },
    )
    job = experiment_service.create_job(experiment["id"])
    _start_reliability_evaluation(
        job_id=job["id"],
        model_id=request.model_id,
        method=request.method,
        edits=[edit.model_dump() for edit in request.edits],
        include_generations=request.include_generations,
    )
    return JobResponse(
        job_id=job["id"],
        experiment_id=experiment["id"],
        status="running",
        logs=job.get("logs", []),
        log_count=int(job.get("log_count", len(job.get("logs", [])))),
    )


@router.get("/evaluate/reliability/jobs/{job_id}", response_model=JobResponse)
def get_reliability_evaluation_job(job_id: str) -> JobResponse:
    job = experiment_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Reliability evaluation job not found")
    return _reliability_evaluation_job_response(job)


@router.get("/evaluate/reliability/jobs/{job_id}/page")
def get_reliability_evaluation_page(
    job_id: str,
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    page = experiment_service.get_job_result_page(job_id, offset)
    if page is None:
        raise HTTPException(status_code=404, detail="Reliability evaluation job not found")
    return page


def _start_reliability_evaluation(
    *,
    job_id: str,
    model_id: str,
    method: str,
    edits: list[dict[str, Any]],
    include_generations: bool,
) -> None:
    threading.Thread(
        target=_run_reliability_evaluation,
        kwargs={
            "job_id": job_id,
            "model_id": model_id,
            "method": method,
            "edits": edits,
            "include_generations": include_generations,
        },
        daemon=True,
    ).start()


def _run_reliability_evaluation(
    *,
    job_id: str,
    model_id: str,
    method: str,
    edits: list[dict[str, Any]],
    include_generations: bool,
) -> None:
    try:
        with _reliability_evaluation_lock:
            job = experiment_service.get_job(job_id)
            if job is None or job.get("status") != "running":
                return
            total = len(edits)
            completed_count = 0
            reliability_total = 0.0
            reliability_count = 0
            generalization_total = 0.0
            generalization_count = 0
            active_patch_ids: list[str] = []

            def summary() -> dict[str, Any]:
                return _reliability_result(
                    model_id=model_id,
                    method=method,
                    active_patch_ids=active_patch_ids,
                    completed_count=completed_count,
                    total_count=total,
                    reliability_total=reliability_total,
                    reliability_count=reliability_count,
                    generalization_total=generalization_total,
                    generalization_count=generalization_count,
                )

            experiment_service.append_job_log(job_id, f"Testing {total} edit fact{'s' if total != 1 else ''}")
            for index, edit in enumerate(edits, start=1):
                current_job = experiment_service.get_job(job_id)
                if current_job is None or current_job.get("status") != "running":
                    return
                experiment_service.append_job_log(job_id, f"Generating and evaluating fact {index}/{total}")

                def persist_generation(generation: dict[str, Any]) -> None:
                    experiment_service.upsert_job_result_item(
                        job_id,
                        index - 1,
                        {"generation": generation},
                        RELIABILITY_TEST_PAGE_SIZE,
                    )
                    experiment_service.set_job_partial_result(job_id, summary())
                    experiment_service.append_job_log(job_id, f"Generated fact {index}/{total}; computing rewrite accuracy")

                pair = evaluation_service.evaluate_edit_reliability_test_pair(
                    model_id=model_id,
                    method=method,
                    edit=edit,
                    include_generation=include_generations,
                    on_generation=persist_generation if include_generations else None,
                    cancel_job_id=job_id,
                )
                current_job = experiment_service.get_job(job_id)
                if current_job is None or current_job.get("status") != "running":
                    return
                case = pair["case"]
                score = case.get("reliability_score")
                if isinstance(score, (int, float)):
                    reliability_total += float(score)
                    reliability_count += 1
                generalization_score = case.get("generalization_score")
                if isinstance(generalization_score, (int, float)):
                    generalization_total += float(generalization_score)
                    generalization_count += 1
                patch_ids = pair.get("active_patch_ids")
                active_patch_ids = [str(patch_id) for patch_id in patch_ids] if isinstance(patch_ids, list) else []
                completed_count = index
                experiment_service.upsert_job_result_item(
                    job_id,
                    index - 1,
                    {"case": case},
                    RELIABILITY_TEST_PAGE_SIZE,
                )
                partial_result = summary()
                experiment_service.set_job_partial_result(job_id, partial_result)
                experiment_service.append_job_log(
                    job_id,
                    f"Completed fact {index}/{total}",
                    progress=index / total,
                    progress_current=index,
                    progress_total=total,
                )
            experiment_service.complete_job(job_id, partial_result)
    except Exception as exc:
        experiment_service.fail_job(job_id, str(exc))


def _reliability_result(
    *,
    model_id: str,
    method: str,
    active_patch_ids: list[str],
    completed_count: int,
    total_count: int,
    reliability_total: float,
    reliability_count: int,
    generalization_total: float,
    generalization_count: int,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "method": method,
        "active_patch_ids": active_patch_ids,
        "metric_name": "rewrite_acc",
        "mean_reliability": _mean_score(reliability_total, reliability_count),
        "mean_generalization": _mean_score(generalization_total, generalization_count),
        "completed_fact_count": completed_count,
        "total_fact_count": total_count,
    }


def _mean_score(total: float, count: int) -> float | None:
    return round(total / count, 4) if count else None


def _reliability_evaluation_job_response(job: dict[str, Any]) -> JobResponse:
    result = job.get("partial_result")
    if job.get("status") == "completed":
        details = experiment_service.get_experiment_details(str(job["experiment_id"]))
        if details and isinstance(details.get("result"), dict):
            result = details["result"]
    logs = job.get("logs", [])
    return JobResponse(
        job_id=str(job["id"]),
        experiment_id=str(job["experiment_id"]),
        status=str(job.get("status") or "running"),
        progress=float(job.get("progress") or 0.0),
        progress_current=job.get("progress_current"),
        progress_total=job.get("progress_total"),
        logs=logs if isinstance(logs, list) else [],
        log_count=int(job.get("log_count", len(logs) if isinstance(logs, list) else 0)),
        result_item_count=int(job.get("result_item_count") or 0),
        result_page_size=int(job["result_page_size"]) if isinstance(job.get("result_page_size"), int) else None,
        result=result if isinstance(result, dict) else None,
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
