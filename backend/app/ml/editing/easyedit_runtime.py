from __future__ import annotations

import importlib
from contextlib import contextmanager
import json
import os
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from typing import Any, Literal

import yaml

from app.config import settings
from app.ml.editing.edit_cancellation import check_edit_cancellation, current_edit, memit_cancellation_hook
from app.ml.editing.mps_compat import configure_mps_hparams, editing_precision, mps_fp32
from app.services.model_registry import model_registry
from app.services.model_runtime_service import model_runtime_service
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


class EasyEditRuntimeError(RuntimeError):
    pass


class EasyEditUnsupportedError(ValueError):
    pass


@dataclass
class EasyEditSeriesContext:
    """State shared by the individual EasyEdit calls in one fact series.

    The baseline is captured from the model state before the first fact. Each
    later step writes a delta against that same baseline, so one patch can
    replay the complete series on top of its original parent patch stack.
    """

    patch_id: str | None = None
    initial_baseline_weights: dict[str, Any] | None = None
    parent_patch_ids: list[str] | None = None
    continue_from_active_patches: bool = True
    runtime_stack_initialized: bool = False
    completed_edits: list[dict[str, Any]] = field(default_factory=list)
    metrics_by_edit: list[Any] = field(default_factory=list)
    patch_created: bool = False


METHODS = {
    "rome": {
        "hparams_class": "ROMEHyperParams",
        "hparams_key": "rome_hparams",
        "support_key": "supports_rome",
        "requires_subject": True,
    },
    "r_rome": {
        "hparams_class": "R_ROMEHyperParams",
        "hparams_key": "r_rome_hparams",
        "support_key": "supports_r_rome",
        "requires_subject": True,
    },
    "memit": {
        "hparams_class": "MEMITHyperParams",
        "hparams_key": "memit_hparams",
        "support_key": "supports_memit",
        "requires_subject": True,
    },
    "ft": {
        "hparams_class": "FTHyperParams",
        "hparams_key": "ft_hparams",
        "support_key": "supports_ft",
    },
    "lora": {
        "hparams_class": "LoRAHyperParams",
        "hparams_key": "lora_hparams",
        "support_key": "supports_lora",
    },
    "kn": {
        "hparams_class": "KNHyperParams",
        "hparams_key": "kn_hparams",
        "support_key": "supports_kn",
    },
}

MOM2_DTYPES = ("float16", "float32", "float64")
_mom2_precision_lock = threading.Lock()
_mom2_covariance_profiles: dict[tuple[str, tuple[str, str]], tuple[Any, ...]] = {}


def build_prompt(subject: str, relation_prompt: str) -> str:
    if "{}" in relation_prompt:
        return relation_prompt.format(subject)
    return f"{subject} {relation_prompt}".strip()


def get_easyedit_default_layers(model_id: str, method: str) -> dict[str, Any]:
    method_key = method.lower()
    spec = METHODS.get(method_key)
    if spec is None:
        raise EasyEditUnsupportedError(f"EasyEdit method is not configured: {method}")

    model = model_registry.get(model_id)
    if not model.get(spec["support_key"], False):
        raise EasyEditUnsupportedError(f"{model_id} is not configured for {method_key.upper()}.")

    hparams_path = _resolve_hparams_path(model, spec["hparams_key"])
    try:
        payload = yaml.safe_load(hparams_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise EasyEditRuntimeError(f"Invalid EasyEdit hparams YAML: {hparams_path}") from exc

    configured_layers = payload.get("layers", payload.get("layer"))
    if isinstance(configured_layers, list):
        layers = [int(layer) for layer in configured_layers]
    elif configured_layers is not None:
        layers = [int(configured_layers)]
    else:
        layers = []
    if not layers and method_key != "lora":
        raise EasyEditRuntimeError(f"EasyEdit hparams has no configured edit layers: {hparams_path}")
    return {"layers": layers, "hparams_path": str(hparams_path)}


def run_easyedit(
    method: str,
    model_id: str,
    edits: list[dict[str, Any]],
    layers: list[int] | None,
    rephrase_prompts: list[str] | None = None,
    locality_inputs: dict[str, dict[str, list[str]]] | None = None,
    portability_inputs: dict[str, dict[str, list[str]]] | None = None,
    sequential_edit: bool = False,
    layer_source: str | None = None,
    mom2_adjustment: bool | None = None,
    patch_mode: Literal["individual", "series"] = "individual",
    series_context: EasyEditSeriesContext | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    if patch_mode not in {"individual", "series"}:
        raise ValueError(f"Unsupported EasyEdit patch mode: {patch_mode}")
    if patch_mode == "series" and series_context is None:
        raise ValueError("EasyEdit series mode requires a shared series context.")

    method_key = method.lower()
    if method_key == "memit":
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("MEMIT batch size must be a positive integer.")
        if not edits:
            raise ValueError("Provide at least one MEMIT edit fact.")
    spec = METHODS.get(method_key)
    if spec is None:
        raise EasyEditUnsupportedError(f"EasyEdit method is not configured: {method}")

    model = model_registry.get(model_id)
    print(f"Running EasyEdit with model {model_id} ({model['hf_id']}) and method {method_key.upper()}...")
    if not model.get(spec["support_key"], False):
        raise EasyEditUnsupportedError(
            f"{model_id} is not configured for {method_key.upper()} in this workbench. "
            "Select an EasyEdit-compatible model such as gpt2-xl."
        )

    BaseEditor, hparams_cls = _load_easyedit(spec["hparams_class"])
    hparams_path = _resolve_hparams_path(model, spec["hparams_key"])
    hparams = hparams_cls.from_hparams(str(hparams_path))
    if method_key == "memit":
        hparams.batch_size = batch_size
    print(f"Loaded EasyEdit hparams from {hparams_path}.")

    _override_model_name(hparams, model)
    _override_layers(hparams, layers)
    _override_mom2_adjustment(hparams, mom2_adjustment)
    _configure_stats_dir(hparams)
    easyedit_model_name = hparams.model_name
    effective_layers = _hparams_layers(hparams)
    print(f"{method_key.upper()} hparams: {hparams}")

    prompts: list[str] = []
    subjects: list[str] = []
    target_new: list[str] = []
    ground_truth: list[Any] = []
    for edit in edits:
        subject = edit["subject"]
        prompts.append(build_prompt(subject, edit["relation_prompt"]))
        subjects.append(subject)
        target_new.append(edit["new_object"])
        ground_truth.append(edit.get("ground_truth"))
    if not any(ground_truth):
        ground_truth = None
    editor = None
    edited_model = None
    baseline_weights = None
    metrics = None
    used_loaded_runtime = False
    weights_restored = False
    mom2_model_identity = None

    with _easyedit_runtime_access(model_id) as runtime_bundle:
        used_loaded_runtime = True
        check_edit_cancellation()
        transaction = current_edit.get()
        if transaction is not None:
            names = [f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in _hparams_layers(hparams)]
            transaction.capture(runtime_bundle["model"], model_id, names)
        if patch_mode == "series" and series_context is not None:
            if not series_context.runtime_stack_initialized:
                if not series_context.continue_from_active_patches:
                    # This controls the workbench patch baseline only. EasyEdit still
                    # receives sequential_edit=True so every fact in the series builds
                    # on the preceding fact within this one job.
                    patch_service.deactivate_runtime_patches_for_model(model_id)
                    patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
                    if transaction is not None:
                        transaction.notify_patch_state_changed()
                series_context.runtime_stack_initialized = True
            active_patch_ids = patch_service.active_patches_for_model(model_id)
            if series_context.parent_patch_ids is None:
                series_context.parent_patch_ids = list(active_patch_ids)
        elif not sequential_edit:
            # Starting a new non-continuing edit must update both the runtime weights
            # and persisted active-stack state. Otherwise a failed edit could leave the
            # UI claiming old patches are active while the model has been reset.
            patch_service.deactivate_runtime_patches_for_model(model_id)
            patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
            if transaction is not None:
                transaction.notify_patch_state_changed()
            active_patch_ids = patch_service.active_patches_for_model(model_id)
        else:
            active_patch_ids = patch_service.active_patches_for_model(model_id)
        _configure_loaded_tokenizer(runtime_bundle["tokenizer"], str(easyedit_model_name), method_key)
        if getattr(hparams, "mom2_adjustment", False):
            _configure_mom2_precision(hparams, model_id, method_key, str(runtime_bundle.get("device") or "cpu"))
        _override_easyedit_device(hparams, runtime_bundle.get("device"))
        hparams.model_name = (runtime_bundle["model"], runtime_bundle["tokenizer"])
        if getattr(hparams, "mom2_adjustment", False):
            mom2_model_identity = _set_loaded_model_mom2_cache_identity(runtime_bundle["model"], model)

        try:
            editor = BaseEditor.from_hparams(hparams)
            if used_loaded_runtime:
                hparams.model_name = easyedit_model_name
                editor.model_name = easyedit_model_name
                editor.hparams.model_name = easyedit_model_name

            baseline_weights = _snapshot_rewrite_weights(editor.model, hparams)
            if not baseline_weights:
                raise EasyEditRuntimeError(
                    "Unable to snapshot target rewrite weights before editing. "
                    "Check the EasyEdit hparams rewrite_module_tmp/layers for this model."
                )
            if patch_mode == "series" and series_context is not None and series_context.initial_baseline_weights is None:
                series_context.initial_baseline_weights = baseline_weights

            # EasyEdit restores singleton edits before returning when sequential_edit=False.
            # We keep the edited weights briefly, save our delta, then restore the baseline
            # ourselves so workbench patch state remains explicit and replayable.
            easyedit_sequential = True
            edit_args = {
                "prompts": prompts,
                "ground_truth": ground_truth,
                "target_new": target_new,
                "rephrase_prompts": rephrase_prompts or None,
                "locality_inputs": locality_inputs,
                "portability_inputs": portability_inputs,
                "sequential_edit": easyedit_sequential,
                "keep_original_weight": True,
            }
            if spec.get("requires_subject", False):
                edit_args["subject"] = subjects
            if getattr(hparams, "mom2_adjustment", False):
                _refresh_mom2_covariance_cache(method_key, hparams, editor.model, runtime_bundle.get("device") or "cpu")
            # Keep this stable marker in the captured job log. It separates
            # setup (hparams, snapshots, and runtime wiring) from EasyEdit's
            # actual update algorithm without modifying EasyEdit itself.
            print("Starting EasyEdit algorithm.")
            check_edit_cancellation()
            with memit_cancellation_hook(editor.model, enabled=method_key == "memit"):
                if method_key == "memit":
                    # Always accumulate chunks within this job, independently of
                    # whether the job started from clean or already patched weights.
                    # One pre-job snapshot/delta keeps cancellation all-or-nothing.
                    print(f"MEMIT batch size: {batch_size}; facts: {len(edits)}.")
                    metrics, edited_model, _ = editor.batch_edit(**edit_args)
                else:
                    metrics, edited_model, _ = editor.edit(**edit_args)
            check_edit_cancellation()
            patch_id = (
                series_context.patch_id
                if patch_mode == "series" and series_context is not None and series_context.patch_id
                else patch_service.new_patch_id()
            )
            if patch_mode == "series" and series_context is not None:
                series_context.patch_id = patch_id
            if transaction is not None:
                transaction.protect_patch(patch_id)
            delta_baseline_weights = (
                series_context.initial_baseline_weights
                if patch_mode == "series" and series_context is not None
                else baseline_weights
            )
            tensor_delta_paths = _save_weight_deltas(edited_model, delta_baseline_weights, patch_id)
            if not tensor_delta_paths:
                raise EasyEditRuntimeError(
                    "EasyEdit completed, but no changed tensor weights were captured. "
                    "No patch was created because generation/evaluation would stay unchanged."
                )
            json_metrics = _jsonable(metrics)
            if patch_mode == "series" and series_context is not None:
                series_context.completed_edits.extend(edits)
                series_context.metrics_by_edit.append(json_metrics)
                series_metadata = {
                    "easyedit_hparams": str(hparams_path),
                    "hf_model": model["hf_id"],
                    "sequential_edit": True,
                    "series_edit": True,
                    "series_fact_count": len(series_context.completed_edits),
                    "series_metrics": series_context.metrics_by_edit,
                    "parent_patch_ids": series_context.parent_patch_ids or [],
                    "layer_source": layer_source or ("request" if layers else "easyedit_hparams"),
                    "mom2_adjustment": bool(getattr(hparams, "mom2_adjustment", False)),
                    "used_loaded_runtime": used_loaded_runtime,
                }
                if series_context.patch_created:
                    patch = patch_service.update_patch(
                        patch_id,
                        edits=series_context.completed_edits,
                        modified_layers=effective_layers,
                        metrics=json_metrics,
                        delta_paths=tensor_delta_paths,
                        extra_metadata=series_metadata,
                    )
                else:
                    patch = patch_service.create_patch(
                        model_id=model_id,
                        method=method_key,
                        edits=series_context.completed_edits,
                        modified_layers=effective_layers,
                        metrics=json_metrics,
                        delta_paths=tensor_delta_paths,
                        runtime="easyedit",
                        extra_metadata=series_metadata,
                        patch_id=patch_id,
                    )
                    series_context.patch_created = True
                    # The model already has this edit; restore before applying
                    # its newly activated delta, otherwise the first fact doubles.
                    _restore_weights(edited_model, baseline_weights)
                    patch_service.activate_patch(patch_id, append=True)
                    if used_loaded_runtime:
                        patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
                # Keep the newly edited weights for the next fact. On failure,
                # the finally block restores that call's local pre-edit state.
                weights_restored = True
            else:
                _restore_weights(edited_model, baseline_weights)
                weights_restored = True
                patch = patch_service.create_patch(
                    model_id=model_id,
                    method=method_key,
                    edits=edits,
                    modified_layers=effective_layers,
                    metrics=json_metrics,
                    delta_paths=tensor_delta_paths,
                    runtime="easyedit",
                    extra_metadata={
                        "easyedit_hparams": str(hparams_path),
                        "hf_model": model["hf_id"],
                        "sequential_edit": sequential_edit,
                        "parent_patch_ids": active_patch_ids if sequential_edit else [],
                        "layer_source": layer_source or ("request" if layers else "easyedit_hparams"),
                        "mom2_adjustment": bool(getattr(hparams, "mom2_adjustment", False)),
                        "used_loaded_runtime": used_loaded_runtime,
                        **({"batch_size": batch_size} if method_key == "memit" else {}),
                    },
                    patch_id=patch_id,
                )
                patch_service.activate_patch(patch_id, append=sequential_edit)
                if used_loaded_runtime:
                    patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
            local_store.write_json(
                f"patches/{patch['patch_id']}/easyedit_metrics.json",
                {"metrics": json_metrics},
            )
            if transaction is not None and patch_mode == "series":
                transaction.checkpoint()
            return {
                "patch_id": patch["patch_id"],
                "metrics": _jsonable(metrics),
                "patch": patch,
                "runtime": "easyedit",
                "delta_count": len(tensor_delta_paths),
                "used_loaded_runtime": used_loaded_runtime,
            }
        finally:
            _restore_loaded_model_mom2_cache_identity(mom2_model_identity)
            if used_loaded_runtime and baseline_weights and not weights_restored:
                _restore_weights(runtime_bundle["model"], baseline_weights)
            if used_loaded_runtime:
                hparams.model_name = easyedit_model_name
                patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
            editor = None
            edited_model = None
            baseline_weights = None
            metrics = None
            _collect_runtime_memory()


def evaluate_easyedit_quality(
    model_id: str,
    method: str,
    edits: list[dict[str, Any]],
    include_generations: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_job_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate EasyEdit rewrite, rephrase, and locality quality on active weights.

    This deliberately bypasses ``BaseEditor.edit``: it reads the currently loaded
    model (including its active workbench patches) and performs no weight update.
    A record can include both ``rephrase_prompt`` and a ``locality`` payload, so
    EasyEdit computes rewrite/rephrase/locality outputs as one per-fact quality
    evaluation.  Locality additionally needs an unpatched baseline output; we
    collect that under the same runtime lock and let EasyEdit's locality
    finalizer calculate its normal pre/post preservation accuracy.  A job ID is
    checked between facts so canceling an evaluation does not terminate the
    model-owning worker or unload its model.
    """
    method_key = method.lower()
    spec = METHODS.get(method_key)
    if spec is None:
        raise EasyEditUnsupportedError(f"EasyEdit method is not configured: {method}")

    model_info = model_registry.get(model_id)
    if not model_info.get(spec["support_key"], False):
        raise EasyEditUnsupportedError(f"{model_id} is not configured for {method_key.upper()}.")
    if not edits:
        raise ValueError("Provide at least one edit fact to test reliability.")

    _, hparams_cls = _load_easyedit(spec["hparams_class"])
    hparams = hparams_cls.from_hparams(str(_resolve_hparams_path(model_info, spec["hparams_key"])))
    _override_model_name(hparams, model_info)
    model_name = str(model_info.get("local_path") or model_info["hf_id"])

    with _easyedit_runtime_access(model_id) as runtime_bundle:
        patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
        _configure_loaded_tokenizer(runtime_bundle["tokenizer"], model_name, method_key)
        _override_easyedit_device(hparams, runtime_bundle.get("device"))

        try:
            evaluation_module = importlib.import_module("easyeditor.evaluate")
            compute_edit_quality = evaluation_module.compute_edit_quality
            editor_module = importlib.import_module("easyeditor.editors.editor")
            finalize_locality_metrics = editor_module.finalize_locality_metrics
        except (ImportError, AttributeError) as exc:
            raise EasyEditRuntimeError(
                "Installed EasyEdit does not expose its edit-quality evaluator."
            ) from exc

        prepared_edits: list[tuple[dict[str, Any], str | None]] = []
        for raw_edit in edits:
            edit = raw_edit.model_dump() if hasattr(raw_edit, "model_dump") else dict(raw_edit)
            subject = str(edit.get("subject") or "").strip()
            relation_prompt = str(edit.get("relation_prompt") or "").strip()
            target_new = str(edit.get("new_object") or "").strip()
            if not subject or not relation_prompt or not target_new:
                raise ValueError("Each reliability test fact needs a subject, relation prompt, and new object.")
            prompt = build_prompt(subject, relation_prompt)
            record: dict[str, Any] = {
                "prompt": prompt,
                "target_new": target_new,
                "ground_truth": str(edit.get("ground_truth") or ""),
            }
            rephrase_prompts = [str(value).strip() for value in edit.get("rephrase_prompts") or [] if str(value).strip()]
            rephrase_prompt = rephrase_prompts[0] if rephrase_prompts else None
            if rephrase_prompt:
                # ``compute_edit_quality`` evaluates one edit record at a time.
                # Its rephrase path likewise expects one prompt aligned to this
                # record, rather than a list of prompts.
                record["rephrase_prompt"] = rephrase_prompt

            source_metadata = edit.get("source_metadata")
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            locality_prompt = str(edit.get("locality_prompt") or source_metadata.get("locality_prompt") or "").strip()
            locality_ground_truth = str(
                edit.get("locality_ground_truth") or source_metadata.get("locality_ground_truth") or ""
            ).strip()
            if locality_prompt and locality_ground_truth:
                # This is the same per-record shape BaseEditor._prepare_requests
                # gives compute_edit_quality for EasyEdit locality evaluation.
                record["locality"] = {
                    "neighborhood": {
                        "prompt": locality_prompt,
                        "ground_truth": locality_ground_truth,
                    }
                }
            prepared_edits.append((record, rephrase_prompt))

        locality_baselines: list[dict[str, Any] | None] = [None] * len(prepared_edits)
        if any("locality" in record for record, _ in prepared_edits):
            # EasyEdit locality is not target accuracy.  It compares the active
            # model's output against the same model before the edit.  Temporarily
            # replaying an empty patch stack gives that clean baseline without
            # creating another model instance.
            patch_service.sync_patch_stack_to_model(runtime_bundle["model"], model_id, [])
            try:
                for index, (record, _) in enumerate(prepared_edits):
                    if "locality" not in record:
                        continue
                    _ensure_evaluation_job_running(cancel_job_id)
                    locality_baselines[index] = compute_edit_quality(
                        runtime_bundle["model"],
                        model_name,
                        hparams,
                        runtime_bundle["tokenizer"],
                        record,
                        hparams.device,
                    )
            finally:
                patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)

        cases = []
        sample_generations: list[dict[str, str]] = []
        sample_rephrase_generations: list[dict[str, str]] = []
        sample_locality_generations: list[dict[str, str] | None] = []
        if include_generations:
            from app.ml.inference.generate import generate_text

            def generate_sample(prompt: str) -> dict[str, str]:
                return {
                    "prompt": prompt,
                    "clean_model": generate_text(
                        model_id,
                        prompt,
                        max_new_tokens=32,
                        patch_id=None,
                        use_active_patches=False,
                    )["text"],
                    "active_model": generate_text(
                        model_id,
                        prompt,
                        max_new_tokens=32,
                        patch_id=None,
                        use_active_patches=True,
                    )["text"],
                }

        total_edits = len(prepared_edits)
        for index, (record, rephrase_prompt) in enumerate(prepared_edits, start=1):
            _ensure_evaluation_job_running(cancel_job_id)
            metrics = compute_edit_quality(
                runtime_bundle["model"],
                model_name,
                hparams,
                runtime_bundle["tokenizer"],
                record,
                hparams.device,
            )
            locality_baseline = locality_baselines[index - 1]
            if locality_baseline is not None:
                # Keep EasyEdit's metric name, scoring semantics, and metric
                # metadata instead of inventing a workbench-specific locality
                # score from the raw token IDs.
                finalize_locality_metrics(
                    {"pre": locality_baseline, "post": metrics},
                    record,
                    hparams,
                    model_name,
                )
            metrics = _jsonable(metrics)
            cases.append(
                {
                    "prompt": record["prompt"],
                    "target_new": record["target_new"],
                    "reliability_score": _rewrite_accuracy_score(metrics),
                    "generalization_score": _rephrase_accuracy_score(metrics),
                    "locality_score": _locality_accuracy_score(metrics),
                    "rephrase_prompt": rephrase_prompt,
                    "locality_prompt": (
                        str(record["locality"]["neighborhood"]["prompt"])
                        if "locality" in record else None
                    ),
                    "metrics": metrics,
                }
            )
            if include_generations:
                sample_generations.append(generate_sample(record["prompt"]))
                if rephrase_prompt:
                    sample_rephrase_generations.append(generate_sample(rephrase_prompt))
                locality = record.get("locality")
                locality_prompt = (
                    str(locality["neighborhood"]["prompt"])
                    if isinstance(locality, dict)
                    and isinstance(locality.get("neighborhood"), dict)
                    and locality["neighborhood"].get("prompt")
                    else None
                )
                sample_locality_generations.append(
                    generate_sample(locality_prompt) if locality_prompt else None
                )
            if progress_callback is not None:
                progress_callback(index, total_edits)

    result = {
        "model_id": model_id,
        "method": method_key,
        "active_patch_ids": patch_service.active_patches_for_model(model_id),
        "metric_name": "rewrite_acc",
        "mean_reliability": _mean_case_score(cases, "reliability_score"),
        "mean_generalization": _mean_case_score(cases, "generalization_score"),
        "mean_locality": _mean_case_score(cases, "locality_score"),
        "cases": cases,
    }
    if include_generations:
        result["sample_generations"] = sample_generations
        if len(sample_rephrase_generations) == len(cases):
            result["sample_rephrase_generations"] = sample_rephrase_generations
        if any(sample is not None for sample in sample_locality_generations):
            # Keep positions aligned with ``cases``: facts without a locality
            # prompt intentionally have ``None`` rather than a shifted sample.
            result["sample_locality_generations"] = sample_locality_generations
    return result


def _ensure_evaluation_job_running(job_id: str | None) -> None:
    """Stop at a fact boundary without disrupting the loaded model worker."""
    check_edit_cancellation()
    if job_id is None:
        return

    # Import lazily: this evaluator is also imported from the model worker,
    # whose runtime service imports the editing runtime during command dispatch.
    from app.services.experiment_service import experiment_service

    job = experiment_service.get_job(job_id)
    if job is None or job.get("status") != "running":
        raise EasyEditRuntimeError("Evaluation canceled.")


def evaluate_post_edit_reliability(
    model_id: str,
    method: str,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one post-edit reliability pass and expose its active-model samples."""
    reliability = evaluate_easyedit_quality(
        model_id=model_id,
        method=method,
        edits=edits,
        include_generations=True,
    )
    targets = []
    for raw_edit in edits:
        edit = raw_edit.model_dump() if hasattr(raw_edit, "model_dump") else dict(raw_edit)
        targets.append(str(edit.get("new_object") or ""))

    generations = []
    samples = reliability.get("sample_generations")
    if isinstance(samples, list):
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            prompt = sample.get("prompt")
            active_model = sample.get("active_model")
            if not isinstance(prompt, str) or not isinstance(active_model, str):
                continue
            generations.append(
                {
                    "prompt": prompt,
                    "text": active_model,
                    "target_new": targets[index] if index < len(targets) else "",
                }
            )
    return {"reliability": reliability, "generations": generations}


def _rewrite_accuracy_score(metrics: dict[str, Any]) -> float | None:
    """Reduce EasyEdit's scalar or per-target-token rewrite accuracy to one score."""
    value = metrics.get("rewrite_acc")
    values = _numeric_values(value)
    return round(sum(values) / len(values), 4) if values else None


def _rephrase_accuracy_score(metrics: dict[str, Any]) -> float | None:
    """Reduce EasyEdit's scalar or token-level rephrase accuracy to one score."""
    values = _numeric_values(metrics.get("rephrase_acc"))
    return round(sum(values) / len(values), 4) if values else None


def _locality_accuracy_score(metrics: dict[str, Any]) -> float | None:
    """Reduce EasyEdit's per-locality pre/post preservation accuracies."""
    locality = metrics.get("locality")
    if not isinstance(locality, dict):
        return None
    values = [
        number
        for key, value in locality.items()
        if key.endswith("_acc")
        for number in _numeric_values(value)
    ]
    return round(sum(values) / len(values), 4) if values else None


def _mean_case_score(cases: list[dict[str, Any]], key: str) -> float | None:
    scores = [float(case[key]) for case in cases if isinstance(case.get(key), (int, float))]
    return round(sum(scores) / len(scores), 4) if scores else None


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [number for item in value for number in _numeric_values(item)]
    return []


def run_easyedit_kn(
    model_id: str,
    edit: dict[str, Any],
    sequential_edit: bool = False,
) -> dict[str, Any]:
    """Run EasyEdit Knowledge Neurons and save only its changed neuron vectors.

    KN mutates a sparse set of MLP-output vectors rather than a configured full
    rewrite matrix. Capturing those vectors directly keeps the resulting workbench
    patch small and replayable without retaining a full model-weight snapshot.
    """
    model_info = model_registry.get(model_id)
    if not model_info.get("supports_kn", False):
        raise EasyEditUnsupportedError(
            f"{model_id} is not configured for KN in this workbench. "
            "Select a model with EasyEdit KN hparams."
        )

    ground_truth = str(edit.get("ground_truth") or "").strip()
    if not ground_truth:
        raise EasyEditUnsupportedError("KN requires the original object as ground_truth.")

    _, hparams_cls = _load_easyedit("KNHyperParams")
    hparams_path = _resolve_hparams_path(model_info, "kn_hparams")
    hparams = hparams_cls.from_hparams(str(hparams_path))
    _override_model_name(hparams, model_info)
    prompt = build_prompt(str(edit["subject"]), str(edit["relation_prompt"]))
    active_patch_ids: list[str] = []
    runtime_bundle: dict[str, Any] | None = None
    snapshots: list[dict[str, Any]] = []
    weights_restored = False

    with _easyedit_runtime_access(model_id) as runtime_bundle:
        active_patch_ids = patch_service.active_patches_for_model(model_id)
        runtime_model = runtime_bundle["model"]
        runtime_tokenizer = runtime_bundle["tokenizer"]
        transaction = current_edit.get()
        if transaction is not None:
            transaction.capture(runtime_model, model_id)

        try:
            if not sequential_edit:
                patch_service.deactivate_runtime_patches_for_model(model_id)
                patch_service.sync_active_patches_to_model(runtime_model, model_id)
                active_patch_ids = []
                if transaction is not None:
                    transaction.notify_patch_state_changed()
            _configure_loaded_tokenizer(runtime_tokenizer, str(hparams.model_name), "kn")
            _override_easyedit_device(hparams, runtime_bundle.get("device"))
            KnowledgeNeurons, resolve_model_type = _load_kn_components()
            kn = KnowledgeNeurons(
                runtime_model,
                runtime_tokenizer,
                model_type=_resolve_kn_model_type(model_info, resolve_model_type),
                device=runtime_bundle.get("device") or "cpu",
            )
            refined_neurons = _normalize_kn_neurons(
                kn.get_refined_neurons(
                    [prompt],
                    ground_truth,
                    p=hparams.p,
                    batch_size=hparams.batch_size,
                    steps=hparams.steps,
                    coarse_adaptive_threshold=hparams.adaptive_threshold,
                    refine=hparams.refine,
                ),
                n_layers=int(model_info["n_layers"]),
            )
            if not refined_neurons:
                raise EasyEditRuntimeError(
                    "EasyEdit KN did not identify any knowledge neurons for this fact. "
                    "Try a more specific relation prompt or adjust the KN hparams."
                )

            snapshots = _snapshot_kn_neuron_vectors(kn, refined_neurons)
            if transaction is not None:
                transaction.capture(runtime_model, model_id, [item["parameter"] for item in snapshots])
            print("Starting EasyEdit algorithm.")
            metrics, _ = kn.edit_knowledge(
                prompt,
                target=str(edit["new_object"]),
                neurons=refined_neurons,
                undo_modification=False,
            )
            check_edit_cancellation()
            patch_id = patch_service.new_patch_id()
            if transaction is not None:
                transaction.created_patch_ids.add(patch_id)
            tensor_delta_paths, sparse_delta_specs = _save_kn_sparse_deltas(
                runtime_model,
                snapshots,
                patch_id,
            )
            if not tensor_delta_paths:
                raise EasyEditRuntimeError(
                    "EasyEdit KN completed, but did not produce any changed neuron vectors."
                )

            _restore_kn_neuron_vectors(runtime_model, snapshots)
            weights_restored = True
            patch = patch_service.create_patch(
                model_id=model_id,
                method="kn",
                edits=[edit],
                modified_layers=sorted({layer for layer, _ in refined_neurons}),
                metrics=_jsonable(metrics),
                delta_paths=tensor_delta_paths,
                runtime="easyedit",
                extra_metadata={
                    "easyedit_hparams": str(hparams_path),
                    "hf_model": model_info["hf_id"],
                    "sequential_edit": sequential_edit,
                    "parent_patch_ids": active_patch_ids if sequential_edit else [],
                    "used_loaded_runtime": True,
                    "kn_neurons": [{"layer": layer, "neuron": neuron} for layer, neuron in refined_neurons],
                    "sparse_delta_format": "kn_v1",
                    "sparse_delta_specs": sparse_delta_specs,
                },
                patch_id=patch_id,
            )
            patch_service.activate_patch(patch_id, append=sequential_edit)
            patch_service.sync_active_patches_to_model(runtime_model, model_id)
            local_store.write_json(
                f"patches/{patch['patch_id']}/easyedit_metrics.json",
                {"metrics": _jsonable(metrics)},
            )
            return {
                "patch_id": patch["patch_id"],
                "metrics": _jsonable(metrics),
                "patch": patch,
                "runtime": "easyedit",
                "neuron_count": len(refined_neurons),
                "modified_layers": patch["modified_layers"],
                "used_loaded_runtime": True,
            }
        finally:
            if snapshots and not weights_restored:
                _restore_kn_neuron_vectors(runtime_model, snapshots)
            patch_service.sync_active_patches_to_model(runtime_model, model_id)
            _collect_runtime_memory()


def _load_kn_components() -> tuple[Any, Any]:
    _add_easyedit_to_python_path()
    try:
        from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
    except ImportError as exc:
        raise EasyEditRuntimeError(
            "Installed EasyEdit does not include the Knowledge Neurons runtime. "
            "Update EasyEdit and install its requirements."
        ) from exc
    return KnowledgeNeurons, model_type


def _resolve_kn_model_type(model_info: dict[str, Any], resolver: Any) -> str:
    model_name = str(model_info.get("hf_id") or model_info.get("id") or "")
    try:
        return str(resolver(model_name))
    except ValueError:
        fallback_types = {
            "gpt2": "gpt2",
            "gptj": "gptj",
            "llama": "llama",
            "baichuan": "baichuan",
            "chatglm": "chatglm2",
            "internlm": "internlm",
            "mistral": "mistral",
            "qwen": "qwen2" if "qwen2" in str(model_info.get("id", "")).lower() else "qwen",
        }
        fallback = fallback_types.get(str(model_info.get("model_type") or "").lower())
        if fallback:
            return fallback
        raise EasyEditUnsupportedError(f"KN does not support this model architecture: {model_name}")


def _normalize_kn_neurons(value: Any, n_layers: int) -> list[tuple[int, int]]:
    if not isinstance(value, (list, tuple)):
        raise EasyEditRuntimeError("EasyEdit KN returned an invalid neuron list.")
    neurons: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        layer, neuron = int(item[0]), int(item[1])
        if not 0 <= layer < n_layers or neuron < 0:
            continue
        if (layer, neuron) not in neurons:
            neurons.append((layer, neuron))
    return neurons


def _snapshot_kn_neuron_vectors(kn: Any, neurons: list[tuple[int, int]]) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch to save KN neuron deltas.") from exc

    grouped: dict[int, list[int]] = {}
    for layer, neuron in neurons:
        grouped.setdefault(layer, []).append(neuron)

    parameter_names = {id(parameter): name for name, parameter in kn.model.named_parameters()}
    orientation = "row" if kn.model_type in {"gpt2", "chatglm2"} else "column"
    snapshots = []
    for layer, raw_indices in grouped.items():
        weight = kn._get_output_ff_layer(layer)
        parameter_name = parameter_names.get(id(weight))
        if parameter_name is None or not torch.is_tensor(weight):
            raise EasyEditRuntimeError(f"Unable to identify KN output-weight parameter for layer {layer}.")
        indices = sorted(set(raw_indices))
        limit = weight.shape[0] if orientation == "row" else weight.shape[1]
        if any(index >= limit for index in indices):
            raise EasyEditRuntimeError(f"EasyEdit KN returned an invalid neuron index for layer {layer}.")
        vector = weight[indices, :] if orientation == "row" else weight[:, indices].transpose(0, 1)
        snapshots.append(
            {
                "parameter": parameter_name,
                "orientation": orientation,
                "indices": indices,
                "vectors": vector.detach().cpu().clone(),
            }
        )
    return snapshots


def _save_kn_sparse_deltas(
    model: Any,
    snapshots: list[dict[str, Any]],
    patch_id: str,
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch and safetensors to save KN neuron deltas.") from exc

    parameter_map = dict(model.named_parameters())
    tensors: dict[str, Any] = {}
    specs: list[dict[str, str]] = []
    for snapshot in snapshots:
        parameter = parameter_map.get(snapshot["parameter"])
        if parameter is None:
            continue
        indices = torch.tensor(snapshot["indices"], dtype=torch.int64, device=parameter.device)
        current = parameter.detach()[indices, :] if snapshot["orientation"] == "row" else parameter.detach()[:, indices].transpose(0, 1)
        delta = (current.detach().cpu() - snapshot["vectors"]).contiguous()
        changed = torch.count_nonzero(delta, dim=1) > 0
        if not bool(torch.any(changed)):
            continue
        entry_name = f"entry_{len(specs)}"
        tensors[f"{entry_name}.indices"] = torch.tensor(snapshot["indices"], dtype=torch.int64)[changed]
        tensors[f"{entry_name}.values"] = delta[changed]
        specs.append(
            {
                "entry": entry_name,
                "parameter": str(snapshot["parameter"]),
                "orientation": str(snapshot["orientation"]),
            }
        )
    if not tensors:
        return [], []
    path = patch_service.patch_dir(patch_id) / "kn_sparse_deltas.safetensors"
    save_file(tensors, str(path), metadata={"format": "kew_kn_sparse_v1"})
    return [str(path)], specs


def _restore_kn_neuron_vectors(model: Any, snapshots: list[dict[str, Any]]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch to restore KN neuron vectors.") from exc

    parameter_map = dict(model.named_parameters())
    with torch.no_grad():
        for snapshot in snapshots:
            parameter = parameter_map.get(snapshot["parameter"])
            if parameter is None:
                continue
            indices = torch.tensor(snapshot["indices"], dtype=torch.int64, device=parameter.device)
            values = snapshot["vectors"].to(device=parameter.device, dtype=parameter.dtype)
            if snapshot["orientation"] == "row":
                parameter[indices, :] = values
            else:
                parameter[:, indices] = values.transpose(0, 1)


def _load_easyedit(hparams_class_name: str) -> tuple[Any, Any]:
    _add_easyedit_to_python_path()
    try:
        import easyeditor
        from easyeditor import BaseEditor
    except ImportError as exc:
        raise EasyEditRuntimeError(
            "EasyEdit is not importable. Clone https://github.com/zjunlp/EasyEdit, "
            "install its requirements, and set EASYEDIT_ROOT to that checkout."
        ) from exc

    _configure_easyedit_statistics_loader()
    hparams_cls = getattr(easyeditor, hparams_class_name, None)
    if hparams_cls is None:
        raise EasyEditRuntimeError(
            f"Installed EasyEdit does not expose {hparams_class_name}. "
            "Update EasyEdit from https://github.com/zjunlp/EasyEdit."
        )
    return BaseEditor, hparams_cls


def _configure_easyedit_statistics_loader() -> None:
    """Avoid worker spawning for EasyEdit's nested, non-picklable collate function."""
    for module_name in (
        "easyeditor.models.rome.layer_stats",
        "easyeditor.models.r_rome.layer_stats",
    ):
        try:
            layer_stats_module = importlib.import_module(module_name)
        except ImportError:
            continue

        if getattr(layer_stats_module, "_workbench_single_process_loader", False):
            continue

        original_tally = layer_stats_module.tally

        def single_process_tally(*args, _original_tally=original_tally, **kwargs):
            kwargs["num_workers"] = 0
            return _original_tally(*args, **kwargs)

        layer_stats_module.tally = single_process_tally
        layer_stats_module._workbench_single_process_loader = True


def _add_easyedit_to_python_path() -> None:
    root = _resolve_easyedit_root()
    if root is None:
        checked = ", ".join(str(path) for path in _candidate_easyedit_roots())
        raise EasyEditRuntimeError(
            f"EASYEDIT_ROOT does not exist. Checked: {checked}. "
            "Clone https://github.com/zjunlp/EasyEdit and set EASYEDIT_ROOT to that checkout."
        )
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)


def _candidate_easyedit_roots() -> list[Path]:
    configured = settings.easyedit_root
    backend_root = Path(__file__).resolve().parents[3]
    repo_root = backend_root.parent
    candidates = [configured.resolve()]

    if not configured.is_absolute():
        candidates.extend(
            [
                (repo_root / configured).resolve(),
                (backend_root / configured).resolve(),
            ]
        )

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_easyedit_root() -> Path | None:
    for candidate in _candidate_easyedit_roots():
        if (candidate / "easyeditor").exists():
            return candidate
    return None


def _resolve_hparams_path(model: dict[str, Any], hparams_key: str) -> Path:
    relative = model.get("easyedit", {}).get(hparams_key)
    if not relative:
        raise EasyEditUnsupportedError(f"No EasyEdit hparams configured for {model['id']}.")

    candidates = [
        *[
            root / relative
            for root in _candidate_easyedit_roots()
        ],
        Path.cwd() / relative,
        Path.cwd().parent / relative,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    try:
        import easyeditor
    except ImportError:
        pass
    else:
        package_root = Path(easyeditor.__file__).resolve().parent.parent
        package_candidate = package_root / relative
        if package_candidate.exists():
            return package_candidate
        candidates.append(package_candidate)

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise EasyEditRuntimeError(
        f"EasyEdit hparams file not found for {model['id']}: {relative}. Checked: {checked}. "
        "Clone https://github.com/zjunlp/EasyEdit and set EASYEDIT_ROOT to that checkout."
    )


def _override_model_name(hparams: Any, model: dict[str, Any]) -> None:
    model_name = model.get("local_path") or model["hf_id"]
    if hasattr(hparams, "model_name"):
        setattr(hparams, "model_name", model_name)
    if hasattr(hparams, "tokenizer_name"):
        setattr(hparams, "tokenizer_name", model_name)


def _mom2_base_model(model: dict[str, Any]) -> dict[str, Any] | None:
    """Return the supported hub model an imported model was derived from."""
    if model.get("source") != "local":
        return None
    base_model_id = model.get("base_model_id")
    if not isinstance(base_model_id, str) or not base_model_id:
        return None
    try:
        base_model = model_registry.get(base_model_id)
    except ValueError:
        return None
    return base_model if base_model.get("source") == "hub" else None


def _resolve_mom2_statistics_source(
    model: dict[str, Any],
    requested_source: Literal["base", "model"] | None = None,
) -> Literal["base", "model"]:
    """Resolve an imported model's persisted MOM2 cache preference."""
    if requested_source not in {None, "base", "model"}:
        raise EasyEditRuntimeError("MOM2 statistics source must be 'base' or 'model'.")

    base_model = _mom2_base_model(model)
    if model.get("source") != "local":
        if requested_source == "base":
            raise EasyEditRuntimeError("Hub models use their own MOM2 statistics.")
        return "model"

    source = requested_source or model.get("mom2_stats_source")
    if source is None:
        # Preserve the established behavior for existing imported models.
        source = "base" if base_model is not None else "model"
    if source == "base" and base_model is None:
        raise EasyEditRuntimeError(
            f"{model['id']} does not have a supported base-model MOM2 cache. "
            "Use this imported model's own statistics instead."
        )
    return source


def _mom2_cache_model_name(
    model: dict[str, Any],
    stats_source: Literal["base", "model"] | None = None,
) -> str:
    """Return the stable model identity used by EasyEdit's MOM2 filenames."""
    source = _resolve_mom2_statistics_source(model, stats_source)
    if source == "base":
        base_model = _mom2_base_model(model)
        if base_model is not None:
            return str(base_model["hf_id"])
    if model.get("source") == "local":
        # The opaque import ID is unique and avoids collisions between two
        # modified snapshots that happen to have the same folder name.
        return str(model["id"])
    return str(model.get("hf_id") or model.get("local_path") or model["id"])


def _set_loaded_model_mom2_cache_identity(
    loaded_model: Any,
    model: dict[str, Any],
    stats_source: Literal["base", "model"] | None = None,
) -> tuple[Any, str] | None:
    """Temporarily align EasyEdit's model-config cache key with the chosen cache."""
    config = getattr(loaded_model, "config", None)
    if config is None or not hasattr(config, "_name_or_path"):
        return None
    previous_name = str(config._name_or_path)
    cache_name = _mom2_cache_model_name(model, stats_source)
    if previous_name == cache_name:
        return None
    config._name_or_path = cache_name
    return config, previous_name


def _restore_loaded_model_mom2_cache_identity(identity: tuple[Any, str] | None) -> None:
    if identity is None:
        return
    config, previous_name = identity
    config._name_or_path = previous_name


@contextmanager
def _easyedit_runtime_access(model_id: str):
    with model_runtime_service.exclusive_runtime_access():
        bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
        device = str(bundle.get("device") or "cpu")
        with mps_fp32(device):
            yield bundle


def _override_easyedit_device(hparams: Any, loaded_device: str | None = None) -> None:
    if not hasattr(hparams, "device"):
        return
    if loaded_device == "cpu" and getattr(hparams, "alg_name", None) in {"FT", "LoRA"}:
        # Training profiles often contain a machine-specific CUDA index.
        hparams.device = "cpu"
        return
    if loaded_device and loaded_device.startswith("cuda"):
        setattr(hparams, "device", _cuda_device_index(loaded_device))
        return
    if loaded_device and loaded_device.startswith("mps"):
        configure_mps_hparams(hparams)
        setattr(hparams, "device", "mps")
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        setattr(hparams, "device", 0)


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.rsplit(":", 1)[1])
    except ValueError:
        return 0


def _configure_loaded_tokenizer(tokenizer: Any, model_name: str, method_key: str) -> None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if getattr(tokenizer, "pad_token_id", None) is None and eos_token_id is not None:
        tokenizer.pad_token_id = eos_token_id
    if method_key in {"rome", "r_rome", "memit"} and any(name in model_name.lower() for name in ("mistral", "llama", "qwen")):
        tokenizer.padding_side = "right"


def _override_layers(hparams: Any, layers: list[int] | None) -> None:
    if not layers:
        return
    if hasattr(hparams, "layers"):
        setattr(hparams, "layers", layers)
    if hasattr(hparams, "layer"):
        setattr(hparams, "layer", layers[0])


def _normalize_mom2_dtype(dtype: str) -> str:
    dtype = {"half": "float16", "float": "float32", "double": "float64"}.get(dtype, dtype)
    if dtype not in MOM2_DTYPES:
        raise EasyEditRuntimeError(f"Unsupported MOM2 precision: {dtype}. Choose float16, float32, or float64.")
    return dtype


def _mom2_precision_device(model_id: str) -> str:
    status = model_runtime_service.runtime_status(model_id)
    device = str(status.get("device") or "not_loaded")
    if device == "not_loaded":
        gpu = status.get("gpu", {})
        device = "cuda" if gpu.get("cuda_available") else "mps" if gpu.get("mps_available") else "cpu"
    return device


def _read_mom2_precision_preferences() -> dict[str, Any]:
    try:
        return local_store.read_json("settings/mom2_precision.json")
    except FileNotFoundError:
        return {}


def _resolve_mom2_precision(model_id: str, method: str, default: str, device: str, requested: str | None = None) -> str:
    if requested is not None:
        return _normalize_mom2_dtype(requested)
    with _mom2_precision_lock:
        selected = _read_mom2_precision_preferences().get(model_id, {}).get(method.lower())
    # A cache saved on a CPU/CUDA installation may later be opened on MPS.
    return editing_precision(_normalize_mom2_dtype(selected or default), device)


def _configure_mom2_precision(hparams: Any, model_id: str, method: str, device: str, requested: str | None = None) -> None:
    precision = _resolve_mom2_precision(model_id, method, str(hparams.mom2_dtype), device, requested)
    if precision == "float64" and device.startswith("mps"):
        raise EasyEditRuntimeError("FP64 second-moment computation is not supported on MPS. Select FP32 or FP16.")
    hparams.mom2_dtype = precision


def _mom2_precision_metadata(hparams: Any) -> dict[str, Any]:
    return {
        "mom2_dtype": hparams.mom2_dtype,
        "default_mom2_dtype": hparams.default_mom2_dtype,
        "available_mom2_dtypes": hparams.available_mom2_dtypes,
        "device": hparams.device,
    }


def set_mom2_statistics_precision(model_id: str, method: str, mom2_dtype: str | None) -> dict[str, Any]:
    profile, _ = _load_mom2_cache_profile(model_id, method, None, mom2_dtype=mom2_dtype)
    if mom2_dtype is not None and profile.mom2_dtype not in profile.available_mom2_dtypes:
        raise EasyEditRuntimeError("FP64 second-moment computation is not supported on MPS. Select FP32 or FP16.")
    with _mom2_precision_lock:
        preferences = _read_mom2_precision_preferences()
        selected = preferences.setdefault(model_id, {})
        if mom2_dtype is None:
            selected.pop(method.lower(), None)
        else:
            selected[method.lower()] = profile.mom2_dtype
        local_store.write_json("settings/mom2_precision.json", preferences)
    return get_mom2_statistics_status(model_id, method)


def _refresh_mom2_covariance_cache(method: str, hparams: Any, model: Any, device: str) -> None:
    module_name, cache_name = (
        ("easyeditor.models.memit.memit_main", "COV_CACHE") if method == "memit"
        else (f"easyeditor.models.{method}.compute_u", "inv_mom2_cache")
    )
    module = sys.modules.get(module_name)
    cache = getattr(module, cache_name, None)
    if not isinstance(cache, dict):
        return
    for layer in _hparams_layers(hparams):
        path = _mom2_statistics_path(hparams, model, layer)
        key = (str(model.config._name_or_path).replace("/", "_"), hparams.rewrite_module_tmp.format(layer))
        stamp = path.stat() if path.exists() else None
        profile = (str(path), stamp.st_mtime_ns if stamp else None, stamp.st_size if stamp else None)
        profile_key = (module_name, key)
        # EasyEdit's cache key omits dataset, sample count, precision, and file version.
        if _mom2_covariance_profiles.get(profile_key) != profile:
            cache.pop(key, None)
            _mom2_covariance_profiles[profile_key] = profile
        if method != "memit" and hparams.mom2_dtype == "float16" and key not in cache:
            # ROME inverses cannot operate on FP16 tensors. Keep the stored stats
            # FP16, but perform the inverse in FP32, as EasyEdit returns FP32 anyway.
            import numpy as np
            import torch

            with np.load(path, allow_pickle=False) as archive:
                moment = torch.from_numpy(archive["mom2.mom2"]).float() / int(archive["mom2.count"])
            cache[key] = torch.linalg.inv(moment.to(device)).float()


def get_mom2_statistics_status(
    model_id: str,
    method: str,
    layers: list[int] | None = None,
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> dict[str, Any]:
    """Report the MOM2 cache files EasyEdit will use for this runtime model."""
    hparams, effective_layers = _load_mom2_cache_profile(
        model_id, method, layers, stats_source, mom2_dtype
    )

    statistics = []
    for layer in effective_layers:
        path = _mom2_statistics_path(hparams, None, layer)
        statistics.append({"layer": layer, "path": str(path), "ready": path.exists() and _mom2_archive_dtype(path) == hparams.mom2_dtype})
    missing_layers = [entry["layer"] for entry in statistics if not entry["ready"]]
    return {
        "model_id": model_id,
        "method": method.lower(),
        "mom2_adjustment": True,
        "dataset": str(getattr(hparams, "mom2_dataset")),
        "layers": effective_layers,
        "stats_dir": str(getattr(hparams, "stats_dir")),
        "stats_source": str(getattr(hparams, "mom2_stats_source")),
        "available_stats_sources": list(getattr(hparams, "available_mom2_stats_sources")),
        "base_model_id": getattr(hparams, "base_model_id"),
        "cache_model_id": str(getattr(hparams, "mom2_cache_model_name")),
        **_mom2_precision_metadata(hparams),
        "ready": not missing_layers,
        "missing_layers": missing_layers,
        "statistics": statistics,
    }


def set_mom2_statistics_source(
    model_id: str,
    stats_source: Literal["base", "model"],
) -> dict[str, Any]:
    """Persist which cache an imported model should use for future edits."""
    model = model_registry.get(model_id)
    resolved_source = _resolve_mom2_statistics_source(model, stats_source)
    if model.get("source") == "local":
        model_registry.set_mom2_statistics_source(model_id, resolved_source)
    return {
        "model_id": model_id,
        "stats_source": resolved_source,
        "available_stats_sources": ["base", "model"]
        if _mom2_base_model(model) is not None
        else ["model"],
        "base_model_id": (_mom2_base_model(model) or {}).get("id"),
    }


def require_mom2_statistics(
    model_id: str,
    method: str,
    layers: list[int] | None,
    mom2_adjustment: bool,
) -> dict[str, Any]:
    if not mom2_adjustment:
        return {"model_id": model_id, "method": method.lower(), "mom2_adjustment": False, "ready": True}

    # The stats path is derived solely from the configured hparams and requested
    # layers. Check it in the API process before dispatching an edit to the
    # model worker, so missing statistics cannot start an EasyEdit operation.
    status = get_mom2_statistics_status(model_id, method, layers)
    if not status["ready"]:
        rendered_layers = ", ".join(str(layer) for layer in status["missing_layers"])
        raise EasyEditRuntimeError(
            f"MOM2 statistics are missing for {model_id} layer(s) {rendered_layers}. "
            "Prepare them from Settings before applying this edit."
        )
    return status


def prepare_mom2_statistics(
    model_id: str,
    method: str,
    layers: list[int] | None = None,
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> dict[str, Any]:
    """Compute the missing EasyEdit MOM2 covariance caches for selected layers."""
    model = model_registry.get(model_id)
    resolved_source = _resolve_mom2_statistics_source(model, stats_source)
    if model.get("source") == "local" and resolved_source == "base":
        raise EasyEditRuntimeError(
            "Base-model MOM2 statistics cannot be computed from an imported model's "
            "modified weights. Select this imported model's own statistics first."
        )
    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "prepare_mom2",
            {
                "model_id": model_id,
                "method": method,
                "layers": layers,
                "stats_source": resolved_source,
                "mom2_dtype": mom2_dtype,
            },
            timeout=7200.0,
        )
    hparams, effective_layers = _load_edit_hparams(model_id, method, layers)
    _override_mom2_adjustment(hparams, True)

    with _easyedit_runtime_access(model_id) as runtime_bundle:
        _configure_loaded_tokenizer(runtime_bundle["tokenizer"], str(hparams.model_name), method.lower())
        _configure_mom2_precision(hparams, model_id, method, str(runtime_bundle.get("device") or "cpu"), mom2_dtype)
        _override_easyedit_device(hparams, runtime_bundle.get("device"))
        mom2_model_identity = _set_loaded_model_mom2_cache_identity(
            runtime_bundle["model"], model, resolved_source
        )

        try:
            from easyeditor.models.rome.layer_stats import layer_stats

            for layer in effective_layers:
                path = _mom2_statistics_path(hparams, runtime_bundle["model"], layer)
                if path.exists():
                    print(f"MOM2 statistics already cached for layer {layer}: {path}")
                    continue
                print(f"Preparing MOM2 statistics for layer {layer}: {path}")
                layer_stats(
                    runtime_bundle["model"],
                    runtime_bundle["tokenizer"],
                    str(getattr(hparams, "rewrite_module_tmp")).format(layer),
                    getattr(hparams, "stats_dir"),
                    getattr(hparams, "mom2_dataset"),
                    to_collect=["mom2"],
                    sample_size=getattr(hparams, "mom2_n_samples"),
                    precision=getattr(hparams, "mom2_dtype"),
                    hparams=hparams,
                )
                if _mom2_archive_dtype(path) != hparams.mom2_dtype:
                    path.unlink(missing_ok=True)
                    raise EasyEditRuntimeError("Computed MOM2 archive does not match the selected precision.")
        finally:
            _restore_loaded_model_mom2_cache_identity(mom2_model_identity)

    if model.get("source") == "local" and resolved_source == "model":
        model_registry.set_mom2_statistics_source(model_id, "model")
    return get_mom2_statistics_status(model_id, method, effective_layers, resolved_source, hparams.mom2_dtype)


def import_mom2_statistics(
    model_id: str,
    method: str,
    dataset: str,
    layers: list[int] | None,
    files: list[tuple[str, Any]],
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically install precomputed MOM2 cache files."""
    hparams, effective_layers = _load_mom2_cache_profile(
        model_id, method, layers, stats_source, mom2_dtype
    )
    expected_dataset = str(getattr(hparams, "mom2_dataset"))
    if dataset.strip() != expected_dataset:
        raise EasyEditRuntimeError(
            f"{method.upper()} for {model_id} expects {expected_dataset!r} MOM2 statistics, not {dataset!r}."
        )

    expected_files = {
        _mom2_statistics_path(hparams, None, layer).name: (layer, _mom2_statistics_path(hparams, None, layer))
        for layer in effective_layers
    }
    imported_layers: list[int] = []
    staged_files: list[tuple[int, Path, Path]] = []
    try:
        for uploaded_name, uploaded_file in files:
            filename = Path(uploaded_name).name
            if Path(filename).suffix.lower() != ".npz":
                raise EasyEditRuntimeError(f"{filename!r} is not an NPZ statistics file.")
            expected = expected_files.get(filename)
            if expected is None:
                raise EasyEditRuntimeError(
                    f"{filename!r} does not match a selected {method.upper()} layer "
                    f"at {hparams.mom2_dtype} for {model_id}."
                )
            layer, destination = expected
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                staged_files.append((layer, temporary_path, destination))
                shutil.copyfileobj(uploaded_file, temporary)
            _validate_mom2_archive(temporary_path, hparams.mom2_n_samples, hparams.mom2_dtype)
        # Validate every upload before replacing any existing precision-specific file.
        for layer, temporary_path, destination in staged_files:
            os.replace(temporary_path, destination)
            imported_layers.append(layer)
    finally:
        for _, temporary_path, _ in staged_files:
            temporary_path.unlink(missing_ok=True)

    if imported_layers and getattr(hparams, "mom2_stats_source") == "model":
        model = model_registry.get(model_id)
        if model.get("source") == "local":
            model_registry.set_mom2_statistics_source(model_id, "model")
    status = get_mom2_statistics_status(
        model_id, method, effective_layers, getattr(hparams, "mom2_stats_source"), hparams.mom2_dtype
    )
    status["imported_layers"] = sorted(set(imported_layers))
    return status


def list_mom2_statistics(
    model_id: str,
    method: str,
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> dict[str, Any]:
    """List all cached second-moment files that match a model/method hparams profile."""
    hparams, _ = _load_mom2_cache_profile(model_id, method, None, stats_source, mom2_dtype)
    model = model_registry.get(model_id)
    layer_count = int(model.get("n_layers", 0))
    statistics = []
    for layer in range(max(0, layer_count)):
        path = _mom2_statistics_path(hparams, None, layer)
        if not path.exists():
            continue
        stat = path.stat()
        statistics.append(
            {
                "layer": layer,
                "path": str(path),
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "mom2_dtype": hparams.mom2_dtype,
            }
        )
    return {
        "model_id": model_id,
        "method": method.lower(),
        "dataset": str(getattr(hparams, "mom2_dataset")),
        "stats_dir": str(getattr(hparams, "stats_dir")),
        "stats_source": str(getattr(hparams, "mom2_stats_source")),
        "available_stats_sources": list(getattr(hparams, "available_mom2_stats_sources")),
        "base_model_id": getattr(hparams, "base_model_id"),
        "cache_model_id": str(getattr(hparams, "mom2_cache_model_name")),
        **_mom2_precision_metadata(hparams),
        "statistics": statistics,
    }


def delete_mom2_statistics(
    model_id: str,
    method: str,
    layers: list[int],
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> dict[str, Any]:
    if not layers:
        raise EasyEditRuntimeError("Select one or more second-moment statistics files to delete.")
    hparams, _ = _load_mom2_cache_profile(model_id, method, None, stats_source, mom2_dtype)
    model = model_registry.get(model_id)
    layer_count = int(model.get("n_layers", 0))
    invalid_layers = [layer for layer in layers if not isinstance(layer, int) or layer < 0 or layer >= layer_count]
    if invalid_layers:
        raise EasyEditRuntimeError(f"Invalid {model_id} layer(s): {', '.join(str(layer) for layer in invalid_layers)}.")

    deleted_layers = []
    for layer in sorted(set(layers)):
        path = _mom2_statistics_path(hparams, None, layer)
        if path.exists():
            path.unlink()
            deleted_layers.append(layer)
    result = list_mom2_statistics(
        model_id, method, getattr(hparams, "mom2_stats_source"), hparams.mom2_dtype
    )
    result["deleted_layers"] = deleted_layers
    return result


def _load_mom2_cache_profile(
    model_id: str,
    method: str,
    layers: list[int] | None,
    stats_source: Literal["base", "model"] | None = None,
    mom2_dtype: str | None = None,
) -> tuple[SimpleNamespace, list[int]]:
    """Read the hparams fields that determine a MOM2 cache filename.

    Cache inspection, import, and preflight validation do not instantiate an
    EasyEdit editor. Parsing the YAML directly keeps those operations fast and
    avoids importing EasyEdit's full optional dependency graph in the API
    process.
    """
    method_key = method.lower()
    spec = METHODS.get(method_key)
    if spec is None:
        raise EasyEditUnsupportedError(f"EasyEdit method is not configured: {method}")

    model = model_registry.get(model_id)
    if not model.get(spec["support_key"], False):
        raise EasyEditUnsupportedError(f"{model_id} is not configured for {method_key.upper()}.")

    hparams_path = _resolve_hparams_path(model, spec["hparams_key"])
    try:
        payload = yaml.safe_load(hparams_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise EasyEditRuntimeError(f"Invalid EasyEdit hparams YAML: {hparams_path}") from exc
    if not isinstance(payload, dict):
        raise EasyEditRuntimeError(f"EasyEdit hparams must be a mapping: {hparams_path}")

    required = ("rewrite_module_tmp", "mom2_dataset", "mom2_dtype")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise EasyEditRuntimeError(
            f"EasyEdit hparams is missing MOM2 field(s) {', '.join(missing)}: {hparams_path}"
        )

    configured_layers = payload.get("layers", payload.get("layer"))
    if layers:
        effective_layers = [int(layer) for layer in layers]
    elif isinstance(configured_layers, list):
        effective_layers = [int(layer) for layer in configured_layers]
    elif configured_layers is not None:
        effective_layers = [int(configured_layers)]
    else:
        effective_layers = []
    if not effective_layers:
        raise EasyEditRuntimeError(f"EasyEdit hparams has no configured edit layers for {model_id}.")

    resolved_source = _resolve_mom2_statistics_source(model, stats_source)
    base_model = _mom2_base_model(model)
    default_precision = _normalize_mom2_dtype(str(payload["mom2_dtype"]))
    device = _mom2_precision_device(model_id)
    precision = _resolve_mom2_precision(model_id, method_key, default_precision, device, mom2_dtype)
    return (
        SimpleNamespace(
            model_name=_mom2_cache_model_name(model, resolved_source),
            stats_dir=str(_mom2_stats_dir()),
            rewrite_module_tmp=str(payload["rewrite_module_tmp"]),
            mom2_dataset=str(payload["mom2_dataset"]),
            mom2_dtype=precision,
            default_mom2_dtype=default_precision,
            available_mom2_dtypes=[dtype for dtype in MOM2_DTYPES if dtype != "float64" or not device.startswith("mps")],
            device=device,
            mom2_n_samples=payload.get("mom2_n_samples"),
            layers=effective_layers,
            mom2_stats_source=resolved_source,
            available_mom2_stats_sources=(
                ["base", "model"] if base_model is not None else ["model"]
            ),
            base_model_id=base_model.get("id") if base_model is not None else None,
            mom2_cache_model_name=_mom2_cache_model_name(model, resolved_source),
        ),
        effective_layers,
    )


def _load_edit_hparams(model_id: str, method: str, layers: list[int] | None) -> tuple[Any, list[int]]:
    method_key = method.lower()
    spec = METHODS.get(method_key)
    if spec is None:
        raise EasyEditUnsupportedError(f"EasyEdit method is not configured: {method}")

    model = model_registry.get(model_id)
    if not model.get(spec["support_key"], False):
        raise EasyEditUnsupportedError(f"{model_id} is not configured for {method_key.upper()}.")

    _, hparams_cls = _load_easyedit(spec["hparams_class"])
    hparams = hparams_cls.from_hparams(str(_resolve_hparams_path(model, spec["hparams_key"])))
    _override_model_name(hparams, model)
    _override_layers(hparams, layers)
    _configure_stats_dir(hparams)
    effective_layers = _hparams_layers(hparams)
    if not effective_layers:
        raise EasyEditRuntimeError(f"EasyEdit hparams has no configured edit layers for {model_id}.")
    return hparams, effective_layers


def _override_mom2_adjustment(hparams: Any, enabled: bool | None) -> None:
    if enabled is not None and hasattr(hparams, "mom2_adjustment"):
        setattr(hparams, "mom2_adjustment", enabled)


def _configure_stats_dir(hparams: Any) -> None:
    if not hasattr(hparams, "stats_dir"):
        return
    setattr(hparams, "stats_dir", str(_mom2_stats_dir()))


def _mom2_stats_dir() -> Path:
    backend_root = Path(__file__).resolve().parents[3]
    data_dir = settings.data_dir
    if not data_dir.is_absolute():
        data_dir = (backend_root / data_dir).resolve()
    return data_dir / "stats"


def _mom2_statistics_path(hparams: Any, model: Any | None, layer: int) -> Path:
    model_source = getattr(model.config, "_name_or_path", None) if model is not None else getattr(hparams, "model_name", "model")
    model_name = str(model_source).rsplit("/", 1)[-1]
    layer_name = str(getattr(hparams, "rewrite_module_tmp")).format(layer)
    dataset = str(getattr(hparams, "mom2_dataset"))
    precision = str(getattr(hparams, "mom2_dtype"))
    sample_size = getattr(hparams, "mom2_n_samples")
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    return Path(str(getattr(hparams, "stats_dir"))) / f"{model_name}/{dataset}_stats/{layer_name}_{precision}_mom2{size_suffix}.npz"


def _mom2_archive_dtype(path: Path) -> str | None:
    # Inspect only the NPY header; cache preflight must not load full covariance
    # matrices into the API process just to verify their stored precision.
    import numpy as np

    try:
        with zipfile.ZipFile(path) as archive, archive.open("mom2.mom2.npy") as matrix:
            version = np.lib.format.read_magic(matrix)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(matrix)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(matrix)
            else:
                return None
            if len(shape) != 2 or shape[0] != shape[1]:
                return None
            return dtype.name
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _validate_mom2_archive(path: Path, expected_sample_size: int | None, expected_dtype: str | None = None) -> None:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"sample_size", "mom2.count", "mom2.mom2"}
            missing = required - keys
            if missing:
                raise EasyEditRuntimeError(
                    f"Statistics archive is missing required data: {', '.join(sorted(missing))}."
                )
            if expected_sample_size is not None and int(archive["sample_size"]) != expected_sample_size:
                raise EasyEditRuntimeError(
                    f"Statistics archive uses sample size {int(archive['sample_size'])}, "
                    f"but this hparams configuration requires {expected_sample_size}."
                )
            matrix = archive["mom2.mom2"]
            if int(archive["mom2.count"]) <= 0:
                raise EasyEditRuntimeError("Statistics archive has no collected samples.")
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise EasyEditRuntimeError("Statistics archive has an invalid second-moment matrix.")
            if expected_dtype is not None and matrix.dtype != np.dtype(expected_dtype):
                raise EasyEditRuntimeError(
                    f"Statistics matrix precision is {matrix.dtype}, but {expected_dtype} was selected. "
                    "Upload matching files without renaming their precision."
                )
            if not np.isfinite(matrix).all():
                raise EasyEditRuntimeError("Statistics matrix contains non-finite values. Use a higher precision.")
    except EasyEditRuntimeError:
        raise
    except Exception as exc:
        raise EasyEditRuntimeError("Unable to read the uploaded MOM2 statistics archive.") from exc


def _hparams_layers(hparams: Any) -> list[int]:
    if hasattr(hparams, "layers"):
        layers = getattr(hparams, "layers")
        if isinstance(layers, list):
            return [int(layer) for layer in layers]
        if layers is not None:
            return [int(layers)]
    if hasattr(hparams, "layer"):
        layer = getattr(hparams, "layer")
        if layer is not None:
            return [int(layer)]
    return []


def _state_dict_for_weight_names(model: Any, names: list[str]) -> dict[str, Any]:
    """Return the state dict whose keys match the configured EasyEdit modules."""
    candidates = [model]
    inner_model = getattr(model, "model", None)
    if inner_model is not None and inner_model is not model:
        candidates.append(inner_model)

    fallback = candidates[0].state_dict()
    for candidate in candidates:
        state_dict = fallback if candidate is candidates[0] else candidate.state_dict()
        if any(name in state_dict for name in names):
            return state_dict
    return fallback


def _snapshot_rewrite_weights(model: Any, hparams: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch to snapshot EasyEdit weights.") from exc

    rewrite_module_tmp = getattr(hparams, "rewrite_module_tmp", None)
    if not rewrite_module_tmp:
        return {}

    names = [f"{rewrite_module_tmp.format(layer)}.weight" for layer in _hparams_layers(hparams)]
    state_dict = _state_dict_for_weight_names(model, names)
    weights = {}
    for name in names:
        value = state_dict.get(name)
        if value is not None and torch.is_tensor(value):
            weights[name] = value.detach().cpu().clone()
    return weights


def _save_weight_deltas(edited_model: Any, baseline_weights: Any, patch_id: str) -> list[str]:
    if not baseline_weights:
        return []

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch and safetensors to save EasyEdit deltas.") from exc

    state_dict = _state_dict_for_weight_names(edited_model, list(dict(baseline_weights)))
    deltas = {}

    for name, original in dict(baseline_weights).items():
        if name not in state_dict or not torch.is_tensor(original):
            continue
        edited = state_dict[name].detach().cpu()
        original_cpu = original.detach().cpu()
        delta = (edited - original_cpu).contiguous()
        if torch.count_nonzero(delta).item() == 0:
            continue
        deltas[name] = delta

    if not deltas:
        return []

    patch_dir = patch_service.patch_dir(patch_id)
    path = patch_dir / "weight_deltas.safetensors"
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".weight_deltas-",
        suffix=".safetensors",
        dir=patch_dir,
    )
    os.close(file_descriptor)
    try:
        save_file(deltas, temporary_path, metadata={"format": "easyedit_delta"})
        Path(temporary_path).replace(path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return [str(path)]


def _restore_weights(edited_model: Any, baseline_weights: Any) -> None:
    if not baseline_weights:
        return
    try:
        import torch
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch to restore EasyEdit weights.") from exc

    state_dict = _state_dict_for_weight_names(edited_model, list(dict(baseline_weights)))
    with torch.no_grad():
        for name, original in dict(baseline_weights).items():
            current = state_dict.get(name)
            if current is None or not torch.is_tensor(original):
                continue
            current.copy_(original.to(device=current.device, dtype=current.dtype))


def _collect_runtime_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            from app.ml.device import release_mps_cache

            release_mps_cache()
    except ImportError:
        pass


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_jsonable(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return str(value)
