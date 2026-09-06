from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from app.config import settings
from app.services.model_registry import model_registry
from app.services.model_runtime_service import model_runtime_service
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


class EasyEditRuntimeError(RuntimeError):
    pass


class EasyEditUnsupportedError(ValueError):
    pass


METHODS = {
    "rome": {
        "hparams_class": "ROMEHyperParams",
        "hparams_key": "rome_hparams",
        "support_key": "supports_rome",
    },
    "memit": {
        "hparams_class": "MEMITHyperParams",
        "hparams_key": "memit_hparams",
        "support_key": "supports_memit",
    },
    "kn": {
        "hparams_class": "KNHyperParams",
        "hparams_key": "kn_hparams",
        "support_key": "supports_kn",
    },
}


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
    if not layers:
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
) -> dict[str, Any]:
    method_key = method.lower()
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
    print(f"Loaded EasyEdit hparams from {hparams_path}.")

    _override_model_name(hparams, model)
    _override_layers(hparams, layers)
    _override_mom2_adjustment(hparams, mom2_adjustment)
    _configure_stats_dir(hparams)
    easyedit_model_name = hparams.model_name
    effective_layers = _hparams_layers(hparams)
    print(f"{method_key.upper()} hparams: {hparams}")

    prompts = [build_prompt(edit["subject"], edit["relation_prompt"]) for edit in edits]
    target_new = [edit["new_object"] for edit in edits]
    ground_truth = [edit.get("ground_truth") for edit in edits]
    if not any(ground_truth):
        ground_truth = None
    editor = None
    edited_model = None
    baseline_weights = None
    metrics = None
    used_loaded_runtime = False
    weights_restored = False

    with model_runtime_service.exclusive_runtime_access():
        active_patch_ids = patch_service.active_patches_for_model(model_id)
        runtime_bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
        used_loaded_runtime = True
        if not sequential_edit:
            patch_service.sync_patch_stack_to_model(runtime_bundle["model"], model_id, [])
        _configure_loaded_tokenizer(runtime_bundle["tokenizer"], str(easyedit_model_name), method_key)
        _override_easyedit_device(hparams, runtime_bundle.get("device"))
        hparams.model_name = (runtime_bundle["model"], runtime_bundle["tokenizer"])

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

            # EasyEdit restores singleton edits before returning when sequential_edit=False.
            # We keep the edited weights briefly, save our delta, then restore the baseline
            # ourselves so workbench patch state remains explicit and replayable.
            easyedit_sequential = True
            if hparams_cls.__name__ == "ROMEHyperParams" or hparams_cls.__name__ == "MEMITHyperParams":
                metrics, edited_model, _ = editor.edit(
                    prompts=prompts,
                    subject=[edit["subject"] for edit in edits],
                    ground_truth=ground_truth,
                    target_new=target_new,
                    rephrase_prompts=rephrase_prompts or None,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                    sequential_edit=easyedit_sequential,
                    keep_original_weight=True,
                )
            else:
                metrics, edited_model, _ = editor.edit(
                    prompts=prompts,
                    ground_truth=ground_truth,
                    target_new=target_new,
                    rephrase_prompts=rephrase_prompts or None,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                    sequential_edit=easyedit_sequential,
                    keep_original_weight=True,
                )
            patch_id = patch_service.new_patch_id()
            tensor_delta_paths = _save_weight_deltas(edited_model, baseline_weights, patch_id)
            if not tensor_delta_paths:
                raise EasyEditRuntimeError(
                    "EasyEdit completed, but no changed tensor weights were captured. "
                    "No patch was created because generation/evaluation would stay unchanged."
                )
            _restore_weights(edited_model, baseline_weights)
            weights_restored = True
            patch = patch_service.create_patch(
                model_id=model_id,
                method=method_key,
                edits=edits,
                modified_layers=effective_layers,
                metrics=_jsonable(metrics),
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
                },
                patch_id=patch_id,
            )
            patch_service.activate_patch(patch_id, append=sequential_edit)
            if used_loaded_runtime:
                patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
            local_store.write_json(
                f"patches/{patch['patch_id']}/easyedit_metrics.json",
                {"metrics": _jsonable(metrics)},
            )
            return {
                "patch_id": patch["patch_id"],
                "metrics": _jsonable(metrics),
                "patch": patch,
                "runtime": "easyedit",
                "delta_count": len(tensor_delta_paths),
                "used_loaded_runtime": used_loaded_runtime,
            }
        finally:
            if used_loaded_runtime and baseline_weights and edited_model is not None and not weights_restored:
                _restore_weights(edited_model, baseline_weights)
            if used_loaded_runtime:
                hparams.model_name = easyedit_model_name
                patch_service.sync_active_patches_to_model(runtime_bundle["model"], model_id)
            editor = None
            edited_model = None
            baseline_weights = None
            metrics = None
            _collect_runtime_memory()


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

    with model_runtime_service.exclusive_runtime_access():
        active_patch_ids = patch_service.active_patches_for_model(model_id)
        runtime_bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
        runtime_model = runtime_bundle["model"]
        runtime_tokenizer = runtime_bundle["tokenizer"]

        try:
            if not sequential_edit:
                patch_service.sync_patch_stack_to_model(runtime_model, model_id, [])
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
            metrics, _ = kn.edit_knowledge(
                prompt,
                target=str(edit["new_object"]),
                neurons=refined_neurons,
                undo_modification=False,
            )
            patch_id = patch_service.new_patch_id()
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
    try:
        from easyeditor.models.rome import layer_stats as layer_stats_module
    except ImportError:
        return

    if getattr(layer_stats_module, "_workbench_single_process_loader", False):
        return

    original_tally = layer_stats_module.tally

    def single_process_tally(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original_tally(*args, **kwargs)

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


def _override_easyedit_device(hparams: Any, loaded_device: str | None = None) -> None:
    if not hasattr(hparams, "device"):
        return
    if loaded_device and loaded_device.startswith("cuda"):
        setattr(hparams, "device", _cuda_device_index(loaded_device))
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
    if method_key in {"rome", "memit"} and any(name in model_name.lower() for name in ("mistral", "llama", "qwen")):
        tokenizer.padding_side = "right"


def _override_layers(hparams: Any, layers: list[int] | None) -> None:
    if not layers:
        return
    if hasattr(hparams, "layers"):
        setattr(hparams, "layers", layers)
    if hasattr(hparams, "layer"):
        setattr(hparams, "layer", layers[0])


def get_mom2_statistics_status(
    model_id: str,
    method: str,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Report the MOM2 cache files EasyEdit will use for this runtime model."""
    hparams, effective_layers = _load_mom2_cache_profile(model_id, method, layers)

    statistics = []
    for layer in effective_layers:
        path = _mom2_statistics_path(hparams, None, layer)
        statistics.append({"layer": layer, "path": str(path), "ready": path.exists()})
    missing_layers = [entry["layer"] for entry in statistics if not entry["ready"]]
    return {
        "model_id": model_id,
        "method": method.lower(),
        "mom2_adjustment": True,
        "dataset": str(getattr(hparams, "mom2_dataset")),
        "layers": effective_layers,
        "stats_dir": str(getattr(hparams, "stats_dir")),
        "ready": not missing_layers,
        "missing_layers": missing_layers,
        "statistics": statistics,
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


def prepare_mom2_statistics(model_id: str, method: str, layers: list[int] | None = None) -> dict[str, Any]:
    """Compute the missing EasyEdit MOM2 covariance caches for selected layers."""
    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(
            "prepare_mom2",
            {"model_id": model_id, "method": method, "layers": layers},
            timeout=7200.0,
        )
    hparams, effective_layers = _load_edit_hparams(model_id, method, layers)
    _override_mom2_adjustment(hparams, True)

    with model_runtime_service.exclusive_runtime_access():
        runtime_bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
        _configure_loaded_tokenizer(runtime_bundle["tokenizer"], str(hparams.model_name), method.lower())
        _override_easyedit_device(hparams, runtime_bundle.get("device"))

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

    return get_mom2_statistics_status(model_id, method, effective_layers)


def import_mom2_statistics(
    model_id: str,
    method: str,
    dataset: str,
    layers: list[int] | None,
    files: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Validate and atomically install precomputed MOM2 cache files."""
    hparams, effective_layers = _load_mom2_cache_profile(model_id, method, layers)
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
    for uploaded_name, uploaded_file in files:
        filename = Path(uploaded_name).name
        if Path(filename).suffix.lower() != ".npz":
            raise EasyEditRuntimeError(f"{filename!r} is not an NPZ statistics file.")
        expected = expected_files.get(filename)
        if expected is None:
            raise EasyEditRuntimeError(
                f"{filename!r} does not match a selected {method.upper()} layer for {model_id}."
            )
        layer, destination = expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as temporary:
                temporary_path = temporary.name
                shutil.copyfileobj(uploaded_file, temporary)
            _validate_mom2_archive(Path(temporary_path), getattr(hparams, "mom2_n_samples"))
            os.replace(temporary_path, destination)
            temporary_path = None
            imported_layers.append(layer)
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    status = get_mom2_statistics_status(model_id, method, effective_layers)
    status["imported_layers"] = sorted(set(imported_layers))
    return status


def list_mom2_statistics(model_id: str, method: str) -> dict[str, Any]:
    """List all cached second-moment files that match a model/method hparams profile."""
    hparams, _ = _load_mom2_cache_profile(model_id, method, None)
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
            }
        )
    return {
        "model_id": model_id,
        "method": method.lower(),
        "dataset": str(getattr(hparams, "mom2_dataset")),
        "stats_dir": str(getattr(hparams, "stats_dir")),
        "statistics": statistics,
    }


def delete_mom2_statistics(model_id: str, method: str, layers: list[int]) -> dict[str, Any]:
    if not layers:
        raise EasyEditRuntimeError("Select one or more second-moment statistics files to delete.")
    hparams, _ = _load_mom2_cache_profile(model_id, method, None)
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
    result = list_mom2_statistics(model_id, method)
    result["deleted_layers"] = deleted_layers
    return result


def _load_mom2_cache_profile(
    model_id: str,
    method: str,
    layers: list[int] | None,
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

    return (
        SimpleNamespace(
            model_name=model.get("local_path") or model["hf_id"],
            stats_dir=str(_mom2_stats_dir()),
            rewrite_module_tmp=str(payload["rewrite_module_tmp"]),
            mom2_dataset=str(payload["mom2_dataset"]),
            mom2_dtype=str(payload["mom2_dtype"]),
            mom2_n_samples=payload.get("mom2_n_samples"),
            layers=effective_layers,
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


def _validate_mom2_archive(path: Path, expected_sample_size: int | None) -> None:
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
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise EasyEditRuntimeError("Statistics archive has an invalid second-moment matrix.")
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
    save_file(deltas, str(path), metadata={"format": "easyedit_delta"})
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
