from __future__ import annotations

import json
import sys
from pathlib import Path
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
    print(f"Original hparams: {hparams}")
    _override_model_name(hparams, model)
    _override_layers(hparams, layers)
    easyedit_model_name = hparams.model_name
    effective_layers = _hparams_layers(hparams)

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

    hparams_cls = getattr(easyeditor, hparams_class_name, None)
    if hparams_cls is None:
        raise EasyEditRuntimeError(
            f"Installed EasyEdit does not expose {hparams_class_name}. "
            "Update EasyEdit from https://github.com/zjunlp/EasyEdit."
        )
    return BaseEditor, hparams_cls


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

    try:
        import easyeditor

        package_root = Path(easyeditor.__file__).resolve().parent.parent
        candidates.append(package_root / relative)
    except ImportError:
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

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


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _snapshot_rewrite_weights(model: Any, hparams: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise EasyEditRuntimeError("Install torch to snapshot EasyEdit weights.") from exc

    rewrite_module_tmp = getattr(hparams, "rewrite_module_tmp", None)
    if not rewrite_module_tmp:
        return {}

    state_dict = _unwrap_model(model).state_dict()
    weights = {}
    for layer in _hparams_layers(hparams):
        name = f"{rewrite_module_tmp.format(layer)}.weight"
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

    model = _unwrap_model(edited_model)
    state_dict = model.state_dict()
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

    model = _unwrap_model(edited_model)
    state_dict = model.state_dict()
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
