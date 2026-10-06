"""EasyEdit FT/LoRA batches stored using the workbench's reversible patch format."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.ml.editing import easyedit_runtime as runtime
from app.ml.editing.edit_cancellation import check_edit_cancellation, current_edit
from app.ml.editing.memit_wrapper import _single_rephrase_per_edit
from app.services.model_runtime_service import model_runtime_service
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


def run_training_edit(
    method: str,
    model_id: str,
    edits: list[dict],
    layers: list[int] | None = None,
    evaluation_prompts: list[str] | None = None,
    locality_inputs: dict | None = None,
    portability_inputs: dict | None = None,
    sequential_edit: bool = False,
    batch_size: int = 1,
) -> dict:
    if method not in {"ft", "lora"}:
        raise ValueError("Training method must be ft or lora.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    edits = [edit.model_dump() if hasattr(edit, "model_dump") else dict(edit) for edit in edits]
    if not edits or any(not str(edit.get(key) or "").strip() for edit in edits
                        for key in ("subject", "relation_prompt", "new_object")):
        raise ValueError("Provide edit facts with a subject, relation prompt, and new object.")
    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute(f"{method}_edit", {
            "model_id": model_id, "edits": edits, "layers": layers,
            "evaluation_prompts": evaluation_prompts, "locality_inputs": locality_inputs,
            "portability_inputs": portability_inputs, "sequential_edit": sequential_edit,
            "batch_size": batch_size,
        }, timeout=7200.0)

    spec = runtime.METHODS[method]
    info = runtime.model_registry.get(model_id)
    if not info.get(spec["support_key"]):
        raise runtime.EasyEditUnsupportedError(f"{model_id} is not configured for {method.upper()}.")
    BaseEditor, hparams_cls = runtime._load_easyedit(spec["hparams_class"])
    path = runtime._resolve_hparams_path(info, spec["hparams_key"])
    hparams = hparams_cls.from_hparams(str(path))
    runtime._override_model_name(hparams, info)
    if layers is not None:
        if (not layers and method == "ft") or any(type(layer) is not int or not 0 <= layer < info["n_layers"] for layer in layers):
            raise ValueError(f"Edit layers must be integers from 0 to {info['n_layers'] - 1}.")
        hparams.layers = layers
    hparams.batch_size = batch_size
    hparams.model_parallel = False  # Reuse the already placed runtime model.
    if method == "lora":
        # Several upstream LoRA profiles default to AdaLoRA; this tab is standard LoRA.
        hparams.lora_type = "lora"
    model_name = hparams.model_name
    with runtime._easyedit_runtime_access(model_id) as bundle:
        model, tokenizer = bundle["model"], bundle["tokenizer"]
        names, target_modules = _training_targets(model, hparams, method)
        if not names:
            raise runtime.EasyEditRuntimeError("No parameters match the EasyEdit training configuration.")
        transaction = current_edit.get()
        if transaction is not None:
            transaction.capture(model, model_id, names)
        if not sequential_edit:
            patch_service.deactivate_runtime_patches_for_model(model_id)
            patch_service.sync_active_patches_to_model(model, model_id)
        parents = patch_service.active_patches_for_model(model_id)
        state = model.state_dict()
        baseline = {name: state[name].detach().cpu().clone() for name in names}
        del state
        try:
            with _training_state(model, tokenizer, target_modules):
                runtime._configure_loaded_tokenizer(tokenizer, str(model_name), method)
                # EasyEdit FT prompt_last indexes by mask length; target_new/LoRA mask left pads.
                tokenizer.padding_side = "right" if method == "ft" and hparams.objective_optimization == "prompt_last" else "left"
                runtime._override_easyedit_device(hparams, bundle.get("device"))
                hparams.model_name = (model, tokenizer)
                editor = BaseEditor.from_hparams(hparams)
                hparams.model_name = model_name
                editor.model_name = model_name
                print(f"{method.upper()} batch size: {batch_size}; facts: {len(edits)}.")
                print("Starting EasyEdit algorithm.")
                metrics, edited_model, _ = editor.batch_edit(
                    prompts=[runtime.build_prompt(edit["subject"], edit["relation_prompt"]) for edit in edits],
                    target_new=[edit["new_object"] for edit in edits],
                    ground_truth=[edit.get("ground_truth") for edit in edits],
                    rephrase_prompts=_single_rephrase_per_edit(edits),
                    locality_inputs=locality_inputs, portability_inputs=portability_inputs,
                    sequential_edit=True,
                )
                check_edit_cancellation()
                if method == "lora":
                    # Fold the trained adapters into original parameter names for replay/export.
                    edited_model = edited_model.merge_and_unload(safe_merge=True)
                patch_id = patch_service.new_patch_id()
                if transaction is not None:
                    transaction.protect_patch(patch_id)
                delta_paths = runtime._save_weight_deltas(edited_model, baseline, patch_id)
                if not delta_paths:
                    raise runtime.EasyEditRuntimeError("EasyEdit completed without changing any weights; no patch was created.")
            runtime._restore_weights(model, baseline)
            patch = patch_service.create_patch(
                model_id=model_id, method=method, edits=edits,
                modified_layers=runtime._hparams_layers(hparams) or list(range(info["n_layers"])),
                metrics=runtime._jsonable(metrics), delta_paths=delta_paths, runtime="easyedit",
                extra_metadata={
                    "easyedit_hparams": str(path), "hf_model": info["hf_id"],
                    "batch_size": batch_size, "sequential_edit": sequential_edit,
                    "parent_patch_ids": parents if sequential_edit else [],
                    "used_loaded_runtime": True,
                    "configured_layers": list(hparams.layers),
                    **({"lora_type": "lora", "merged_lora": True} if method == "lora" else {}),
                }, patch_id=patch_id,
            )
            patch_service.activate_patch(patch_id, append=sequential_edit)
            local_store.write_json(f"patches/{patch_id}/easyedit_metrics.json", {"metrics": runtime._jsonable(metrics)})
        finally:
            runtime._restore_weights(model, baseline)
            patch_service.sync_active_patches_to_model(model, model_id)
            runtime._collect_runtime_memory()
    result = {
        "patch_id": patch_id, "patch": patch, "metrics": runtime._jsonable(metrics),
        "runtime": "easyedit", "delta_count": len(delta_paths),
        "used_loaded_runtime": True, "edited_facts": len(edits),
    }
    result.update(runtime.evaluate_post_edit_reliability(model_id, method, edits))
    return result


def _training_targets(model: Any, hparams: Any, method: str) -> tuple[list[str], dict]:
    if method == "ft":
        # EasyEdit FT selects every parameter containing the configured module path,
        # including biases (ROME/MEMIT's weight-only snapshot is insufficient).
        return [name for name, _ in model.named_parameters() if any(
            hparams.rewrite_module_tmp.format(layer) in name for layer in hparams.layers
        )], {}
    from peft import LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists

    if getattr(model, "peft_config", None):
        raise runtime.EasyEditUnsupportedError("Load a merged base model before applying a workbench LoRA edit.")
    config = LoraConfig(target_modules=hparams.target_modules, layers_to_transform=hparams.layers or None)
    modules = {name: module for name, module in model.named_modules()
               if name and check_target_module_exists(config, name)}
    for target in hparams.target_modules:
        if not any(name == target or name.endswith(f".{target}") for name in modules):
            raise ValueError(f"LoRA target module '{target}' does not match the selected model/layers.")
    names = [f"{name}.weight" for name in modules]
    return names, modules


@contextmanager
def _training_state(model: Any, tokenizer: Any, target_modules: dict):
    """Remove adapters even when EasyEdit raises before returning its PEFT wrapper."""
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    modules = [(module, module.training) for module in model.modules()]
    missing = object()
    attributes = {key: getattr(model, key, missing) for key in (
        "peft_config", "supports_gradient_checkpointing", "is_parallelizable", "model_parallel", "_require_grads_hook",
    )}
    use_cache = getattr(model.config, "use_cache", missing)
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    embedding = model.get_input_embeddings()
    embedding_hooks = set(embedding._forward_hooks)
    padding_side = tokenizer.padding_side
    handle = model.register_forward_pre_hook(lambda *_: check_edit_cancellation(), prepend=True)
    try:
        check_edit_cancellation()
        yield
    finally:
        handle.remove()
        for name, module in target_modules.items():
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, module)
        for key in set(embedding._forward_hooks) - embedding_hooks:
            del embedding._forward_hooks[key]
        if not checkpointing and getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        for key, value in attributes.items():
            if value is missing:
                if hasattr(model, key):
                    delattr(model, key)
            else:
                setattr(model, key, value)
        if use_cache is not missing:
            model.config.use_cache = use_cache
        tokenizer.padding_side = padding_side
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        for module, training in modules:
            module.training = training
