"""Read-only discovery using the same module support rules as LoRA patch saving."""
from collections import Counter, defaultdict
from typing import Any


def lora_module_support(model: Any) -> dict[str, str | None]:
    """Map projection paths to an exclusion reason, or None when editable."""
    import torch
    from transformers.pytorch_utils import Conv1D

    parameters = dict(model.named_parameters(remove_duplicate=False))
    occurrences = Counter(id(parameter) for parameter in parameters.values())
    support = {}
    for name, module in model.named_modules():
        if not name or not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        parameter = parameters.get(f"{name}.weight")
        reason = None
        if parameter is None:
            reason = "Weight is not a directly editable parameter."
        elif not parameter.is_floating_point() or parameter.ndim != 2 or parameter.__class__ is not torch.nn.Parameter:
            reason = "This weight format is not supported by the workbench LoRA editor."
        elif occurrences[id(parameter)] > 1:
            reason = "Shares weights with another module; reversible LoRA patches require untied weights."
        support[name] = reason
    return support


def inspect_lora_modules(model: Any, model_id: str) -> dict:
    from app.ml.editing.easyedit_runtime import EasyEditUnsupportedError

    if getattr(model, "peft_config", None):
        raise EasyEditUnsupportedError("Load a merged base model before inspecting workbench LoRA targets.")
    modules = dict(model.named_modules())
    support = lora_module_support(model)
    groups = defaultdict(list)
    for path, reason in support.items():
        module = modules[path]
        groups[path.rsplit(".", 1)[-1]].append({
            "path": path,
            "type": type(module).__name__,
            "weight_shape": list(module.weight.shape),
            "available": reason is None,
            "reason": reason,
        })
    result = []
    for name, entries in sorted(groups.items()):
        available = [entry["path"] for entry in entries if entry["available"]]
        matches = {path for path in modules if path == name or path.endswith(f".{name}")}
        # A suffix must not accidentally include an unsupported or tied module.
        targets = [name] if available and matches == set(available) else available
        result.append({"name": name, "targets": targets, "modules": entries})
    return {"model_id": model_id, "groups": result, "module_count": sum(reason is None for reason in support.values())}


def get_lora_modules(model_id: str) -> dict:
    from app.services.model_runtime_service import model_runtime_service

    if getattr(model_runtime_service, "uses_hard_worker", False):
        return model_runtime_service.execute("lora_modules", {"model_id": model_id}, timeout=120.0)
    with model_runtime_service.exclusive_runtime_access():
        bundle = model_runtime_service.require_loaded_runtime_bundle(model_id)
        return inspect_lora_modules(bundle["model"], model_id)
