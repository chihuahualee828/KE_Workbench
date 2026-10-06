"""Editable, method-specific EasyEdit training options."""
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]


class TrainingHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Positive | None = None
    num_steps: int | None = Field(default=None, ge=1, strict=True)
    weight_decay: NonNegative | None = None
    objective_optimization: Literal["prompt_last", "target_new"] | None = None
    norm_constraint: NonNegative | Literal[False] | None = None
    target_modules: list[str] | None = Field(default=None, min_length=1)
    rank: int | None = Field(default=None, ge=1, strict=True)
    lora_alpha: Positive | None = None
    lora_dropout: float | None = Field(default=None, ge=0, lt=1, allow_inf_nan=False, strict=True)

    @field_validator("target_modules")
    @classmethod
    def validate_modules(cls, values):
        if values is None:
            return values
        values = [value.strip() for value in values]
        if any(not value or value == "all-linear" for value in values):
            raise ValueError("Specify individual target module names, for example q_proj, v_proj.")
        return list(dict.fromkeys(values))


COMMON_FIELDS = {"lr", "num_steps", "weight_decay"}
METHOD_FIELDS = {
    "ft": COMMON_FIELDS | {"objective_optimization", "norm_constraint"},
    "lora": COMMON_FIELDS | {"target_modules", "rank", "lora_alpha", "lora_dropout"},
}


def validate_training_hparams(method: str, values: dict | TrainingHyperparameters | None) -> dict:
    if method not in METHOD_FIELDS:
        raise ValueError("Training method must be ft or lora.")
    parsed = values if isinstance(values, TrainingHyperparameters) else TrainingHyperparameters.model_validate(values or {})
    overrides = parsed.model_dump(exclude_none=True)
    invalid = overrides.keys() - METHOD_FIELDS[method]
    if invalid:
        raise ValueError(f"Unsupported {method.upper()} hyperparameters: {', '.join(sorted(invalid))}.")
    return overrides


def get_training_defaults(model_id: str, method: str) -> dict:
    from app.ml.editing import easyedit_runtime as runtime

    validate_training_hparams(method, {})
    spec = runtime.METHODS[method]
    info = runtime.model_registry.get(model_id)
    if not info.get(spec["support_key"]):
        raise runtime.EasyEditUnsupportedError(f"{model_id} is not configured for {method.upper()}.")
    path = runtime._resolve_hparams_path(info, spec["hparams_key"])
    try:
        payload = yaml.safe_load(path.read_text())
        values = {key: payload[key] for key in METHOD_FIELDS[method]}
        # YAML scientific notation may be parsed as text, as in EasyEdit's own loader.
        for key in ("lr", "weight_decay", "norm_constraint", "lora_alpha", "lora_dropout"):
            if key in values and values[key] is not False:
                values[key] = float(values[key])
        # Show invalid YAML values too, so the user can correct them with an override.
        # The complete effective configuration is validated again before editing.
        warning = None
        try:
            validate_training_hparams(method, values)
        except ValidationError as exc:
            warning = "Check YAML defaults: " + "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
        return {"layers": payload["layers"], "hparams": values, "warning": warning}
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise runtime.EasyEditRuntimeError(f"Invalid training hparams YAML: {path}") from exc
