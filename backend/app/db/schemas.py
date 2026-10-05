from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiSchema(BaseModel):
    """Base schema for API payloads with model identifier fields."""

    model_config = ConfigDict(protected_namespaces=("model_validate", "model_dump"))


class GenerateRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    prompt: str
    max_new_tokens: int = Field(default=20, ge=1, le=128)
    decoding_strategy: Literal["greedy", "sampling", "beam"] = "greedy"
    temperature: float = Field(default=0.7, gt=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    num_beams: int = Field(default=4, ge=1, le=16)
    patch_id: str | None = None


class GenerateResponse(ApiSchema):
    experiment_id: str
    text: str


class LogitLensRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    prompt: str
    target_token: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class LogitLensLayer(ApiSchema):
    layer: int
    target_rank: int | None = None
    target_logit: float | None = None
    target_probability: float | None = None
    top_tokens: list[tuple[str, float]]


class LogitLensResponse(ApiSchema):
    experiment_id: str
    layers: list[LogitLensLayer]


class CorruptionConfig(ApiSchema):
    type: str = "gaussian_noise"
    noise_level: float = Field(default=3.0, ge=0.0, le=20.0)


class CausalTraceRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    prompt: str
    subject: str
    target: str
    component: Literal["residual", "mlp", "attention"] = "mlp"
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)
    samples: int = Field(default=10, ge=1, le=32)
    seed: int = Field(default=1, ge=0, le=2_147_483_647)


class RomeEditRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    subject: str
    relation_prompt: str
    new_object: str
    ground_truth: str | None = None
    target_layer: int | None = Field(default=None, ge=0)
    evaluation_prompts: list[str] = Field(default_factory=list)
    rephrase_prompts: list[str] = Field(default_factory=list)
    locality_inputs: dict[str, dict[str, list[str]]] | None = None
    portability_inputs: dict[str, dict[str, list[str]]] | None = None
    sequential_edit: bool = False
    mom2_adjustment: bool = False


class RRomeEditRequest(RomeEditRequest):
    """Request schema for EasyEdit's rebuilding-ROME implementation."""


class KnEditRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    subject: str
    relation_prompt: str
    new_object: str
    ground_truth: str = Field(min_length=1)
    evaluation_prompts: list[str] = Field(default_factory=list)
    sequential_edit: bool = False


class EditSpec(ApiSchema):
    subject: str
    relation_prompt: str
    new_object: str
    ground_truth: str | None = None
    rephrase_prompts: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class RomeSequentialEditsRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec] = Field(min_length=1)
    target_layer: int = Field(ge=0)
    evaluation_prompts: list[str] = Field(default_factory=list)
    sequential_edit: bool = True
    mom2_adjustment: bool = False


class MemitEditRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec] = Field(min_length=1)
    batch_size: int = Field(default=1, ge=1, strict=True)
    layers: list[int] | None = None
    evaluation_prompts: list[str] = Field(default_factory=list)
    locality_inputs: dict[str, dict[str, list[str]]] | None = None
    portability_inputs: dict[str, dict[str, list[str]]] | None = None
    sequential_edit: bool = False
    mom2_adjustment: Literal[True] = True


class Mom2StatisticsRequest(ApiSchema):
    model_id: str
    method: Literal["rome", "r_rome", "memit"]
    layers: list[int] | None = None
    stats_source: Literal["base", "model"] | None = None
    mom2_dtype: Literal["float16", "float32", "float64"] | None = None


class Mom2StatisticsPrecisionRequest(ApiSchema):
    model_id: str
    method: Literal["rome", "r_rome", "memit"]
    mom2_dtype: Literal["float16", "float32", "float64"] | None = None


class Mom2StatisticsSourceRequest(ApiSchema):
    model_id: str
    stats_source: Literal["base", "model"]


class EvaluationRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    patch_id: str
    prompts: list[str] = Field(default_factory=list)


class ReliabilityEvaluationRequest(ApiSchema):
    """Evaluate edit quality on the current active model without editing it."""

    model_id: str = "gpt2-xl"
    method: Literal["rome", "r_rome", "memit", "kn"]
    edits: list[EditSpec] = Field(min_length=1)
    include_generations: bool = False
    evaluation_modes: list[Literal["reliability", "generalization", "locality", "portability"]] = Field(default_factory=list)


class ActiveModelEvaluationRequest(ApiSchema):
    """Evaluate the edit facts currently active for a model.

    Unlike ``ReliabilityEvaluationRequest``, this deliberately does not receive
    the full edit list from the browser.  The server resolves that list from the
    active patch stack so a large evaluation does not require the page to have
    loaded every fact first.
    """

    model_id: str = "gpt2-xl"
    evaluation_modes: list[Literal["reliability", "generalization", "locality", "portability"]] = Field(default_factory=list)
    rephrase_overrides: dict[str, str] = Field(default_factory=dict)
    include_generations: bool = True


class JobResponse(ApiSchema):
    job_id: str
    experiment_id: str
    status: str
    cancel_requested: bool = False
    progress: float = 0.0
    progress_current: int | None = None
    progress_total: int | None = None
    logs: list[str] = Field(default_factory=list)
    log_count: int = 0
    log_start_count: int = 0
    result_item_count: int = 0
    result_page_size: int | None = None
    result: dict[str, Any] | None = None


class PatchListResponse(ApiSchema):
    patches: list[dict[str, Any]]
