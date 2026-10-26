from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ml.editing.methods import EditMethod, Mom2Method
from app.ml.editing.training_hparams import TrainingHyperparameters
from app.ml.editing.random_state import DEFAULT_EDIT_SEED, MAX_EDIT_SEED


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


class EditJobRequest(ApiSchema):
    seed: int = Field(default=DEFAULT_EDIT_SEED, ge=0, le=MAX_EDIT_SEED, strict=True)


class RankOneEditRequest(EditJobRequest):
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


class KnEditRequest(EditJobRequest):
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


class RankOneSeriesRequest(EditJobRequest):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec] = Field(min_length=1)
    target_layer: int = Field(ge=0)
    evaluation_prompts: list[str] = Field(default_factory=list)
    sequential_edit: bool = True
    mom2_adjustment: bool = False


class BatchWeightEditRequest(EditJobRequest):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec] = Field(min_length=1)
    batch_size: int = Field(default=1, ge=1, strict=True)
    layers: list[int] | None = None
    evaluation_prompts: list[str] = Field(default_factory=list)
    locality_inputs: dict[str, dict[str, list[str]]] | None = None
    portability_inputs: dict[str, dict[str, list[str]]] | None = None
    sequential_edit: bool = False
    mom2_adjustment: Literal[True] = True


class TrainingEditRequest(EditJobRequest):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec] = Field(min_length=1)
    hparams: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    batch_size: int = Field(default=1, ge=1, strict=True)
    layers: list[int] | None = None
    evaluation_prompts: list[str] = Field(default_factory=list)
    locality_inputs: dict[str, dict[str, list[str]]] | None = None
    portability_inputs: dict[str, dict[str, list[str]]] | None = None
    sequential_edit: bool = False


class DpoEditSpec(EditSpec):
    target_neg: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_preference(self):
        if not self.target_neg.strip() or not self.new_object.strip():
            raise ValueError("DPO needs a desired answer and a rejected answer for every fact.")
        if self.target_neg.strip() == self.new_object.strip():
            raise ValueError("DPO desired and rejected answers must differ.")
        return self


class DpoEditRequest(TrainingEditRequest):
    edits: list[DpoEditSpec] = Field(min_length=1)


class Mom2StatisticsRequest(ApiSchema):
    model_id: str
    method: Mom2Method
    layers: list[int] | None = None
    stats_source: Literal["base", "model"] | None = None
    mom2_dtype: Literal["float16", "float32", "float64"] | None = None


class Mom2StatisticsPrecisionRequest(ApiSchema):
    model_id: str
    method: Mom2Method
    mom2_dtype: Literal["float16", "float32", "float64"] | None = None


class Mom2StatisticsSourceRequest(ApiSchema):
    model_id: str
    stats_source: Literal["base", "model"]


class EvaluationRequest(ApiSchema):
    model_id: str = "gpt2-xl"
    patch_id: str
    prompts: list[str] = Field(default_factory=list)


EvaluationMetric = Literal["reliability", "generalization", "locality", "portability"]
EvaluationProtocol = Literal["traditional", "LLM-judge", "generate-text"]


class EvaluationProtocolsRequest(ApiSchema):
    evaluation_protocols: dict[EvaluationMetric, EvaluationProtocol] = Field(default_factory=dict)

    @field_validator("evaluation_protocols")
    @classmethod
    def validate_evaluation_protocols(
        cls, protocols: dict[EvaluationMetric, EvaluationProtocol]
    ) -> dict[EvaluationMetric, EvaluationProtocol]:
        if protocols.get("locality") == "LLM-judge":
            raise ValueError("Locality does not support the LLM-judge evaluation protocol.")
        if any(metric != "locality" and protocol == "generate-text" for metric, protocol in protocols.items()):
            raise ValueError("Only locality supports the generate-text preservation protocol.")
        return protocols


class ReliabilityEvaluationRequest(EvaluationProtocolsRequest):
    """Evaluate edit quality on the current active model without editing it."""

    model_id: str = "gpt2-xl"
    method: EditMethod
    edits: list[EditSpec] = Field(min_length=1)
    include_generations: bool = False
    evaluation_modes: list[EvaluationMetric] = Field(default_factory=list)


class UploadedEvaluationFact(ApiSchema):
    """A normalized uploaded fact to evaluate against the currently loaded model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    key: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    relation_prompt: str
    new_object: str = Field(min_length=1)
    ground_truth: str = ""
    rephrase_prompt: str | None = None
    locality_prompt: str | None = None
    locality_ground_truth: str | None = None
    portability_prompt: str | None = None
    portability_ground_truth: str | None = None


class ActiveModelEvaluationRequest(EvaluationProtocolsRequest):
    """Evaluate uploaded facts or resolve facts from the active patch stack."""

    model_id: str = "gpt2-xl"
    evaluation_modes: list[EvaluationMetric] = Field(default_factory=list)
    rephrase_overrides: dict[str, str] = Field(default_factory=dict)
    include_generations: bool = True
    fluency_modes: list[EvaluationMetric] = Field(default_factory=list)
    facts: list[UploadedEvaluationFact] | None = Field(default=None, min_length=1)
    fact_source_id: str | None = None


class JobResponse(ApiSchema):
    job_id: str
    experiment_id: str
    status: str
    cancel_requested: bool = False
    progress: float = 0.0
    progress_current: int | None = None
    progress_total: int | None = None
    progress_phase: str | None = None
    patch_state: dict[str, Any] | None = None
    logs: list[str] = Field(default_factory=list)
    log_count: int = 0
    log_start_count: int = 0
    result_item_count: int = 0
    result_page_size: int | None = None
    result: dict[str, Any] | None = None


class PatchListResponse(ApiSchema):
    patches: list[dict[str, Any]]
