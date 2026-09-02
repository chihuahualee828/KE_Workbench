from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model_id: str = "gpt2-xl"
    prompt: str
    max_new_tokens: int = Field(default=20, ge=1, le=128)
    decoding_strategy: Literal["greedy", "sampling", "beam"] = "greedy"
    temperature: float = Field(default=0.7, gt=0.0, le=5.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    num_beams: int = Field(default=4, ge=1, le=16)
    patch_id: str | None = None


class GenerateResponse(BaseModel):
    experiment_id: str
    text: str


class LogitLensRequest(BaseModel):
    model_id: str = "gpt2-xl"
    prompt: str
    target_token: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class LogitLensLayer(BaseModel):
    layer: int
    target_rank: int | None = None
    target_logit: float | None = None
    target_probability: float | None = None
    top_tokens: list[tuple[str, float]]


class LogitLensResponse(BaseModel):
    experiment_id: str
    layers: list[LogitLensLayer]


class CorruptionConfig(BaseModel):
    type: str = "gaussian_noise"
    noise_level: float = Field(default=3.0, ge=0.0, le=20.0)


class CausalTraceRequest(BaseModel):
    model_id: str = "gpt2-xl"
    prompt: str
    subject: str
    target: str
    component: Literal["residual", "mlp", "attention"] = "mlp"
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)


class RomeEditRequest(BaseModel):
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


class KnEditRequest(BaseModel):
    model_id: str = "gpt2-xl"
    subject: str
    relation_prompt: str
    new_object: str
    ground_truth: str = Field(min_length=1)
    evaluation_prompts: list[str] = Field(default_factory=list)
    sequential_edit: bool = False


class EditSpec(BaseModel):
    subject: str
    relation_prompt: str
    new_object: str
    ground_truth: str | None = None
    rephrase_prompts: list[str] = Field(default_factory=list)


class MemitEditRequest(BaseModel):
    model_id: str = "gpt2-xl"
    edits: list[EditSpec]
    layers: list[int] | None = None
    evaluation_prompts: list[str] = Field(default_factory=list)
    locality_inputs: dict[str, dict[str, list[str]]] | None = None
    portability_inputs: dict[str, dict[str, list[str]]] | None = None
    sequential_edit: bool = False
    mom2_adjustment: Literal[True] = True


class Mom2StatisticsRequest(BaseModel):
    model_id: str
    method: Literal["rome", "memit"]
    layers: list[int] | None = None


class EvaluationRequest(BaseModel):
    model_id: str = "gpt2-xl"
    patch_id: str
    prompts: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    experiment_id: str
    status: str
    progress: float = 0.0
    logs: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None


class PatchListResponse(BaseModel):
    patches: list[dict[str, Any]]
