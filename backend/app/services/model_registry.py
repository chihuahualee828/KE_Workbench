from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import settings
from app.ml.editing.methods import METHODS, get_editing_hparams, get_editing_support


# These are the architecture families for which the workbench currently has
# model metadata and an AutoModelForCausalLM runtime path.  Importing is kept
# deliberately narrower than arbitrary Transformers uploads: a model that is
# not one of these families might load, but it cannot be represented safely by
# the workbench's adapters and EasyEdit configuration.
_CONFIG_MODEL_TYPE_ALIASES = {
    "gpt2": "gpt2",
    "gptj": "gptj",
    "baichuan": "baichuan",
    "chatglm": "chatglm",
    "internlm": "internlm",
    "llama": "llama",
    "mistral": "mistral",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2_moe": "qwen",
    "qwen3": "qwen",
    "qwen3_moe": "qwen",
}

_REMOTE_CODE_MODEL_TYPES = {"baichuan", "chatglm", "internlm", "qwen"}


def _hub_model(
    *,
    model_id: str,
    name: str,
    hf_id: str,
    n_layers: int,
    hparams: dict[str, str],
    model_type: str = "auto",
    supports_logit_lens: bool = False,
    supports_causal_trace: bool = False,
    trust_remote_code: bool = False,
    generation_use_cache: bool = True,
    estimated_vram_gb: dict[str, int] | None = None,
) -> dict[str, Any]:
    easyedit = {METHODS[method]["hparams_key"]: path for method, path in hparams.items() if path}
    if estimated_vram_gb:
        easyedit["estimated_vram_gb"] = estimated_vram_gb
    return {
        "id": model_id,
        "name": name,
        "hf_id": hf_id,
        "model_type": model_type,
        "n_layers": n_layers,
        "supports_generation": True,
        "supports_logit_lens": supports_logit_lens,
        "supports_causal_trace": supports_causal_trace,
        "trust_remote_code": trust_remote_code,
        "generation_use_cache": generation_use_cache,
        **{spec["support_key"]: bool(easyedit.get(spec["hparams_key"])) for spec in METHODS.values()},
        "easyedit": easyedit,
        "source": "hub",
    }

SUPPORTED_MODELS = [
    _hub_model(
        model_id="gpt2-xl",
        name="GPT-2 XL",
        hf_id="gpt2-xl",
        model_type="gpt2",
        n_layers=48,
        supports_logit_lens=True,
        supports_causal_trace=True,
        estimated_vram_gb={"rome": 10, "memit": 11},
        hparams={
            "rome": "hparams/ROME/gpt2-xl.yaml",
            "r_rome": "hparams/R-ROME/gpt2-xl.yaml",
            "memit": "hparams/MEMIT/gpt2-xl.yaml",
            "ft": "hparams/FT/gpt2-xl.yaml",
            "lora": "hparams/LoRA/gpt2-xl.yaml",
            "kn": "hparams/KN/gpt2-xl.yaml",
            "emmet": "hparams/EMMET/gpt2-xl.yaml",
            "memit_fe": "hparams/MEMIT-FE/gpt2-xl.yaml",
            "alphaedit": "hparams/AlphaEdit/gpt2-xl.yaml",
            "sphere": "hparams/SPHERE/gpt2-xl.yaml",
        },
    ),
    _hub_model(
        model_id="gpt-j-6b",
        name="GPT-J 6B",
        hf_id="EleutherAI/gpt-j-6B",
        model_type="gptj",
        n_layers=28,
        estimated_vram_gb={"rome": 16, "memit": 18},
        hparams={
            "rome": "hparams/ROME/gpt-j-6B.yaml",
            "r_rome": "hparams/R-ROME/gpt-j-6B.yaml",
            "memit": "hparams/MEMIT/gpt-j-6B.yaml",
            "ft": "hparams/FT/gpt-j-6B.yaml",
            "lora": "hparams/LoRA/gpt-j-6B.yaml",
            "kn": "hparams/KN/gpt-j-6B.yaml",
            "emmet": "hparams/EMMET/gpt-j-6B.yaml",
            "memit_fe": "hparams/MEMIT-FE/gpt-j-6B.yaml",
            "pmet": "hparams/PMET/gpt-j-6B.yaml",
        },
    ),
    _hub_model(
        model_id="baichuan-7b",
        name="Baichuan 2 7B Chat",
        hf_id="baichuan-inc/Baichuan2-7B-Chat",
        model_type="baichuan",
        n_layers=32,
        trust_remote_code=True,
        hparams={
            "rome": "hparams/ROME/baichuan-7b.yaml",
            "memit": "hparams/MEMIT/baichuan-7b.yaml",
            "ft": "hparams/FT/baichuan-7b.yaml",
            "kn": "hparams/KN/baichuan-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/baichuan-7b.yaml",
        },
    ),
    _hub_model(
        model_id="chatglm2-6b",
        name="ChatGLM2 6B",
        hf_id="THUDM/chatglm2-6b",
        model_type="chatglm",
        n_layers=28,
        trust_remote_code=True,
        hparams={
            "rome": "hparams/ROME/chatglm2-6b.yaml",
            "memit": "hparams/MEMIT/chatglm2-6b.yaml",
            "ft": "hparams/FT/chatglm2-6b.yaml",
            "lora": "hparams/LoRA/chatglm2-6b.yaml",
            "kn": "hparams/KN/chatglm2-6b.yaml",
            "memit_fe": "hparams/MEMIT-FE/chatglm2-6b.yaml",
        },
    ),
    _hub_model(
        model_id="chatglm4-9b",
        name="GLM-4 9B Chat",
        hf_id="THUDM/glm-4-9b-chat",
        model_type="chatglm",
        n_layers=40,
        trust_remote_code=True,
        hparams={
            "rome": "hparams/ROME/chatglm4-9b.yaml",
            "r_rome": "hparams/R-ROME/chatglm4-9b.yaml",
            "ft": "hparams/FT/chatglm4-9b.yaml",
            "lora": "hparams/LoRA/chatglm4-9b.yaml",
        },
    ),
    _hub_model(
        model_id="internlm-7b",
        name="InternLM 7B",
        hf_id="internlm/internlm-7b",
        model_type="internlm",
        n_layers=32,
        trust_remote_code=True,
        generation_use_cache=False,
        hparams={
            "rome": "hparams/ROME/internlm-7b.yaml",
            "memit": "hparams/MEMIT/internlm-7b.yaml",
            "ft": "hparams/FT/internlm-7b.yaml",
            "kn": "hparams/KN/internlm-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/internlm-7b.yaml",
        },
    ),
    _hub_model(
        model_id="llama-7b",
        name="Llama 2 7B",
        hf_id="meta-llama/Llama-2-7b-hf",
        model_type="llama",
        n_layers=32,
        hparams={
            "rome": "hparams/ROME/llama-7b.yaml",
            "r_rome": "hparams/R-ROME/llama-7b.yaml",
            "memit": "hparams/MEMIT/llama-7b.yaml",
            "ft": "hparams/FT/llama-7b.yaml",
            "lora": "hparams/LoRA/llama-7b.yaml",
            "kn": "hparams/KN/llama-7b.yaml",
            "emmet": "hparams/EMMET/llama-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/llama-7b.yaml",
            "pmet": "hparams/PMET/llama-7b.yaml",
            "dpo": "hparams/DPO/llama-7b.yaml",
        },
    ),
    _hub_model(
        model_id="llama3-8b",
        name="Llama 3 8B",
        hf_id="meta-llama/Meta-Llama-3-8B",
        model_type="llama",
        n_layers=32,
        estimated_vram_gb={"rome": 18},
        hparams={
            "rome": "hparams/ROME/llama3-8b.yaml",
            "ft": "hparams/FT/llama3-8b.yaml",
            "lora": "hparams/LoRA/llama3-8b.yaml",
            "kn": "hparams/KN/llama3-8b.yaml",
            "memit_fe": "hparams/MEMIT-FE/llama3-8b.yaml",
            "dpo": "hparams/DPO/llama3-8b.yaml",
            "alphaedit": "hparams/AlphaEdit/llama3-8b.yaml",
            "sphere": "hparams/SPHERE/llama3-8b.yaml",
        },
    ),
    _hub_model(
        model_id="llama3.2-3b",
        name="Llama 3.2 3B",
        hf_id="meta-llama/Llama-3.2-3B",
        model_type="llama",
        n_layers=28,
        estimated_vram_gb={"rome": 8, "memit": 9},
        hparams={
            "rome": "hparams/ROME/llama3.2-3b.yaml",
            "r_rome": "hparams/R-ROME/llama3.2-3b.yaml",
            "memit": "hparams/MEMIT/llama3.2-3b.yaml",
            "ft": "hparams/FT/llama3.2-3b.yaml",
            "lora": "hparams/LoRA/llama3.2-3b.yaml",
            "kn": "hparams/KN/llama3.2-3b.yaml",
            "emmet": "hparams/EMMET/llama3.2-3b.yaml",
            "pmet": "hparams/PMET/llama3.2-3b.yaml",
        },
    ),
    _hub_model(
        model_id="mistral-7b",
        name="Mistral 7B",
        hf_id="mistralai/Mistral-7B-v0.1",
        model_type="mistral",
        n_layers=32,
        estimated_vram_gb={"rome": 16, "memit": 18},
        hparams={
            "rome": "hparams/ROME/mistral-7b.yaml",
            "memit": "hparams/MEMIT/mistral-7b.yaml",
            "ft": "hparams/FT/mistral-7b.yaml",
            "kn": "hparams/KN/mistral-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/mistral-7b.yaml",
        },
    ),
    _hub_model(
        model_id="qwen2.5-7b",
        name="Qwen2.5 7B",
        hf_id="Qwen/Qwen2.5-7B",
        model_type="qwen",
        n_layers=28,
        estimated_vram_gb={"rome": 16, "memit": 18},
        hparams={
            "rome": "hparams/ROME/qwen2.5-7b.yaml",
            "r_rome": "hparams/R-ROME/qwen2.5-7b.yaml",
            "memit": "hparams/MEMIT/qwen2.5-7b.yaml",
            "ft": "hparams/FT/qwen2.5-7b.yaml",
            "lora": "hparams/LoRA/qwen2.5-7b.yaml",
            "kn": "hparams/KN/qwen2.5-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/qwen2.5-7b.yaml",
            "eamet": "hparams/EAMET/qwen2.5-7b.yaml",
            "alphaedit": "hparams/AlphaEdit/qwen2.5-7b.yaml",
            "sphere": "hparams/SPHERE/qwen2.5-7b.yaml",
        },
    ),
    _hub_model(
        model_id="qwen2-7b",
        name="Qwen2 7B",
        hf_id="Qwen/Qwen2-7B",
        model_type="qwen",
        n_layers=28,
        estimated_vram_gb={"rome": 16, "memit": 18},
        hparams={
            "rome": "hparams/ROME/qwen2-7b.yaml",
            "memit": "hparams/MEMIT/qwen2-7b.yaml",
            "ft": "hparams/FT/qwen2-7b.yaml",
            "kn": "hparams/KN/qwen2-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/qwen2-7b.yaml",
        },
    ),
    _hub_model(
        model_id="qwen-7b",
        name="Qwen 7B",
        hf_id="Qwen/Qwen-7B",
        model_type="qwen",
        n_layers=32,
        trust_remote_code=True,
        estimated_vram_gb={"rome": 16, "memit": 18},
        hparams={
            "rome": "hparams/ROME/qwen-7b.yaml",
            "memit": "hparams/MEMIT/qwen-7b.yaml",
            "ft": "hparams/FT/qwen-7b.yaml",
            "kn": "hparams/KN/qwen-7b.yaml",
            "memit_fe": "hparams/MEMIT-FE/qwen-7b.yaml",
        },
    ),
    _hub_model(
        model_id="qwen3.5-9b",
        name="Qwen3.5 9B",
        hf_id="Qwen/Qwen3.5-9B",
        model_type="qwen",
        n_layers=32,
        hparams={
            "rome": "hparams/ROME/qwen3.5-9b.yaml",
            "r_rome": "hparams/R-ROME/qwen3.5-9b.yaml",
            "ft": "hparams/FT/qwen3.5-9b.yaml",
        },
    ),
]


class ModelRegistry:
    def list_models(self) -> list[dict]:
        # Registrations can be copied between servers with the data directory.
        # Expose only local models present on this backend, but retain all saved
        # records for registry updates and for when their folders return.
        local_models = [
            model for model in self.list_local_models() if _local_model_is_available(model)
        ]
        return [*SUPPORTED_MODELS, *local_models]

    def get(self, model_id: str) -> dict:
        for model in self.list_models():
            if model["id"] == model_id:
                return model
        raise ValueError(f"Unsupported model: {model_id}")

    def set_mom2_statistics_source(self, model_id: str, stats_source: str) -> dict:
        """Persist the MOM2 cache choice for an imported model.

        A local model that was identified as a copy of a supported hub model can
        either reuse that base model's covariance statistics or own separate
        statistics. Hub models have only their own cache identity.
        """
        if stats_source not in {"base", "model"}:
            raise ValueError("MOM2 statistics source must be 'base' or 'model'.")

        models = self.list_local_models()
        for model in models:
            if model.get("id") != model_id:
                continue
            if model.get("source") != "local":
                raise ValueError("Only imported models can choose a MOM2 statistics source.")
            if stats_source == "base" and not self._has_supported_hub_base(model):
                raise ValueError("This imported model does not have a supported base-model MOM2 cache.")
            model["mom2_stats_source"] = stats_source
            _write_local_models(models)
            return model

        if any(model.get("id") == model_id for model in SUPPORTED_MODELS):
            if stats_source != "model":
                raise ValueError("Hub models use their own MOM2 statistics.")
            return self.get(model_id)
        raise ValueError(f"Unsupported model: {model_id}")

    def _has_supported_hub_base(self, model: dict[str, Any]) -> bool:
        base_model_id = model.get("base_model_id")
        return isinstance(base_model_id, str) and any(
            candidate["id"] == base_model_id and candidate.get("source") == "hub"
            for candidate in SUPPORTED_MODELS
        )

    def list_local_models(self) -> list[dict]:
        models = _read_local_models()
        if _migrate_managed_local_models(models):
            _write_local_models(models)
        for model in models:
            profile = next((item for item in SUPPORTED_MODELS if item["id"] == model.get("base_model_id")), {})
            for spec in METHODS.values():
                support_key, hparams_key = spec["support_key"], spec["hparams_key"]
                model.setdefault(support_key, bool(profile.get(support_key)))
                path = profile.get("easyedit", {}).get(hparams_key)
                if path and model.get(support_key):
                    model.setdefault("easyedit", {}).setdefault(hparams_key, path)
            relative_path = _managed_local_model_relative_path(model)
            if relative_path is not None:
                resolved_path = str(settings.data_dir / relative_path)
                model["local_path"] = resolved_path
                model["hf_id"] = resolved_path
            local_path = model.get("local_path")
            if not isinstance(local_path, str) or not local_path.strip():
                continue
            configured_layers = _architecture_layer_count(Path(local_path))
            if configured_layers is not None:
                model["n_layers"] = configured_layers
        return models

    def add_local_model(self, model: dict[str, Any], *, managed_local: bool = False) -> dict[str, Any]:
        local_path = _resolve_local_model_path(model.get("local_path"))
        try:
            path_exists = local_path.exists()
            has_config = (local_path / "config.json").is_file()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Local model path cannot be accessed: {local_path}") from exc
        if not path_exists:
            raise ValueError(f"Local model path does not exist: {local_path}")
        if not has_config:
            raise ValueError(f"Local model path must contain config.json: {local_path}")
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            model_id = self.allocate_imported_model_id()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", model_id):
            raise ValueError("Model ID may contain only letters, numbers, dot, underscore, colon, and hyphen.")
        if any(item["id"] == model_id for item in SUPPORTED_MODELS):
            raise ValueError(f"{model_id} is a built-in model ID.")
        model_type = (model.get("model_type") or "auto").strip().lower()
        configured_layers = _architecture_layer_count(local_path) or int(model.get("n_layers") or 0)
        gpt2_only_features = [
            label
            for label, enabled in {
                "causal trace": model.get("supports_causal_trace"),
            }.items()
            if enabled
        ]
        if gpt2_only_features and model_type != "gpt2":
            raise ValueError(
                f"{', '.join(gpt2_only_features)} local support is currently limited to GPT-2 architecture models."
            )
        if model.get("supports_causal_trace") and configured_layers <= 0:
            raise ValueError("Causal trace local models need a positive layer count.")
        for method, spec in METHODS.items():
            if model.get(spec["support_key"]) and not model.get(spec["hparams_key"]):
                label = method.upper().replace("_", "-")
                raise ValueError(f"{label} local models need a hparams path.")
        if model.get("supports_kn") and model_type not in {
            "gpt2", "gptj", "llama", "baichuan", "chatglm", "internlm", "mistral", "qwen"
        }:
            raise ValueError("KN local support needs a supported model type, not auto.")

        stored = {
            "id": model_id,
            "name": model.get("name") or model_id,
            "hf_id": str(local_path),
            "local_path": str(local_path),
            "source": "local",
            "managed_local": managed_local,
            "base_model_id": model.get("base_model_id"),
            "mom2_stats_source": model.get("mom2_stats_source"),
            "architecture_model_type": model.get("architecture_model_type"),
            "model_type": model_type,
            "n_layers": configured_layers,
            "supports_generation": bool(model.get("supports_generation", True)),
            "supports_logit_lens": bool(model.get("supports_logit_lens", False)),
            "supports_causal_trace": bool(model.get("supports_causal_trace", False)),
            **get_editing_support(model),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "generation_use_cache": bool(model.get("generation_use_cache", True)),
            **(
                {"import_metadata": dict(model["import_metadata"])}
                if isinstance(model.get("import_metadata"), dict)
                else {}
            ),
            "easyedit": {
                key: value
                for key, value in get_editing_hparams(model).items()
                if value
            },
        }

        models = [item for item in self.list_local_models() if item["id"] != model_id]
        models.append(stored)
        _write_local_models(models)
        try:
            from app.services.patch_service import patch_service

            lineage = patch_service.import_model_lineage(stored)
            stored["lineage_imported"] = lineage["imported"]
        except Exception:
            stored["lineage_imported"] = 0
        return stored

    def inspect_imported_local_model_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Derive the safe workbench metadata for a browser-imported snapshot.

        Capabilities are never supplied by the browser.  Generation is available
        for each supported causal-LM family; inspection and EasyEdit capabilities
        are enabled only where the local config gives us enough evidence that the
        existing implementation/hparams apply.
        """
        if not isinstance(config, dict):
            raise ValueError("config.json must contain a JSON object.")
        if config.get("is_encoder_decoder") is True:
            raise ValueError("Only decoder-only causal language models can be imported.")

        raw_model_type = config.get("model_type")
        if not isinstance(raw_model_type, str) or not raw_model_type.strip():
            raise ValueError("config.json is missing its model_type.")
        raw_model_type = raw_model_type.strip().lower()
        model_type = _CONFIG_MODEL_TYPE_ALIASES.get(raw_model_type)
        if model_type is None:
            supported = ", ".join(sorted(set(_CONFIG_MODEL_TYPE_ALIASES.values())))
            raise ValueError(
                f"Unsupported local model architecture {raw_model_type!r}. "
                f"This workbench currently imports: {supported}."
            )

        architectures = config.get("architectures")
        if architectures is not None:
            if not isinstance(architectures, list) or not all(
                isinstance(item, str) for item in architectures
            ):
                raise ValueError("config.json architectures must be a list of model class names.")
            if architectures and not any(_is_causal_lm_architecture(item) for item in architectures):
                raise ValueError(
                    "The selected config does not declare a causal language-model architecture."
                )

        n_layers = _architecture_layer_count_from_config(config)
        if n_layers is None:
            raise ValueError(
                "config.json does not provide a positive transformer layer count "
                "(n_layer, num_hidden_layers, num_layers, or n_layers)."
            )

        source_name = _configured_model_name(config)
        profile = _matching_supported_profile(
            model_type=model_type,
            n_layers=n_layers,
            source_name=source_name,
        )
        base_model_id = profile["id"] if profile else _local_base_model_id(source_name, model_type)
        display_base_name = profile["name"] if profile else _local_display_name(source_name, model_type)
        easyedit = dict(profile.get("easyedit", {})) if profile else {}

        return {
            "base_model_id": base_model_id,
            "display_base_name": display_base_name,
            "architecture_model_type": raw_model_type,
            "model_type": model_type,
            "n_layers": n_layers,
            "supports_generation": True,
            # The existing logit-lens projection and causal-trace hooks are
            # architecture-specific, so don't advertise them merely because a
            # generic AutoModel can load the checkpoint.
            "supports_logit_lens": model_type == "gpt2",
            "supports_causal_trace": model_type == "gpt2",
            **get_editing_support(profile or {}),
            "easyedit": easyedit,
            "trust_remote_code": bool(config.get("auto_map")) and model_type in _REMOTE_CODE_MODEL_TYPES,
            "generation_use_cache": profile.get("generation_use_cache", True) if profile else True,
        }

    def allocate_imported_model_id(self) -> str:
        """Create an opaque, collision-free-enough key for a local registration.

        This is a registry/cache-directory key rather than a display name.  It
        deliberately has no model-family or upload-order information so users
        do not need to interpret it.
        """
        existing_ids = {str(model.get("id")) for model in self.list_models()}
        while True:
            model_id = f"local-{uuid.uuid4().hex[:12]}"
            if model_id not in existing_ids:
                return model_id

    def add_imported_local_model(
        self,
        *,
        model_id: str,
        local_path: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a model whose files were copied into the managed local cache."""
        resolved_path = local_path.resolve()
        if not (resolved_path / "config.json").is_file():
            raise ValueError("Imported model cache is missing config.json.")
        if any(item["id"] == model_id for item in self.list_models()):
            raise ValueError(f"A model named {model_id} is already registered.")

        stored = {
            "id": model_id,
            "name": metadata["display_base_name"],
            "hf_id": str(resolved_path),
            "local_path": str(resolved_path),
            "source": "local",
            "managed_local": True,
            "base_model_id": metadata["base_model_id"],
            "mom2_stats_source": "base" if self._has_supported_hub_base(metadata) else "model",
            "architecture_model_type": metadata["architecture_model_type"],
            "model_type": metadata["model_type"],
            "n_layers": metadata["n_layers"],
            "supports_generation": metadata["supports_generation"],
            "supports_logit_lens": metadata["supports_logit_lens"],
            "supports_causal_trace": metadata["supports_causal_trace"],
            **get_editing_support(metadata),
            "trust_remote_code": metadata["trust_remote_code"],
            "generation_use_cache": metadata["generation_use_cache"],
            "easyedit": metadata["easyedit"],
        }
        models = self.list_local_models()
        models.append(stored)
        _write_local_models(models)
        try:
            from app.services.patch_service import patch_service

            lineage = patch_service.import_model_lineage(stored)
            stored["lineage_imported"] = lineage["imported"]
        except Exception:
            stored["lineage_imported"] = 0
        return stored

    def delete_local_model(self, model_id: str) -> dict[str, Any]:
        models = self.list_local_models()
        removed = next((item for item in models if item["id"] == model_id), None)
        remaining = [item for item in models if item["id"] != model_id]
        if len(remaining) == len(models):
            raise ValueError(f"Local model registration not found: {model_id}")
        _write_local_models(remaining)
        return removed or {}


def _managed_local_model_relative_path(model: dict[str, Any]) -> Path | None:
    """Recognize managed imports in portable and legacy absolute path formats."""
    if not model.get("managed_local"):
        return None
    model_id = model.get("id")
    local_path = model.get("local_path")
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", model_id):
        return None
    if not isinstance(local_path, str) or not local_path or "\0" in local_path:
        return None
    parts = PurePosixPath(local_path.replace("\\", "/")).parts
    relative_path = Path("models") / "imported_models" / model_id
    if ".." in parts or parts[-3:] != relative_path.parts:
        return None
    return relative_path


def _local_model_is_available(model: dict[str, Any]) -> bool:
    local_path = model.get("local_path")
    if not isinstance(local_path, str) or not local_path.strip():
        return False
    try:
        path = Path(local_path)
        return path.is_dir() and (path / "config.json").is_file()
    except (OSError, ValueError):
        return False


def _local_models_path() -> Path:
    path = settings.data_dir / "models" / "local_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_local_models(models: list[dict[str, Any]]) -> None:
    # Runtime consumers use absolute paths, but persisted managed imports must
    # survive moving DATA_DIR to a different server or container mount.
    stored_models = []
    for model in models:
        stored = dict(model)
        relative_path = _managed_local_model_relative_path(stored)
        if relative_path is not None:
            stored["local_path"] = relative_path.as_posix()
            stored["hf_id"] = relative_path.as_posix()
        stored_models.append(stored)
    path = _local_models_path()
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(stored_models, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def _read_local_models() -> list[dict[str, Any]]:
    path = _local_models_path()
    if not path.exists():
        return []
    try:
        raw = path.read_bytes()
        sanitized = raw.rstrip(b"\0")
        models = json.loads(sanitized.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(models, list):
        return []
    if sanitized != raw:
        # Recover registry files written with trailing NUL padding by older
        # direct-write paths before another request attempts to parse them.
        _write_local_models(models)
    return [model for model in models if isinstance(model, dict)]


def _migrate_managed_local_models(models: list[dict[str, Any]]) -> bool:
    """Backfill import details and replace the former generated display names."""
    import_jobs = _registered_local_import_jobs()
    changed = False
    for model in models:
        if not model.get("managed_local"):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str):
            continue
        relative_path = _managed_local_model_relative_path(model)
        if relative_path is not None:
            portable_path = relative_path.as_posix()
            if model.get("local_path") != portable_path or model.get("hf_id") != portable_path:
                model["local_path"] = portable_path
                model["hf_id"] = portable_path
                changed = True
        if not isinstance(model.get("import_metadata"), dict):
            job = import_jobs.get(model_id)
            if job:
                model["import_metadata"] = {
                    "import_id": job.get("job_id"),
                    "uploaded_at": job.get("created_at"),
                    "source_folder": job.get("source_name"),
                }
                changed = True

        base_model_id = model.get("base_model_id")
        local_index = _local_model_index(model_id, base_model_id) if isinstance(base_model_id, str) else None
        profile = next(
            (item for item in SUPPORTED_MODELS if item["id"] == base_model_id), None
        )
        if profile and local_index and model.get("name") == f"{profile['name']}_local{local_index}":
            model["name"] = profile["name"]
            changed = True
    return changed


def _registered_local_import_jobs() -> dict[str, dict[str, Any]]:
    jobs_dir = settings.data_dir / "model_cache_jobs"
    if not jobs_dir.exists():
        return {}
    jobs: dict[str, dict[str, Any]] = {}
    for path in jobs_dir.glob("model_import_*.json"):
        try:
            job = json.loads(path.read_bytes().rstrip(b"\0").decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        registered_model_id = job.get("registered_model_id") if isinstance(job, dict) else None
        if isinstance(registered_model_id, str):
            jobs[registered_model_id] = job
    return jobs


def _resolve_local_model_path(value: Any) -> Path:
    """Turn user-entered paths into paths the backend host can access.

    When the backend runs in WSL, Windows drive paths are available under
    ``/mnt/<drive>/...``. Persisting that host path also lets the runtime load
    the model without having to reinterpret the user's original input later.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Local model path is required.")
    raw_path = value.strip()
    if "\0" in raw_path:
        raise ValueError("Local model path is invalid.")

    windows_drive_path = re.fullmatch(r"([A-Za-z]):[\\/](.*)", raw_path)
    if windows_drive_path and os.name != "nt":
        drive, remainder = windows_drive_path.groups()
        parts = [part for part in re.split(r"[\\/]+", remainder) if part]
        candidate = Path("/mnt") / drive.lower() / Path(*parts)
    else:
        candidate = Path(raw_path).expanduser()

    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Local model path is invalid.") from exc


def _architecture_layer_count(model_path: Path) -> int | None:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return _architecture_layer_count_from_config(config)


def _architecture_layer_count_from_config(config: dict[str, Any]) -> int | None:
    for key in ("n_layer", "num_hidden_layers", "num_layers", "n_layers"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _is_causal_lm_architecture(name: str) -> bool:
    return (
        "ForCausalLM" in name
        or name.endswith("LMHeadModel")
        or name == "ChatGLMForConditionalGeneration"
    )


def _configured_model_name(config: dict[str, Any]) -> str | None:
    value = config.get("_name_or_path")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _matching_supported_profile(
    *, model_type: str, n_layers: int, source_name: str | None
) -> dict[str, Any] | None:
    candidates = [
        model
        for model in SUPPORTED_MODELS
        if model["model_type"] == model_type and model["n_layers"] == n_layers
    ]
    if source_name:
        normalized_source_name = _normalize_model_reference(source_name)
        exact = [
            model
            for model in candidates
            if normalized_source_name
            in {
                _normalize_model_reference(model["id"]),
                _normalize_model_reference(model["hf_id"]),
            }
        ]
        if len(exact) == 1:
            return exact[0]
    return candidates[0] if len(candidates) == 1 else None


def _normalize_model_reference(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _local_base_model_id(source_name: str | None, model_type: str) -> str:
    if source_name:
        basename = source_name.replace("\\", "/").rstrip("/").split("/")[-1]
        if basename:
            return _safe_local_base_id(basename)
    return model_type


def _local_display_name(source_name: str | None, model_type: str) -> str:
    if source_name:
        basename = source_name.replace("\\", "/").rstrip("/").split("/")[-1]
        if basename:
            return basename
    return model_type.upper() if model_type == "gpt2" else model_type.title()


def _safe_local_base_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip(".-_")
    return safe[:80] or "local-model"


def _local_model_index(model_id: Any, base_model_id: str) -> int | None:
    if not isinstance(model_id, str):
        return None
    match = re.fullmatch(rf"{re.escape(base_model_id)}_local([1-9][0-9]*)", model_id)
    return int(match.group(1)) if match else None


model_registry = ModelRegistry()
