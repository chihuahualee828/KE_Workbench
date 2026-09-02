from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


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
    model_type: str = "auto",
    supports_logit_lens: bool = False,
    supports_causal_trace: bool = False,
    trust_remote_code: bool = False,
    generation_use_cache: bool = True,
    rome_hparams: str | None = None,
    memit_hparams: str | None = None,
    kn_hparams: str | None = None,
    estimated_vram_gb: dict[str, int] | None = None,
) -> dict[str, Any]:
    easyedit = {
        key: value
        for key, value in {
            "rome_hparams": rome_hparams,
            "memit_hparams": memit_hparams,
            "kn_hparams": kn_hparams,
            "estimated_vram_gb": estimated_vram_gb,
        }.items()
        if value
    }
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
        "supports_rome": bool(rome_hparams),
        "supports_memit": bool(memit_hparams),
        "supports_kn": bool(kn_hparams),
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
        rome_hparams="hparams/ROME/gpt2-xl.yaml",
        memit_hparams="hparams/MEMIT/gpt2-xl.yaml",
        kn_hparams="hparams/KN/gpt2-xl.yaml",
        estimated_vram_gb={"rome": 10, "memit": 11},
    ),
    _hub_model(
        model_id="gpt-j-6b",
        name="GPT-J 6B",
        hf_id="EleutherAI/gpt-j-6B",
        model_type="gptj",
        n_layers=28,
        rome_hparams="hparams/ROME/gpt-j-6B.yaml",
        memit_hparams="hparams/MEMIT/gpt-j-6B.yaml",
        kn_hparams="hparams/KN/gpt-j-6B.yaml",
        estimated_vram_gb={"rome": 16, "memit": 18},
    ),
    _hub_model(
        model_id="baichuan-7b",
        name="Baichuan 2 7B Chat",
        hf_id="baichuan-inc/Baichuan2-7B-Chat",
        model_type="baichuan",
        n_layers=32,
        trust_remote_code=True,
        rome_hparams="hparams/ROME/baichuan-7b.yaml",
        memit_hparams="hparams/MEMIT/baichuan-7b.yaml",
        kn_hparams="hparams/KN/baichuan-7b.yaml",
    ),
    _hub_model(
        model_id="chatglm2-6b",
        name="ChatGLM2 6B",
        hf_id="THUDM/chatglm2-6b",
        model_type="chatglm",
        n_layers=28,
        trust_remote_code=True,
        rome_hparams="hparams/ROME/chatglm2-6b.yaml",
        memit_hparams="hparams/MEMIT/chatglm2-6b.yaml",
        kn_hparams="hparams/KN/chatglm2-6b.yaml",
    ),
    _hub_model(
        model_id="chatglm4-9b",
        name="GLM-4 9B Chat",
        hf_id="THUDM/glm-4-9b-chat",
        model_type="chatglm",
        n_layers=40,
        trust_remote_code=True,
        rome_hparams="hparams/ROME/chatglm4-9b.yaml",
    ),
    _hub_model(
        model_id="internlm-7b",
        name="InternLM 7B",
        hf_id="internlm/internlm-7b",
        model_type="internlm",
        n_layers=32,
        trust_remote_code=True,
        generation_use_cache=False,
        rome_hparams="hparams/ROME/internlm-7b.yaml",
        memit_hparams="hparams/MEMIT/internlm-7b.yaml",
        kn_hparams="hparams/KN/internlm-7b.yaml",
    ),
    _hub_model(
        model_id="llama-7b",
        name="Llama 2 7B",
        hf_id="meta-llama/Llama-2-7b-hf",
        model_type="llama",
        n_layers=32,
        rome_hparams="hparams/ROME/llama-7b.yaml",
        memit_hparams="hparams/MEMIT/llama-7b.yaml",
        kn_hparams="hparams/KN/llama-7b.yaml",
    ),
    _hub_model(
        model_id="llama3-8b",
        name="Llama 3 8B",
        hf_id="meta-llama/Meta-Llama-3-8B",
        model_type="llama",
        n_layers=32,
        rome_hparams="hparams/ROME/llama3-8b.yaml",
        kn_hparams="hparams/KN/llama3-8b.yaml",
        estimated_vram_gb={"rome": 18},
    ),
    _hub_model(
        model_id="llama3.2-3b",
        name="Llama 3.2 3B",
        hf_id="meta-llama/Llama-3.2-3B",
        model_type="llama",
        n_layers=28,
        rome_hparams="hparams/ROME/llama3.2-3b.yaml",
        memit_hparams="hparams/MEMIT/llama3.2-3b.yaml",
        kn_hparams="hparams/KN/llama3.2-3b.yaml",
        estimated_vram_gb={"rome": 8, "memit": 9},
    ),
    _hub_model(
        model_id="mistral-7b",
        name="Mistral 7B",
        hf_id="mistralai/Mistral-7B-v0.1",
        model_type="mistral",
        n_layers=32,
        rome_hparams="hparams/ROME/mistral-7b.yaml",
        memit_hparams="hparams/MEMIT/mistral-7b.yaml",
        kn_hparams="hparams/KN/mistral-7b.yaml",
        estimated_vram_gb={"rome": 16, "memit": 18},
    ),
    _hub_model(
        model_id="qwen2.5-7b",
        name="Qwen2.5 7B",
        hf_id="Qwen/Qwen2.5-7B",
        model_type="qwen",
        n_layers=28,
        rome_hparams="hparams/ROME/qwen2.5-7b.yaml",
        memit_hparams="hparams/MEMIT/qwen2.5-7b.yaml",
        kn_hparams="hparams/KN/qwen2.5-7b.yaml",
        estimated_vram_gb={"rome": 16, "memit": 18},
    ),
    _hub_model(
        model_id="qwen2-7b",
        name="Qwen2 7B",
        hf_id="Qwen/Qwen2-7B",
        model_type="qwen",
        n_layers=28,
        rome_hparams="hparams/ROME/qwen2-7b.yaml",
        memit_hparams="hparams/MEMIT/qwen2-7b.yaml",
        kn_hparams="hparams/KN/qwen2-7b.yaml",
        estimated_vram_gb={"rome": 16, "memit": 18},
    ),
    _hub_model(
        model_id="qwen-7b",
        name="Qwen 7B",
        hf_id="Qwen/Qwen-7B",
        model_type="qwen",
        n_layers=32,
        trust_remote_code=True,
        rome_hparams="hparams/ROME/qwen-7b.yaml",
        memit_hparams="hparams/MEMIT/qwen-7b.yaml",
        kn_hparams="hparams/KN/qwen-7b.yaml",
        estimated_vram_gb={"rome": 16, "memit": 18},
    ),
    _hub_model(
        model_id="qwen3.5-9b",
        name="Qwen3.5 9B",
        hf_id="Qwen/Qwen3.5-9B",
        model_type="qwen",
        n_layers=32,
        rome_hparams="hparams/ROME/qwen3.5-9b.yaml",
    ),
]


class ModelRegistry:
    def list_models(self) -> list[dict]:
        return [*SUPPORTED_MODELS, *self.list_local_models()]

    def get(self, model_id: str) -> dict:
        for model in self.list_models():
            if model["id"] == model_id:
                return model
        raise ValueError(f"Unsupported model: {model_id}")

    def list_local_models(self) -> list[dict]:
        path = _local_models_path()
        if not path.exists():
            return []
        models = json.loads(path.read_text(encoding="utf-8"))
        for model in models:
            local_path = model.get("local_path")
            if not local_path:
                continue
            configured_layers = _architecture_layer_count(Path(local_path))
            if configured_layers is not None:
                model["n_layers"] = configured_layers
        return models

    def add_local_model(self, model: dict[str, Any], *, managed_local: bool = False) -> dict[str, Any]:
        model_id = model["id"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", model_id):
            raise ValueError("Model ID may contain only letters, numbers, dot, underscore, colon, and hyphen.")
        if any(item["id"] == model_id for item in SUPPORTED_MODELS):
            raise ValueError(f"{model_id} is a built-in model ID.")
        local_path = Path(model["local_path"]).expanduser().resolve()
        if not local_path.exists():
            raise ValueError(f"Local model path does not exist: {local_path}")
        if not (local_path / "config.json").exists():
            raise ValueError(f"Local model path must contain config.json: {local_path}")
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
        if model.get("supports_rome") and not model.get("rome_hparams"):
            raise ValueError("ROME local models need a ROME hparams path.")
        if model.get("supports_memit") and not model.get("memit_hparams"):
            raise ValueError("MEMIT local models need a MEMIT hparams path.")
        if model.get("supports_kn") and not model.get("kn_hparams"):
            raise ValueError("KN local models need a KN hparams path.")
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
            "architecture_model_type": model.get("architecture_model_type"),
            "model_type": model_type,
            "n_layers": configured_layers,
            "supports_generation": bool(model.get("supports_generation", True)),
            "supports_logit_lens": bool(model.get("supports_logit_lens", False)),
            "supports_causal_trace": bool(model.get("supports_causal_trace", False)),
            "supports_rome": bool(model.get("supports_rome", False)),
            "supports_memit": bool(model.get("supports_memit", False)),
            "supports_kn": bool(model.get("supports_kn", False)),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "generation_use_cache": bool(model.get("generation_use_cache", True)),
            "easyedit": {
                key: value
                for key, value in {
                    "rome_hparams": model.get("rome_hparams") or None,
                    "memit_hparams": model.get("memit_hparams") or None,
                    "kn_hparams": model.get("kn_hparams") or None,
                }.items()
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
            "supports_rome": bool(profile and profile.get("supports_rome")),
            "supports_memit": bool(profile and profile.get("supports_memit")),
            "supports_kn": bool(profile and profile.get("supports_kn")),
            "easyedit": easyedit,
            "trust_remote_code": bool(config.get("auto_map")) and model_type in _REMOTE_CODE_MODEL_TYPES,
            "generation_use_cache": profile.get("generation_use_cache", True) if profile else True,
        }

    def allocate_imported_model_id(self, base_model_id: str) -> tuple[str, int]:
        """Return the next never-reused local index for this model family."""
        safe_base_id = _safe_local_base_id(base_model_id)
        counters = _read_local_model_upload_counters()
        existing_indices = [
            _local_model_index(item.get("id"), safe_base_id)
            for item in self.list_local_models()
        ]
        used_indices = [
            int(counters.get(safe_base_id, 0)),
            *(index or 0 for index in existing_indices),
        ]
        current_index = max(used_indices)
        next_index = current_index + 1
        counters[safe_base_id] = next_index
        _write_local_model_upload_counters(counters)
        return f"{safe_base_id}_local{next_index}", next_index

    def add_imported_local_model(
        self,
        *,
        model_id: str,
        local_path: Path,
        metadata: dict[str, Any],
        local_index: int,
    ) -> dict[str, Any]:
        """Persist a model whose files were copied into the managed local cache."""
        resolved_path = local_path.resolve()
        if not (resolved_path / "config.json").is_file():
            raise ValueError("Imported model cache is missing config.json.")
        if any(item["id"] == model_id for item in self.list_models()):
            raise ValueError(f"A model named {model_id} is already registered.")

        stored = {
            "id": model_id,
            "name": f"{metadata['display_base_name']}_local{local_index}",
            "hf_id": str(resolved_path),
            "local_path": str(resolved_path),
            "source": "local",
            "managed_local": True,
            "base_model_id": metadata["base_model_id"],
            "architecture_model_type": metadata["architecture_model_type"],
            "model_type": metadata["model_type"],
            "n_layers": metadata["n_layers"],
            "supports_generation": metadata["supports_generation"],
            "supports_logit_lens": metadata["supports_logit_lens"],
            "supports_causal_trace": metadata["supports_causal_trace"],
            "supports_rome": metadata["supports_rome"],
            "supports_memit": metadata["supports_memit"],
            "supports_kn": metadata["supports_kn"],
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


def _local_models_path() -> Path:
    path = settings.data_dir / "models" / "local_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_local_models(models: list[dict[str, Any]]) -> None:
    _local_models_path().write_text(json.dumps(models, indent=2), encoding="utf-8")


def _local_model_upload_counters_path() -> Path:
    return _local_models_path().with_name("local_model_upload_counters.json")


def _read_local_model_upload_counters() -> dict[str, int]:
    path = _local_model_upload_counters_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, int) and value >= 0
    }


def _write_local_model_upload_counters(counters: dict[str, int]) -> None:
    _local_model_upload_counters_path().write_text(
        json.dumps(counters, indent=2), encoding="utf-8"
    )


def _architecture_layer_count(model_path: Path) -> int | None:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
