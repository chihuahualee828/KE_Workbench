from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


def _hub_model(
    *,
    model_id: str,
    name: str,
    hf_id: str,
    n_layers: int,
    model_type: str = "auto",
    supports_logit_lens: bool = False,
    supports_causal_trace: bool = False,
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
        rome_hparams="hparams/ROME/chatglm4-9b.yaml",
    ),
    _hub_model(
        model_id="internlm-7b",
        name="InternLM 7B",
        hf_id="internlm/internlm-7b",
        model_type="internlm",
        n_layers=32,
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

    def add_local_model(self, model: dict[str, Any]) -> dict[str, Any]:
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
            "model_type": model_type,
            "n_layers": configured_layers,
            "supports_generation": bool(model.get("supports_generation", True)),
            "supports_logit_lens": bool(model.get("supports_logit_lens", False)),
            "supports_causal_trace": bool(model.get("supports_causal_trace", False)),
            "supports_rome": bool(model.get("supports_rome", False)),
            "supports_memit": bool(model.get("supports_memit", False)),
            "supports_kn": bool(model.get("supports_kn", False)),
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

    def delete_local_model(self, model_id: str) -> None:
        models = self.list_local_models()
        remaining = [item for item in models if item["id"] != model_id]
        if len(remaining) == len(models):
            raise ValueError(f"Local model registration not found: {model_id}")
        _write_local_models(remaining)


def _local_models_path() -> Path:
    path = settings.data_dir / "models" / "local_models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_local_models(models: list[dict[str, Any]]) -> None:
    _local_models_path().write_text(json.dumps(models, indent=2), encoding="utf-8")


def _architecture_layer_count(model_path: Path) -> int | None:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("n_layer", "num_hidden_layers", "n_layers"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


model_registry = ModelRegistry()
