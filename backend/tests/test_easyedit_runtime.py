from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml.editing import easyedit_runtime


def test_mom2_preflight_checks_files_without_dispatching_to_model_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "model_id": "gpt2-xl",
        "method": "memit",
        "mom2_adjustment": True,
        "layers": [13],
        "ready": True,
        "missing_layers": [],
    }
    monkeypatch.setattr(easyedit_runtime, "get_mom2_statistics_status", lambda *_: expected)

    result = easyedit_runtime.require_mom2_statistics("gpt2-xl", "memit", [13], True)

    assert result is expected


def test_mom2_preflight_rejects_missing_layers_before_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        easyedit_runtime,
        "get_mom2_statistics_status",
        lambda *_: {"ready": False, "missing_layers": [13, 14]},
    )

    with pytest.raises(easyedit_runtime.EasyEditRuntimeError, match=r"layer\(s\) 13, 14"):
        easyedit_runtime.require_mom2_statistics("gpt2-xl", "memit", [13, 14], True)


def test_mom2_cache_listing_reads_hparams_yaml_without_importing_easyedit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hparams_path = tmp_path / "memit.yaml"
    hparams_path.write_text(
        """
layers: [1]
rewrite_module_tmp: transformer.h.{}.mlp.c_proj
mom2_dataset: wikipedia
mom2_n_samples: 100
mom2_dtype: float32
""".strip(),
        encoding="utf-8",
    )
    stats_dir = tmp_path / "stats"
    cache_path = stats_dir / "demo" / "wikipedia_stats" / "transformer.h.1.mlp.c_proj_float32_mom2_100.npz"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"statistics")
    model = {"id": "demo", "hf_id": "example/demo", "n_layers": 3, "supports_memit": True}

    monkeypatch.setattr(easyedit_runtime.model_registry, "get", lambda _: model)
    monkeypatch.setattr(easyedit_runtime, "_resolve_hparams_path", lambda *_: hparams_path)
    monkeypatch.setattr(easyedit_runtime, "_mom2_stats_dir", lambda: stats_dir)
    monkeypatch.setattr(
        easyedit_runtime,
        "_load_easyedit",
        lambda *_: pytest.fail("Cache listing must not import EasyEdit."),
    )

    result = easyedit_runtime.list_mom2_statistics("demo", "memit")

    assert result["dataset"] == "wikipedia"
    assert [entry["layer"] for entry in result["statistics"]] == [1]
    assert result["statistics"][0]["filename"] == cache_path.name


def test_imported_model_reuses_base_model_mom2_cache_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    imported_model = {
        "id": "gpt2-xl_local6",
        "hf_id": "/models/gpt2-xl_local6",
        "local_path": "/models/gpt2-xl_local6",
        "source": "local",
        "base_model_id": "gpt2-xl",
    }
    base_model = {"id": "gpt2-xl", "hf_id": "gpt2-xl", "source": "hub"}
    monkeypatch.setattr(
        easyedit_runtime.model_registry,
        "get",
        lambda model_id: base_model if model_id == "gpt2-xl" else imported_model,
    )

    assert easyedit_runtime._mom2_cache_model_name(imported_model) == "gpt2-xl"

    runtime_model = SimpleNamespace(config=SimpleNamespace(_name_or_path="/models/gpt2-xl_local6"))
    identity = easyedit_runtime._set_loaded_model_mom2_cache_identity(runtime_model, imported_model)

    assert runtime_model.config._name_or_path == "gpt2-xl"
    easyedit_runtime._restore_loaded_model_mom2_cache_identity(identity)
    assert runtime_model.config._name_or_path == "/models/gpt2-xl_local6"


def test_r_rome_is_registered_as_an_easyedit_method() -> None:
    assert easyedit_runtime.METHODS["r_rome"] == {
        "hparams_class": "R_ROMEHyperParams",
        "hparams_key": "r_rome_hparams",
        "support_key": "supports_r_rome",
        "requires_subject": True,
    }


def test_reliability_evaluation_uses_active_runtime_without_editing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = {
        "id": "demo",
        "hf_id": "example/demo",
        "n_layers": 3,
        "supports_rome": True,
    }
    hparams = SimpleNamespace(model_name="example/demo", device=0, alg_name="ROME")
    runtime_model = object()
    runtime_tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    calls: list[dict[str, str]] = []

    def compute_edit_quality(_model, _name, _hparams, _tokenizer, record, _device):
        calls.append(record)
        return {
            "rewrite_acc": 1.0 if record["target_new"] == "Paris" else 0.0,
            "rephrase_acc": 0.75 if record.get("rephrase_prompt") else None,
        }

    monkeypatch.setattr(easyedit_runtime.model_registry, "get", lambda _: model)
    monkeypatch.setattr(easyedit_runtime, "_load_easyedit", lambda _: (None, SimpleNamespace(from_hparams=lambda _: hparams)))
    monkeypatch.setattr(easyedit_runtime, "_resolve_hparams_path", lambda *_: Path("/tmp/rome.yaml"))
    monkeypatch.setattr(
        easyedit_runtime,
        "model_runtime_service",
        SimpleNamespace(
            exclusive_runtime_access=contextlib.nullcontext,
            require_loaded_runtime_bundle=lambda _: {
                "model": runtime_model,
                "tokenizer": runtime_tokenizer,
                "device": "cpu",
            },
        ),
    )
    monkeypatch.setattr(easyedit_runtime.patch_service, "active_patches_for_model", lambda _: ["patch_demo"])
    monkeypatch.setattr(easyedit_runtime.patch_service, "sync_active_patches_to_model", lambda *_: None)
    monkeypatch.setattr(
        easyedit_runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            compute_edit_quality=compute_edit_quality
        ) if name == "easyeditor.evaluate" else None,
    )

    result = easyedit_runtime.evaluate_easyedit_reliability(
        model_id="demo",
        method="rome",
        edits=[
            {
                "subject": "The Space Needle",
                "relation_prompt": "{} is in",
                "new_object": "Paris",
                "rephrase_prompts": ["The Space Needle can be found in"],
            },
            {"subject": "The Eiffel Tower", "relation_prompt": "{} is in", "new_object": "Rome"},
        ],
    )

    assert calls == [
        {
            "prompt": "The Space Needle is in",
            "target_new": "Paris",
            "ground_truth": "",
            "rephrase_prompt": "The Space Needle can be found in",
        },
        {"prompt": "The Eiffel Tower is in", "target_new": "Rome", "ground_truth": ""},
    ]
    assert result["active_patch_ids"] == ["patch_demo"]
    assert result["mean_reliability"] == 0.5
    assert result["mean_generalization"] == 0.75
    assert result["cases"][0]["reliability_score"] == 1.0
    assert result["cases"][1]["reliability_score"] == 0.0
    assert result["cases"][0]["generalization_score"] == 0.75
    assert result["cases"][1]["generalization_score"] is None
