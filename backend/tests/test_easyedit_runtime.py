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
