from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.api import routes_editing
from app.db.schemas import Mom2StatisticsPrecisionRequest, Mom2StatisticsRequest
from app.ml.editing import easyedit_runtime as runtime
from app.storage.local_store import LocalStore


@pytest.fixture
def profile(monkeypatch, tmp_path):
    hparams = tmp_path / "hparams.yaml"
    hparams.write_text(
        "layers: [0, 1]\nrewrite_module_tmp: transformer.h.{}.mlp.c_proj\n"
        "mom2_dataset: wikipedia\nmom2_dtype: float32\nmom2_n_samples: 2\n"
    )
    model = {"id": "demo", "hf_id": "example/demo", "n_layers": 2, "source": "hub",
             "supports_rome": True, "supports_r_rome": True, "supports_memit": True}
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: model)
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: hparams)
    monkeypatch.setattr(runtime, "_mom2_stats_dir", lambda: tmp_path / "stats")
    monkeypatch.setattr(runtime, "local_store", LocalStore(tmp_path))
    monkeypatch.setattr(runtime.model_runtime_service, "runtime_status", lambda _: {"device": "cpu"})
    monkeypatch.setattr(runtime, "_mom2_covariance_profiles", {})
    return hparams


def archive(dtype: str, *, invalid: bool = False) -> BytesIO:
    stream = BytesIO()
    matrix = np.eye(2, dtype=dtype) * 2
    if invalid:
        matrix[0, 0] = np.inf
    np.savez(stream, sample_size=2, **{"mom2.count": 2, "mom2.mom2": matrix})
    stream.seek(0)
    return stream


def filename(layer: int, dtype: str) -> str:
    return f"transformer.h.{layer}.mlp.c_proj_{dtype}_mom2_2.npz"


@pytest.mark.parametrize("method", ["rome", "r_rome", "memit"])
def test_selection_is_persisted_and_applied_to_edit_hparams(profile, method):
    result = runtime.set_mom2_statistics_precision("demo", method, "float64")
    assert result["default_mom2_dtype"] == "float32"
    assert result["mom2_dtype"] == "float64"
    assert runtime.local_store.read_json("settings/mom2_precision.json")["demo"][method] == "float64"
    hparams = SimpleNamespace(mom2_dtype="float32")
    runtime._configure_mom2_precision(hparams, "demo", method, "cpu")
    assert hparams.mom2_dtype == "float64"
    reset = runtime.set_mom2_statistics_precision("demo", method, None)
    assert reset["mom2_dtype"] == reset["default_mom2_dtype"] == "float32"


def test_mps_reports_yaml_default_but_disallows_explicit_fp64(profile, monkeypatch):
    profile.write_text(profile.read_text().replace("float32", "float64"))
    monkeypatch.setattr(runtime.model_runtime_service, "runtime_status", lambda _: {
        "device": "not_loaded", "gpu": {"mps_available": True},
    })
    status = runtime.get_mom2_statistics_status("demo", "memit")
    assert status["default_mom2_dtype"] == "float64"
    assert status["mom2_dtype"] == "float32"
    assert status["available_mom2_dtypes"] == ["float16", "float32"]
    with pytest.raises(runtime.EasyEditRuntimeError, match="not supported on MPS"):
        runtime.set_mom2_statistics_precision("demo", "memit", "float64")
    with pytest.raises(runtime.EasyEditRuntimeError, match="not supported on MPS"):
        runtime._configure_mom2_precision(SimpleNamespace(mom2_dtype="float32"), "demo", "memit", "mps", "float64")


def test_uploads_and_preflight_do_not_mix_precisions(profile):
    runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0], [(filename(0, "float32"), archive("float32"))])
    runtime.import_mom2_statistics("demo", "memit", "wikipedia", [1], [(filename(1, "float64"), archive("float64"))], mom2_dtype="float64")
    assert runtime.get_mom2_statistics_status("demo", "memit")["missing_layers"] == [1]
    runtime.set_mom2_statistics_precision("demo", "memit", "float64")
    assert runtime.get_mom2_statistics_status("demo", "memit")["missing_layers"] == [0]
    with pytest.raises(runtime.EasyEditRuntimeError, match=r"layer\(s\) 0"):
        runtime.require_mom2_statistics("demo", "memit", [0, 1], True)


def test_preflight_checks_matrix_header_not_just_precision_in_filename(profile, monkeypatch):
    hparams, _ = runtime._load_mom2_cache_profile("demo", "memit", [0])
    path = runtime._mom2_statistics_path(hparams, None, 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(archive("float64").getvalue())
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Preflight must not load entire matrices"))
    assert runtime.get_mom2_statistics_status("demo", "memit", [0])["missing_layers"] == [0]
    path.write_bytes(archive("float32").getvalue())
    assert runtime.get_mom2_statistics_status("demo", "memit", [0])["ready"]


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_import_validates_dtype_and_delete_only_removes_that_precision(profile, dtype):
    for precision in runtime.MOM2_DTYPES:
        result = runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0],
            [(filename(0, precision), archive(precision))], mom2_dtype=precision)
        assert result["ready"]
        assert result["mom2_dtype"] == precision
    result = runtime.delete_mom2_statistics("demo", "memit", [0], mom2_dtype=dtype)
    assert result["statistics"] == []
    for other in set(runtime.MOM2_DTYPES) - {dtype}:
        cache = runtime.list_mom2_statistics("demo", "memit", mom2_dtype=other)
        assert cache["statistics"][0]["mom2_dtype"] == other


def test_renamed_archive_with_wrong_dtype_cannot_replace_existing_cache(profile):
    result = runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0], [(filename(0, "float32"), archive("float32"))])
    path = Path(result["statistics"][0]["path"])
    original = path.read_bytes()
    with pytest.raises(runtime.EasyEditRuntimeError, match="precision is float64, but float32"):
        runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0], [(filename(0, "float32"), archive("float64"))])
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_invalid_multi_file_upload_does_not_partially_install(profile):
    with pytest.raises(runtime.EasyEditRuntimeError, match="precision is float64"):
        runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0, 1], [
            (filename(0, "float32"), archive("float32")),
            (filename(1, "float32"), archive("float64")),
        ])
    assert runtime.list_mom2_statistics("demo", "memit")["statistics"] == []


def test_non_finite_half_precision_archive_is_rejected(profile):
    with pytest.raises(runtime.EasyEditRuntimeError, match="non-finite"):
        runtime.import_mom2_statistics("demo", "memit", "wikipedia", [0],
            [(filename(0, "float16"), archive("float16", invalid=True))], mom2_dtype="float16")


@pytest.mark.parametrize("method", ["rome", "r_rome", "memit"])
def test_in_memory_covariance_is_reused_only_for_the_same_precision_and_file(profile, monkeypatch, method):
    module_name = "easyeditor.models.memit.memit_main" if method == "memit" else f"easyeditor.models.{method}.compute_u"
    cache = {}
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{"COV_CACHE" if method == "memit" else "inv_mom2_cache": cache}))
    loaded_model = SimpleNamespace(config=SimpleNamespace(_name_or_path="example/demo"))
    hparams, _ = runtime._load_mom2_cache_profile("demo", method, [0])
    key = ("example_demo", "transformer.h.0.mlp.c_proj")
    cache[key] = "old cache"
    runtime._refresh_mom2_covariance_cache(method, hparams, loaded_model, "cpu")
    assert key not in cache
    cache[key] = "current cache"
    runtime._refresh_mom2_covariance_cache(method, hparams, loaded_model, "cpu")
    assert cache[key] == "current cache"
    hparams.mom2_dtype = "float64"
    runtime._refresh_mom2_covariance_cache(method, hparams, loaded_model, "cpu")
    assert key not in cache
    cache[key] = "pre-import cache"
    path = runtime._mom2_statistics_path(hparams, loaded_model, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive("float64").getvalue())
    runtime._refresh_mom2_covariance_cache(method, hparams, loaded_model, "cpu")
    assert key not in cache


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_half_precision_rome_inverse_uses_fp32_without_changing_stored_stats(profile, monkeypatch, method):
    import torch

    runtime.import_mom2_statistics("demo", method, "wikipedia", [0],
        [(filename(0, "float16"), archive("float16"))], mom2_dtype="float16")
    hparams, _ = runtime._load_mom2_cache_profile("demo", method, [0], mom2_dtype="float16")
    loaded_model = SimpleNamespace(config=SimpleNamespace(_name_or_path="example/demo"))
    cache = {}
    monkeypatch.setitem(sys.modules, f"easyeditor.models.{method}.compute_u", SimpleNamespace(inv_mom2_cache=cache))
    runtime._refresh_mom2_covariance_cache(method, hparams, loaded_model, "cpu")
    inverse = cache[("example_demo", "transformer.h.0.mlp.c_proj")]
    assert inverse.dtype == torch.float32
    torch.testing.assert_close(inverse, torch.eye(2))
    with np.load(runtime._mom2_statistics_path(hparams, loaded_model, 0)) as stored:
        assert stored["mom2.mom2"].dtype == np.float16


def test_prepare_passes_selected_precision_to_easyedit_statistics(profile, monkeypatch):
    runtime.set_mom2_statistics_precision("demo", "memit", "float64")
    hparams, _ = runtime._load_mom2_cache_profile("demo", "memit", [0])
    hparams.mom2_dtype = "float32"
    loaded_model = SimpleNamespace(config=SimpleNamespace(_name_or_path="example/demo"))
    monkeypatch.setattr(runtime, "model_runtime_service", SimpleNamespace(uses_hard_worker=False, runtime_status=lambda _: {"device": "cpu"}))
    monkeypatch.setattr(runtime, "_load_edit_hparams", lambda *_: (hparams, [0]))
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", lambda _: nullcontext({"model": loaded_model, "tokenizer": SimpleNamespace(), "device": "cpu"}))
    monkeypatch.setattr(runtime, "_override_easyedit_device", lambda *_: None)
    calls = []

    def layer_stats(*args, **kwargs):
        calls.append(kwargs["precision"])
        path = runtime._mom2_statistics_path(kwargs["hparams"], loaded_model, 0)
        path.parent.mkdir(parents=True)
        path.write_bytes(archive(kwargs["precision"]).getvalue())

    monkeypatch.setitem(sys.modules, "easyeditor.models.rome.layer_stats", SimpleNamespace(layer_stats=layer_stats))
    result = runtime.prepare_mom2_statistics("demo", "memit", [0])
    assert calls == ["float64"]
    assert result["mom2_dtype"] == "float64"
    assert result["ready"]


def test_prepare_api_rejects_fp64_mps_before_creating_job(profile, monkeypatch):
    monkeypatch.setattr(runtime.model_runtime_service, "runtime_status", lambda _: {"device": "mps"})
    monkeypatch.setattr(routes_editing, "_require_loaded_model", lambda _: None)
    monkeypatch.setattr(routes_editing.experiment_service, "create", lambda *_: pytest.fail("An unsupported precision must not start a job"))
    with pytest.raises(HTTPException) as error:
        routes_editing.prepare_mom2(Mom2StatisticsRequest(model_id="demo", method="memit", mom2_dtype="float64"))
    assert error.value.status_code == 400


def test_precision_cannot_change_while_an_edit_job_is_reserved(profile):
    with routes_editing.operation_guard.reserve():
        with pytest.raises(HTTPException) as error:
            routes_editing.select_mom2_statistics_precision(Mom2StatisticsPrecisionRequest(model_id="demo", method="memit", mom2_dtype="float64"))
    assert error.value.status_code == 409
