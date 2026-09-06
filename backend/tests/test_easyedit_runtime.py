from __future__ import annotations

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
