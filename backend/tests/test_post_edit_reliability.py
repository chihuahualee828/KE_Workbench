from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ml.editing import easyedit_runtime, kn_wrapper, memit_wrapper, rome_wrapper
import app.services.model_runtime_service as runtime_module


RELIABILITY_RESULT = {
    "reliability": {"mean_reliability": 1.0},
    "generations": [{"prompt": "Fact is", "text": "updated", "target_new": "updated"}],
}


def test_post_edit_evaluation_worker_receives_its_own_job_and_progress(monkeypatch) -> None:
    calls = []
    progress = []

    def execute(action, payload, **kwargs):
        calls.append((action, payload))
        kwargs["progress_callback"]({"current": 1, "total": 2})
        return RELIABILITY_RESULT

    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=True, execute=execute))
    edits = [{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated"}]
    result = easyedit_runtime.evaluate_post_edit_reliability(
        "demo", "rome", edits, edit_result={"metrics": []}, single_batch=True,
        cancel_job_id="job_evaluation", progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert result == RELIABILITY_RESULT
    assert calls == [("post_edit_evaluation", {
        "model_id": "demo", "method": "rome", "edits": edits,
        "edit_result": {"metrics": []}, "single_batch": True, "cancel_job_id": "job_evaluation",
    })]
    assert progress == [(1, 2)]


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_rank_one_edit_runs_post_edit_reliability(monkeypatch, method) -> None:
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    monkeypatch.setattr(rome_wrapper.model_registry, "get", lambda _: {"n_layers": 24})
    monkeypatch.setattr(rome_wrapper, "run_easyedit", lambda **_: {"patch_id": "patch_demo"})
    calls: list[tuple[str, str, list[dict]]] = []
    monkeypatch.setattr(
        rome_wrapper,
        "evaluate_post_edit_reliability",
        lambda model_id, method, edits, **kwargs: calls.append((model_id, method, edits)) or RELIABILITY_RESULT,
    )

    result = rome_wrapper.run_rome_edit(
        method=method,
        model_id="gpt2-xl",
        subject="Fact",
        relation_prompt="{} is",
        new_object="updated",
        ground_truth="original",
        target_layer=17,
        evaluation_prompts=[],
        rephrase_prompts=[],
        locality_inputs=None,
        portability_inputs=None,
        sequential_edit=False,
        mom2_adjustment=False,
    )

    assert calls == [("gpt2-xl", method, [{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated", "ground_truth": "original"}])]
    assert result["reliability"] == {"mean_reliability": 1.0}


def test_memit_edit_runs_post_edit_reliability(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    monkeypatch.setattr(memit_wrapper, "run_easyedit", lambda **_: {"patch_id": "patch_demo"})
    calls: list[tuple[str, str, list[dict]]] = []
    monkeypatch.setattr(
        memit_wrapper,
        "evaluate_post_edit_reliability",
        lambda model_id, method, edits, **kwargs: calls.append((model_id, method, edits)) or RELIABILITY_RESULT,
    )

    result = memit_wrapper.run_memit_edit(
        method="memit",
        model_id="gpt2-xl",
        edits=[{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated"}],
        layers=[17],
        evaluation_prompts=[],
        locality_inputs=None,
        portability_inputs=None,
        sequential_edit=False,
        mom2_adjustment=True,
    )

    assert calls == [("gpt2-xl", "memit", [{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated"}])]
    assert result["reliability"] == {"mean_reliability": 1.0}


def test_kn_edit_runs_post_edit_reliability(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    monkeypatch.setattr(kn_wrapper, "run_easyedit_kn", lambda **_: {"patch_id": "patch_demo"})
    calls: list[tuple[str, str, list[dict]]] = []
    monkeypatch.setattr(
        kn_wrapper,
        "evaluate_post_edit_reliability",
        lambda model_id, method, edits, **kwargs: calls.append((model_id, method, edits)) or RELIABILITY_RESULT,
    )

    result = kn_wrapper.run_kn_edit(
        model_id="gpt2-xl",
        subject="Fact",
        relation_prompt="{} is",
        new_object="updated",
        ground_truth="original",
        evaluation_prompts=[],
        sequential_edit=False,
    )

    assert calls == [("gpt2-xl", "kn", [{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated", "ground_truth": "original"}])]
    assert result["reliability"] == {"mean_reliability": 1.0}
