from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ml.editing import rome_wrapper
import app.services.model_runtime_service as runtime_module


@pytest.mark.parametrize("method", ["rome", "r_rome"])
@pytest.mark.parametrize("series", [False, True])
def test_worker_selects_rank_one_method_and_operation(monkeypatch, method, series):
    from app.services.model_worker import _dispatch

    runner_name = "run_rome_sequential_edits" if series else "run_rome_edit"
    monkeypatch.setattr(rome_wrapper, runner_name, lambda **payload: payload)
    action = f"{method}_sequential_edits" if series else f"{method}_edit"
    result = _dispatch(SimpleNamespace(), action, {"model_id": "demo", "sequential_edit": False})
    assert result == {"method": method, "model_id": "demo", "sequential_edit": False}


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_rank_one_series_updates_one_cumulative_patch(
    monkeypatch, method,
) -> None:
    calls: list[dict] = []
    evaluated: list[tuple[str, str, list[dict]]] = []

    monkeypatch.setattr(
        runtime_module,
        "model_runtime_service",
        SimpleNamespace(uses_hard_worker=False),
    )

    def fake_rank_one_edit(**kwargs):
        calls.append(kwargs)
        context = kwargs["series_context"]
        context.patch_id = context.patch_id or "patch_series"
        return {"patch_id": context.patch_id}

    monkeypatch.setattr(rome_wrapper, "run_rome_edit", fake_rank_one_edit)
    monkeypatch.setattr(
        rome_wrapper,
        "evaluate_post_edit_reliability",
        lambda model_id, method, edits, **kwargs: evaluated.append((model_id, method, edits)) or {"reliability": {"mean_reliability": 1.0}, "generations": []},
    )

    result = rome_wrapper.run_rome_sequential_edits(
        method=method,
        model_id="gpt2-xl",
        edits=[
            {
                "subject": "Fact one",
                "relation_prompt": "{} is",
                "new_object": "updated",
                "source_metadata": {"case_id": 0, "locality_prompt": "Elsewhere is"},
            },
            {"subject": "Fact two", "relation_prompt": "{} is", "new_object": "changed"},
        ],
        target_layer=17,
        evaluation_prompts=["Fact two is"],
        mom2_adjustment=False,
    )

    assert all(call["method"] == method for call in calls)
    assert [call["subject"] for call in calls] == ["Fact one", "Fact two"]
    assert calls[0]["source_metadata"] == {"case_id": 0, "locality_prompt": "Elsewhere is"}
    assert all(call["sequential_edit"] is True for call in calls)
    assert all(call["generate_evaluation"] is False for call in calls)
    assert all(call["patch_mode"] == "series" for call in calls)
    assert calls[0]["series_context"] is calls[1]["series_context"]
    assert calls[0]["series_context"].continue_from_active_patches is True
    assert result["patch_ids"] == ["patch_series"]
    assert result["patch_count"] == 1
    assert result["completed_fact_count"] == 2
    assert result["reliability"] == {"mean_reliability": 1.0}
    assert evaluated[0][0:2] == ("gpt2-xl", method)
    assert evaluated[0][2][0]["source_metadata"] == {"case_id": 0, "locality_prompt": "Elsewhere is"}


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_rank_one_series_can_start_from_a_clean_patch_stack(monkeypatch, method) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=False))
    monkeypatch.setattr(
        rome_wrapper,
        "run_rome_edit",
        lambda **kwargs: calls.append(kwargs) or {"patch_id": "patch_series"},
    )
    monkeypatch.setattr(rome_wrapper, "evaluate_post_edit_reliability", lambda *_, **kwargs: {"reliability": {}, "generations": []})

    rome_wrapper.run_rome_sequential_edits(
        method=method,
        model_id="gpt2-xl",
        edits=[{"subject": "Fact", "relation_prompt": "{} is", "new_object": "updated"}],
        target_layer=17,
        evaluation_prompts=[],
        mom2_adjustment=False,
        sequential_edit=False,
    )

    assert calls[0]["sequential_edit"] is True
    assert calls[0]["series_context"].continue_from_active_patches is False


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_rank_one_edit_dispatches_to_the_loaded_model_worker(monkeypatch, method) -> None:
    calls: list[tuple[str, dict, float]] = []
    monkeypatch.setattr(
        runtime_module,
        "model_runtime_service",
        SimpleNamespace(
            uses_hard_worker=True,
            execute=lambda action, payload, timeout: calls.append((action, payload, timeout)) or {"ok": True},
        ),
    )

    result = rome_wrapper.run_rome_edit(
        method=method,
        model_id="gpt2-xl",
        subject="The Space Needle",
        relation_prompt="{} is located in the city of",
        new_object="Paris",
        ground_truth="Seattle",
        target_layer=17,
        evaluation_prompts=["The Space Needle is located in the city of"],
        rephrase_prompts=[],
        locality_inputs=None,
        portability_inputs=None,
        sequential_edit=False,
        mom2_adjustment=False,
    )

    assert result == {"ok": True}
    assert calls == [
        (
            f"{method}_edit",
            {
                "model_id": "gpt2-xl",
                "subject": "The Space Needle",
                "relation_prompt": "{} is located in the city of",
                "new_object": "Paris",
                "ground_truth": "Seattle",
                "target_layer": 17,
                "evaluation_prompts": ["The Space Needle is located in the city of"],
                "rephrase_prompts": [],
                "locality_inputs": None,
                "portability_inputs": None,
                "sequential_edit": False,
                "mom2_adjustment": False,
                "source_metadata": None,
                "generate_evaluation": True,
            },
            7200.0,
        )
    ]
