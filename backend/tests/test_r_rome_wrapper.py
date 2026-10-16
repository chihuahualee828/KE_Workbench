from __future__ import annotations

from types import SimpleNamespace

from app.ml.editing import r_rome_wrapper
import app.services.model_runtime_service as runtime_module


def test_r_rome_edit_dispatches_to_the_loaded_model_worker(monkeypatch) -> None:
    calls: list[tuple[str, dict, float]] = []
    monkeypatch.setattr(
        runtime_module,
        "model_runtime_service",
        SimpleNamespace(
            uses_hard_worker=True,
            execute=lambda action, payload, timeout: calls.append((action, payload, timeout)) or {"ok": True},
        ),
    )

    result = r_rome_wrapper.run_r_rome_edit(
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
            "r_rome_edit",
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


def test_r_rome_sequential_series_updates_one_cumulative_patch(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    evaluated: list[tuple[str, str, list[dict]]] = []

    monkeypatch.setattr(
        runtime_module,
        "model_runtime_service",
        SimpleNamespace(uses_hard_worker=False),
    )

    def fake_r_rome_edit(**kwargs):
        calls.append(kwargs)
        context = kwargs["series_context"]
        context.patch_id = context.patch_id or "patch_series"
        return {"patch_id": context.patch_id}

    monkeypatch.setattr(r_rome_wrapper, "run_r_rome_edit", fake_r_rome_edit)
    monkeypatch.setattr(
        r_rome_wrapper,
        "evaluate_post_edit_reliability",
        lambda model_id, method, edits, **kwargs: evaluated.append((model_id, method, edits)) or {"reliability": {"mean_reliability": 1.0}, "generations": []},
    )

    result = r_rome_wrapper.run_r_rome_sequential_edits(
        model_id="gpt2-xl",
        edits=[
            {
                "subject": "Fact one",
                "relation_prompt": "{} is",
                "new_object": "updated",
                "source_metadata": {"case_id": 0, "rephrase_prompt": "Restate fact one"},
            },
            {"subject": "Fact two", "relation_prompt": "{} is", "new_object": "changed"},
        ],
        target_layer=17,
        evaluation_prompts=["Fact two is"],
        mom2_adjustment=False,
    )

    assert [call["subject"] for call in calls] == ["Fact one", "Fact two"]
    assert calls[0]["source_metadata"] == {"case_id": 0, "rephrase_prompt": "Restate fact one"}
    assert all(call["sequential_edit"] is True for call in calls)
    assert all(call["generate_evaluation"] is False for call in calls)
    assert all(call["patch_mode"] == "series" for call in calls)
    assert calls[0]["series_context"] is calls[1]["series_context"]
    assert calls[0]["series_context"].continue_from_active_patches is True
    assert result["patch_ids"] == ["patch_series"]
    assert result["patch_count"] == 1
    assert result["completed_fact_count"] == 2
    assert result["reliability"] == {"mean_reliability": 1.0}
    assert evaluated[0][0:2] == ("gpt2-xl", "r_rome")
    assert evaluated[0][2][0]["source_metadata"] == {"case_id": 0, "rephrase_prompt": "Restate fact one"}
