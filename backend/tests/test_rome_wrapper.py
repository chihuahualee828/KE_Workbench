from __future__ import annotations

from types import SimpleNamespace

from app.ml.editing import rome_wrapper
import app.services.model_runtime_service as runtime_module


def test_rome_sequential_series_creates_one_ordered_patch_per_fact(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    generated: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        runtime_module,
        "model_runtime_service",
        SimpleNamespace(uses_hard_worker=False),
    )

    def fake_rome_edit(**kwargs):
        calls.append(kwargs)
        return {"patch_id": f"patch_{len(calls)}"}

    monkeypatch.setattr(rome_wrapper, "run_rome_edit", fake_rome_edit)
    monkeypatch.setattr(
        rome_wrapper,
        "_generate_evaluation_outputs",
        lambda *, model_id, patch_id, prompts: generated.append((patch_id, prompts)) or [],
    )

    result = rome_wrapper.run_rome_sequential_edits(
        model_id="gpt2-xl",
        edits=[
            {"subject": "Fact one", "relation_prompt": "{} is", "new_object": "updated"},
            {"subject": "Fact two", "relation_prompt": "{} is", "new_object": "changed"},
        ],
        target_layer=17,
        evaluation_prompts=["Fact two is"],
        mom2_adjustment=False,
    )

    assert [call["subject"] for call in calls] == ["Fact one", "Fact two"]
    assert all(call["sequential_edit"] is True for call in calls)
    assert all(call["generate_evaluation"] is False for call in calls)
    assert result["patch_ids"] == ["patch_1", "patch_2"]
    assert result["patch_count"] == 2
    assert generated == [("patch_2", ["Fact two is"])]
