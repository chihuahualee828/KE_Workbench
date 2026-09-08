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
            },
            7200.0,
        )
    ]
