from __future__ import annotations

from app.db.schemas import RomeSequentialEditsRequest


def test_rome_sequential_request_accepts_more_than_one_hundred_facts() -> None:
    fact = {
        "subject": "Example",
        "relation_prompt": "{} is",
        "new_object": "updated",
    }

    request = RomeSequentialEditsRequest.model_validate(
        {"edits": [fact] * 101, "target_layer": 17}
    )

    assert len(request.edits) == 101
    assert request.sequential_edit is True


def test_rome_sequential_request_can_start_from_clean_patch_stack() -> None:
    request = RomeSequentialEditsRequest.model_validate(
        {
            "edits": [{"subject": "Example", "relation_prompt": "{} is", "new_object": "updated"}],
            "target_layer": 17,
            "sequential_edit": False,
        }
    )

    assert request.sequential_edit is False
