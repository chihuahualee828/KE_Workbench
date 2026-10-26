from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.db.schemas import (
    BatchWeightEditRequest, DpoEditRequest, KnEditRequest, RankOneEditRequest,
    RankOneSeriesRequest, TrainingEditRequest,
)


@pytest.mark.parametrize("schema", [RankOneEditRequest, RankOneSeriesRequest, KnEditRequest,
                                    BatchWeightEditRequest, TrainingEditRequest, DpoEditRequest])
def test_edit_seed_is_shared_validated_and_defaults_to_42(schema):
    fact = {"subject": "Paris", "relation_prompt": "{} is in", "new_object": "Italy",
            "ground_truth": "France", "target_neg": "France"}
    payload = {**fact, "edits": [fact], "target_layer": 0}
    assert schema.model_validate(payload).seed == 42
    for seed in (0, 123, 2_147_483_647):
        assert schema.model_validate({**payload, "seed": seed}).model_dump()["seed"] == seed
    for seed in (-1, 2_147_483_648, 1.5, True, "42", None):
        with pytest.raises(ValidationError):
            schema.model_validate({**payload, "seed": seed})


def test_memit_request_preserves_imported_source_metadata() -> None:
    request = BatchWeightEditRequest.model_validate(
        {
            "model_id": "gpt2-xl",
            "edits": [
                {
                    "subject": "Danielle Darrieux",
                    "relation_prompt": "The mother tongue of {} is",
                    "new_object": "English",
                    "ground_truth": "French",
                    "rephrase_prompts": ["Where Danielle Darrieux is from, people speak the language of"],
                    "source_metadata": {
                        "case_id": 0,
                        "prompt": "The mother tongue of Danielle Darrieux is",
                        "target_new": "English",
                        "locality_prompt": "Michel Rocard is a native speaker of",
                        "locality_ground_truth": "French",
                    },
                }
            ],
        }
    )

    assert request.edits[0].source_metadata["case_id"] == 0
    assert request.edits[0].source_metadata["locality_ground_truth"] == "French"


def test_rome_sequential_request_accepts_more_than_one_hundred_facts() -> None:
    fact = {
        "subject": "Example",
        "relation_prompt": "{} is",
        "new_object": "updated",
    }

    request = RankOneSeriesRequest.model_validate(
        {"edits": [fact] * 101, "target_layer": 17}
    )

    assert len(request.edits) == 101
    assert request.sequential_edit is True


def test_rome_sequential_request_can_start_from_clean_patch_stack() -> None:
    request = RankOneSeriesRequest.model_validate(
        {
            "edits": [{"subject": "Example", "relation_prompt": "{} is", "new_object": "updated"}],
            "target_layer": 17,
            "sequential_edit": False,
        }
    )

    assert request.sequential_edit is False
