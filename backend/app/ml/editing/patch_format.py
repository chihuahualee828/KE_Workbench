from __future__ import annotations

from typing import Any


def edit_to_patch_entry(subject: str, relation_prompt: str, new_object: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation_prompt": relation_prompt,
        "new_object": new_object,
    }

