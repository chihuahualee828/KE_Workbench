from __future__ import annotations

import re


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def subject_indices(prompt: str, subject: str) -> list[int]:
    prompt_tokens = simple_tokens(prompt)
    subject_tokens = simple_tokens(subject)
    if not subject_tokens:
        return []
    for start in range(0, len(prompt_tokens) - len(subject_tokens) + 1):
        if prompt_tokens[start : start + len(subject_tokens)] == subject_tokens:
            return list(range(start, start + len(subject_tokens)))
    return []

