from __future__ import annotations


def indirect_effect(restored_probability: float, corrupted_probability: float) -> float:
    return round(restored_probability - corrupted_probability, 4)

