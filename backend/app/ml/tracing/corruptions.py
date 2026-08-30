from __future__ import annotations


def describe_corruption(corruption: dict) -> str:
    if corruption.get("type") == "gaussian_noise":
        return f"Gaussian noise x{corruption.get('noise_level', 3.0)}"
    return corruption.get("type", "unknown")

