"""Small device helpers shared by model loading and worker cleanup.

Keep device-specific loading and precision policy in the adapter.
Import torch lazily so the API controller never owns an accelerator context.
"""
from __future__ import annotations



def mps_available() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def model_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if mps_available() else "cpu"


def release_mps_cache() -> None:
    import torch

    if mps_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
