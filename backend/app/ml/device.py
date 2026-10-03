"""Small device helpers shared by model loading and worker cleanup.

Keep device-specific loading and precision policy in the adapter.
Import torch lazily so the API controller never owns an accelerator context.
"""
from __future__ import annotations

import time
from typing import Any



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


def mps_memory() -> dict[str, Any]:
    import torch

    if not mps_available():
        return {"cuda_available": False, "mps_available": False}
    return {
        "cuda_available": False,
        "mps_available": True,
        "device": "mps",
        "device_name": "Apple GPU (MPS)",
        "memory_source": "torch.mps (worker process)",
        "allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        "sampled_at": time.time(),
    }
