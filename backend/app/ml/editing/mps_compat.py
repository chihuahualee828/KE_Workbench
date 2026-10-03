"""Workbench-owned FP32 policy for EasyEdit calls on Apple GPUs.

No dependency files or global torch functions are modified. CUDA/CPU calls do
not enter the thread-local mode, so their original precision is preserved.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def editing_precision(precision: str, device: str) -> str:
    if device.startswith("mps") and precision in {"float64", "double"}:
        return "float32"
    return precision


def configure_mps_hparams(hparams: Any) -> None:
    precision = getattr(hparams, "mom2_dtype", None)
    if precision is not None:
        effective = editing_precision(str(precision), "mps")
        if effective != precision:
            print(f"MPS editing: MOM2 precision {precision} -> {effective}.")
            hparams.mom2_dtype = effective


@contextmanager
def mps_fp32(device: str):
    """Keep EasyEdit's explicit FP64 casts on MPS in FP32, including MEMIT.

    Handle Tensor.double(), Tensor.to(dtype), and FP64 cached tensors being
    transferred to MPS. CPU-only tensors retain their precision. The mode exits
    even on exceptions; it never moves an MPS calculation onto CPU.
    """
    if not device.startswith("mps"):
        yield
        return

    import torch
    from torch.overrides import TorchFunctionMode

    logged = False

    def report_conversion():
        nonlocal logged
        if not logged:
            print("MPS editing: intercepted an FP64 cast or transfer; used FP32 instead.")
            logged = True

    class MpsFloat32(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            converted = False
            if func is torch.Tensor.double and args[0].device.type == "mps":
                result = args[0].float(**kwargs)
                report_conversion()
                return result
            if func is torch.Tensor.to:
                tensor = args[0]
                target = kwargs.get("device")
                if len(args) > 1:
                    other = args[1]
                    if isinstance(other, (str, torch.device)):
                        target = other
                    elif isinstance(other, torch.Tensor):
                        target = other.device
                to_mps = target is not None and str(target).startswith("mps")
                if tensor.device.type == "mps" or to_mps:
                    converted = (
                        any(isinstance(arg, torch.dtype) and arg == torch.float64 for arg in args)
                        or kwargs.get("dtype") == torch.float64
                        or (to_mps and tensor.dtype == torch.float64)
                    )
                    args = tuple(torch.float32 if isinstance(arg, torch.dtype) and arg == torch.float64 else arg for arg in args)
                    if kwargs.get("dtype") == torch.float64:
                        kwargs = {**kwargs, "dtype": torch.float32}
                    if to_mps and tensor.dtype == torch.float64:
                        # Cast before transfer; don't attempt an FP64 MPS allocation.
                        args = (tensor.float(), *args[1:])
            result = func(*args, **kwargs)
            if converted:
                report_conversion()
            return result

    with MpsFloat32():
        yield
