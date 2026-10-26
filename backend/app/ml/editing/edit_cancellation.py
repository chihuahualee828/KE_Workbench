"""Cooperative edit cancellation, with rollback owned by the model worker.

The job stays running until cleanup finishes. Separate control files avoid
lost cancellation requests when the parent process appends job logs. A small
cross-process lock serializes cancellation with the final commit decision.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import fcntl
import shutil
import tempfile
from typing import Any

from app.storage.local_store import local_store
from app.ml.editing.random_state import DEFAULT_EDIT_SEED, edit_random_state, validate_edit_seed


class EditCanceled(RuntimeError):
    pass


current_edit: ContextVar[EditTransaction | None] = ContextVar("current_edit", default=None)


@contextmanager
def control_lock(job_id: str):
    if not job_id.startswith("job_") or not job_id.replace("_", "").isalnum():
        raise ValueError("Invalid edit job ID")
    path = local_store.root / "experiments" / job_id
    path.mkdir(parents=True, exist_ok=True)
    with (path / "edit.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def request_cancel(job_id: str) -> bool:
    with control_lock(job_id) as path:
        if (path / "edit.finished").exists():
            return False
        (path / "edit.cancel").touch()
        return True


def check_edit_cancellation() -> None:
    transaction = current_edit.get()
    if transaction is not None:
        transaction.check()


@contextmanager
def memit_cancellation_hook(model: Any, *, enabled: bool):
    """Interrupt MEMIT at forward boundaries without modifying EasyEdit.

    Use a model-local hook, not a global hook or a patched EasyEdit function.
    The exception unwinds EasyEdit's TraceDict contexts before our transaction
    restores weights. Backward passes and matrix solves already in progress
    still finish before the next check. No hook is installed for other methods
    or for calls that do not belong to a cancellable edit job.
    """
    transaction = current_edit.get()
    if not enabled or transaction is None:
        yield
        return

    transaction.check()
    # MEMIT freezes model parameters to optimize its separate delta tensor.
    # Preserve mixed flags rather than unconditionally unfreezing the model.
    gradient_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    training_flags = [(module, module.training) for module in model.modules()]

    def before_forward(_module, _args):
        transaction.check()

    handle = model.register_forward_pre_hook(before_forward, prepend=True)
    try:
        yield
    finally:
        handle.remove()
        for parameter, requires_grad in gradient_flags:
            parameter.requires_grad_(requires_grad)
        for module, training in training_flags:
            module.training = training


class EditTransaction:
    def __init__(self, job_id: str, on_patch_state_changed=None, *, seed: int | None = None):
        self.job_id = job_id
        self.seed = seed
        self.on_patch_state_changed = on_patch_state_changed
        self.model: Any = None
        self.model_id = ""
        self.weights: dict[str, Any] = {}
        self.patch_ids: list[str] = []
        self.created_patch_ids: set[str] = set()
        self.patch_backup: tuple[str, tempfile.TemporaryDirectory] | None = None

    def check(self) -> None:
        if (local_store.root / "experiments" / self.job_id / "edit.cancel").exists():
            raise EditCanceled("Edit canceled.")

    def notify_patch_state_changed(self) -> None:
        if self.on_patch_state_changed is not None:
            self.on_patch_state_changed()

    def capture(self, model: Any, model_id: str, names=()) -> None:
        """Capture only affected parameters, before resetting the old patch stack."""
        from app.services.patch_service import patch_service

        self.check()
        names = set(names)
        if self.model is None:
            self.model, self.model_id = model, model_id
            self.patch_ids = list(patch_service.active_patches_for_model(model_id))
            for patch_id in self.patch_ids:
                patch = patch_service.get_patch(patch_id)
                if patch.get("sparse_delta_format") == "kn_v1":
                    names.update(spec["parameter"] for spec in patch.get("sparse_delta_specs", []))
                else:
                    from safetensors import safe_open

                    for path in patch.get("delta_paths", []):
                        resolved = patch_service.resolve_delta_path(path)
                        if resolved.suffix == ".safetensors":
                            with safe_open(str(resolved), framework="pt", device="cpu") as tensors:
                                names.update(tensors.keys())
        from app.ml.editing.easyedit_runtime import _state_dict_for_weight_names

        state = _state_dict_for_weight_names(model, list(names))
        for name in names - self.weights.keys():
            self.weights[name] = state[name].detach().cpu().clone()

    def checkpoint(self) -> None:
        """A fully saved ROME fact becomes the next rollback boundary."""
        from app.services.patch_service import patch_service
        from app.ml.editing.easyedit_runtime import _state_dict_for_weight_names

        state = _state_dict_for_weight_names(self.model, list(self.weights))
        self.weights = {name: state[name].detach().cpu().clone() for name in self.weights}
        self.patch_ids = list(patch_service.active_patches_for_model(self.model_id))
        self.created_patch_ids.clear()
        self.discard_backup()
        self.notify_patch_state_changed()

    def protect_patch(self, patch_id: str) -> None:
        """Preserve the last series artifact if a later save fails partway."""
        if patch_id not in self.patch_ids:
            self.created_patch_ids.add(patch_id)
            return
        backup = tempfile.TemporaryDirectory(prefix="kew-edit-rollback-")
        self.patch_backup = (patch_id, backup)
        shutil.copytree(local_store.root / "patches" / patch_id, backup.name, dirs_exist_ok=True)

    def discard_backup(self) -> None:
        if self.patch_backup is not None:
            self.patch_backup[1].cleanup()
            self.patch_backup = None

    def rollback(self) -> None:
        if self.model is None:
            return
        import torch
        from app.services.patch_service import patch_service
        from app.ml.editing.easyedit_runtime import _state_dict_for_weight_names

        # Do not subtract deltas: copying snapshots avoids rounding drift and
        # also handles EasyEdit throwing before it returns an edited_model.
        with torch.no_grad():
            state = _state_dict_for_weight_names(self.model, list(self.weights))
            for name, original in self.weights.items():
                state[name].copy_(original.to(device=state[name].device, dtype=state[name].dtype))
        active = patch_service.active_patches_by_model()
        active[self.model_id] = list(self.patch_ids)
        patch_service.restore_active_patches(active)
        self.model._kew_active_patch_ids = tuple(self.patch_ids)
        if self.patch_backup is not None:
            patch_id, backup = self.patch_backup
            shutil.copytree(backup.name, local_store.root / "patches" / patch_id, dirs_exist_ok=True)
        for patch_id in self.created_patch_ids:
            # Only artifacts allocated by this transaction, never old patches.
            path = local_store.root / "patches" / patch_id
            if path.exists():
                shutil.rmtree(path)


def run_cancellable_edit(job_id: str, runner, *, seed: int = DEFAULT_EDIT_SEED, on_patch_state_changed=None, **payload):
    """Execute in either the in-process runtime or the model-owning worker."""
    from app.services.model_runtime_service import model_runtime_service

    validate_edit_seed(seed)
    # The parent delegates to a worker; only that worker owns model snapshots/RNGs.
    if getattr(model_runtime_service, "uses_hard_worker", False):
        transaction = EditTransaction(job_id, on_patch_state_changed, seed=seed)
        token = current_edit.set(transaction)
        try:
            transaction.check()
            return runner(**payload)
        except EditCanceled:
            return {"_edit_canceled": True}
        finally:
            current_edit.reset(token)
    with model_runtime_service.exclusive_runtime_access():
        return _run_local_edit(job_id, runner, payload, on_patch_state_changed, seed=seed)


def _run_local_edit(job_id: str, runner, payload, on_patch_state_changed=None, *, seed: int | None = None):
    transaction = EditTransaction(job_id, on_patch_state_changed, seed=seed)
    token = current_edit.set(transaction)
    try:
        transaction.check()
        with edit_random_state(seed) if seed is not None else nullcontext():
            result = runner(**payload)
        with control_lock(job_id) as path:
            transaction.check()
            (path / "edit.finished").touch()
        return result
    except EditCanceled:
        transaction.rollback()
        return {"_edit_canceled": True}
    except Exception:
        transaction.rollback()
        raise
    finally:
        transaction.discard_backup()
        current_edit.reset(token)
