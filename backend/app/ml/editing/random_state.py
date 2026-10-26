"""Job-local randomness for the persistent model worker (CPU, CUDA and MPS)."""
from contextlib import contextmanager
import importlib
import random

DEFAULT_EDIT_SEED = 42
MAX_EDIT_SEED = 2_147_483_647


def validate_edit_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed <= MAX_EDIT_SEED:
        raise ValueError(f"Random seed must be an integer from 0 to {MAX_EDIT_SEED}.")
    return seed


@contextmanager
def edit_random_state(seed: int):
    """Seed once per job and restore the surrounding process state on every exit.

    The caller holds exclusive runtime access. Import EasyEdit before applying
    the chosen seed, since its first import seeds these same global generators.
    This controls randomness, not numerical determinism or algorithm caches.
    """
    validate_edit_seed(seed)
    import numpy as np
    import torch
    from app.ml.editing.easyedit_runtime import _add_easyedit_to_python_path

    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        _add_easyedit_to_python_path()
        importlib.import_module("easyeditor")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
