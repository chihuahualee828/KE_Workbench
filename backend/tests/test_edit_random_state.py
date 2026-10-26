import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.ml.editing import random_state
from app.ml.editing.edit_cancellation import EditCanceled, _run_local_edit, current_edit
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


def draws():
    return random.random(), float(np.random.random()), float(torch.rand(()))


@pytest.fixture
def isolated_import(monkeypatch):
    from app.ml.editing import easyedit_runtime

    monkeypatch.setattr(easyedit_runtime, "_add_easyedit_to_python_path", lambda: None)
    # Simulate EasyEdit's initial import-time seed. Job seeding must follow it.
    def import_easyedit(name):
        assert name == "easyeditor"
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    monkeypatch.setattr(random_state.importlib, "import_module", import_easyedit)


@pytest.mark.parametrize("outcome", [None, RuntimeError, EditCanceled])
def test_seeded_job_restores_rngs_on_every_exit(isolated_import, monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(local_store, "root", tmp_path)
    python_state, numpy_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected = draws()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)

    def run():
        draws()
        assert current_edit.get().seed == 123
        if outcome:
            raise outcome("stopped")
        patch = patch_service.create_patch("tiny", "ft", [], [])
        return {"patch": patch}

    if outcome is RuntimeError:
        with pytest.raises(RuntimeError, match="stopped"):
            _run_local_edit("job_seed", run, {}, seed=123)
    else:
        result = _run_local_edit("job_seed", run, {}, seed=123)
        if outcome is EditCanceled:
            assert result == {"_edit_canceled": True}
        else:
            assert result["patch"]["seed"] == 123
            assert patch_service.get_patch(result["patch"]["patch_id"])["seed"] == 123
    assert draws() == expected
    assert current_edit.get() is None


def test_one_random_sequence_per_job_after_easyedit_import(isolated_import, monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    def run():
        # Batches or sequential facts consume the same advancing stream.
        return {"batches": [draws() for _ in range(3)]}
    first = _run_local_edit("job_first", run, {}, seed=123)
    draws()  # Unrelated activity between jobs must not affect the chosen seed.
    other = _run_local_edit("job_other", run, {}, seed=124)
    repeated = _run_local_edit("job_repeated", run, {}, seed=123)
    assert first == repeated
    assert first != other
    assert len(set(first["batches"])) == 3
    assert first["batches"][0][0] == random.Random(123).random()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_device_rng_is_repeatable_and_restored(isolated_import, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    backend = getattr(torch, device)
    before = backend.get_rng_state().clone()
    with random_state.edit_random_state(123):
        first = torch.rand(8, device=device).cpu()
    assert torch.equal(before, backend.get_rng_state())
    with random_state.edit_random_state(123):
        repeated = torch.rand(8, device=device).cpu()
    assert torch.equal(first, repeated)
    assert torch.equal(before, backend.get_rng_state())


def test_parent_forwards_seed_without_touching_its_rngs(monkeypatch, tmp_path):
    from app.ml.editing import edit_cancellation
    from app.services import model_runtime_service as runtime_module

    monkeypatch.setattr(local_store, "root", tmp_path)
    monkeypatch.setattr(runtime_module, "model_runtime_service", SimpleNamespace(uses_hard_worker=True))
    def forbidden(*args):
        pytest.fail("The parent must not initialize editing RNGs")
    monkeypatch.setattr(edit_cancellation, "edit_random_state", forbidden)
    def delegate():
        return {"seed": current_edit.get().seed}
    assert edit_cancellation.run_cancellable_edit("job_parent", delegate, seed=17) == {"seed": 17}
