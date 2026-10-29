from contextlib import nullcontext

import pytest
import torch
from safetensors.torch import save_file

from app.services.patch_service import PatchService
from app.storage.local_store import local_store


@pytest.fixture
def patches(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    return PatchService()


def make_model(dtype=torch.float32):
    model = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False, dtype=dtype) for _ in range(3)])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.tensor([[0.1, -0.1234], [0.0001, 0.9999]], dtype=dtype))
    return model


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_weights(model, expected):
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def dense_patch(service, patch_id, deltas, *, parents=()):
    path = service.patch_dir(patch_id) / "weight_deltas.safetensors"
    save_file(deltas, str(path))
    return service.create_patch(
        "demo", "rome", [], [], patch_id=patch_id, delta_paths=[str(path)],
        extra_metadata={"parent_patch_ids": list(parents), "sequential_edit": bool(parents)},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_overlapping_stack_unloads_and_replays_exactly(patches, dtype):
    model = make_model(dtype)
    clean = snapshot(model)
    delta = torch.full((2, 2), 0.3, dtype=dtype)
    # These values expose inverse-subtraction drift, unlike integer fixtures.
    assert not torch.equal((clean["0.weight"] + delta) - delta, clean["0.weight"])
    states = [clean]
    ids = []
    for index, layer in enumerate((0, 1, 0, 1)):
        patch_id = f"patch_{index}"
        dense_patch(patches, patch_id, {f"{layer}.weight": delta}, parents=ids)
        ids.append(patch_id)
        patches.load_saved_patch(patch_id)
        patches.sync_active_patches_to_model(model, "demo")
        states.append(snapshot(model))
    originals = dict(model._kew_patch_originals)
    assert set(originals) == {"0.weight", "1.weight"}
    assert all(value.device.type == "cpu" and value.dtype == dtype for value in originals.values())
    assert all(torch.equal(value, clean[name]) for name, value in originals.items())
    assert sum(t.numel() * t.element_size() for t in originals.values()) == 8 * delta.element_size()

    for _ in range(10):
        # A -> B -> C -> D: discard C and retain exactly A -> B.
        patches.discard_active_patch(ids[2])
        patches.sync_active_patches_to_model(model, "demo")
        assert_weights(model, states[2])
        assert all(model._kew_patch_originals[name] is value for name, value in originals.items())
        patches.load_saved_patch(ids[-1])
        patches.sync_active_patches_to_model(model, "demo")
        assert_weights(model, states[4])

    patches.discard_active_patch(ids[1])
    patches.sync_active_patches_to_model(model, "demo")
    assert_weights(model, states[1])
    assert set(model._kew_patch_originals) == {"0.weight"}
    patches.discard_active_patch(ids[0])
    patches.sync_active_patches_to_model(model, "demo")
    assert_weights(model, clean)
    assert model._kew_patch_originals == {}

    # Snapshots belong to a loaded model, never to another runtime with the same ID.
    reloaded = make_model(dtype)
    patches.load_saved_patch(ids[-1])
    patches.sync_active_patches_to_model(reloaded, "demo")
    assert_weights(reloaded, states[4])
    patches.sync_patch_stack_to_model(reloaded, "demo", [])
    assert_weights(reloaded, clean)


@pytest.mark.parametrize("orientation", ["row", "column"])
def test_sparse_kn_and_dense_patches_share_exact_originals(patches, orientation):
    model = make_model()
    clean = snapshot(model)
    path = patches.patch_dir("patch_kn") / "kn_deltas.safetensors"
    save_file({"vectors.indices": torch.tensor([0]), "vectors.values": torch.tensor([[0.3, -0.7]])}, str(path))
    patches.create_patch(
        "demo", "kn", [], [0], patch_id="patch_kn", delta_paths=[str(path)],
        extra_metadata={"sparse_delta_format": "kn_v1", "sparse_delta_specs": [
            {"entry": "vectors", "parameter": "0.weight", "orientation": orientation},
        ]},
    )
    dense_patch(patches, "patch_dense", {"0.weight": torch.full((2, 2), 0.4)})
    patches.sync_patch_stack_to_model(model, "demo", ["patch_kn"])
    sparse_state = snapshot(model)
    for _ in range(5):
        patches.sync_patch_stack_to_model(model, "demo", ["patch_kn", "patch_dense"])
        patches.sync_patch_stack_to_model(model, "demo", ["patch_kn"])
        assert_weights(model, sparse_state)
        patches.sync_patch_stack_to_model(model, "demo", [])
        assert_weights(model, clean)
        patches.sync_patch_stack_to_model(model, "demo", ["patch_kn"])


@pytest.mark.parametrize("use_active", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_temporary_inference_patch_restores_weights_on_exit(patches, use_active, failure):
    model = make_model()
    clean = snapshot(model)
    dense_patch(patches, "patch_active", {"0.weight": torch.full((2, 2), 0.3)})
    dense_patch(patches, "patch_preview", {"0.weight": torch.full((2, 2), -0.7)})
    patches.activate_patch("patch_active")
    patches.sync_active_patches_to_model(model, "demo")
    active = snapshot(model)
    for _ in range(5):
        with pytest.raises(RuntimeError, match="inference failed") if failure else nullcontext():
            with patches.applied_to_model(model, "patch_preview", "demo", use_active_patches=use_active):
                expected = active if use_active else clean
                assert torch.equal(model[0].weight, expected["0.weight"] - 0.7)
                if failure:
                    raise RuntimeError("inference failed")
        assert_weights(model, active)
        assert model._kew_active_patch_ids == ("patch_active",)
    patches.sync_patch_stack_to_model(model, "demo", [])
    assert_weights(model, clean)


def test_failed_transition_restores_exact_entry_weights_and_originals(patches, monkeypatch):
    model = make_model()
    clean = snapshot(model)
    for patch_id, layer in (("patch_a", 0), ("patch_b", 1)):
        dense_patch(patches, patch_id, {f"{layer}.weight": torch.full((2, 2), 0.3)})
    patches.sync_patch_stack_to_model(model, "demo", ["patch_a"])
    original = model._kew_patch_originals["0.weight"]
    # Simulate a cumulative series that continues editing an already active patch.
    with torch.no_grad():
        model[0].weight.add_(0.13)
    before = snapshot(model)
    apply = patches._apply_patch_to_model

    def fail_after_apply(model, patch):
        apply(model, patch)
        if patch["patch_id"] == "patch_b":
            raise RuntimeError("apply failed")

    monkeypatch.setattr(patches, "_apply_patch_to_model", fail_after_apply)
    for desired in (["patch_a", "patch_b"], ["patch_b"]):
        with pytest.raises(RuntimeError, match="apply failed"):
            patches.sync_patch_stack_to_model(model, "demo", desired)
        assert_weights(model, before)
        assert model._kew_active_patch_ids == ("patch_a",)
        assert model._kew_patch_originals == {"0.weight": original}
    patches.sync_patch_stack_to_model(model, "demo", [])
    assert_weights(model, clean)


@pytest.mark.parametrize("use_active", [False, True])
def test_failed_preview_setup_restores_active_stack(patches, monkeypatch, use_active):
    model = make_model()
    dense_patch(patches, "patch_active", {"0.weight": torch.full((2, 2), 0.3)})
    dense_patch(patches, "patch_preview", {"0.weight": torch.full((2, 2), -0.7)})
    patches.activate_patch("patch_active")
    patches.sync_active_patches_to_model(model, "demo")
    before = snapshot(model)
    with pytest.raises(FileNotFoundError):
        with patches.applied_to_model(model, "patch_missing", "demo", use_active_patches=use_active):
            pytest.fail("Missing patch must fail before inference")
    assert_weights(model, before)
    apply = patches._apply_patch_to_model

    def fail_after_apply(model, patch):
        apply(model, patch)
        if patch["patch_id"] == "patch_preview":
            raise RuntimeError("preview failed")

    monkeypatch.setattr(patches, "_apply_patch_to_model", fail_after_apply)
    with pytest.raises(RuntimeError, match="preview failed"):
        with patches.applied_to_model(model, "patch_preview", "demo", use_active_patches=use_active):
            pytest.fail("Failed patch must not reach inference")
    assert_weights(model, before)
    assert model._kew_active_patch_ids == ("patch_active",)


def test_append_does_not_reapply_parents(patches, monkeypatch):
    model = make_model()
    for patch_id in ("patch_a", "patch_b"):
        dense_patch(patches, patch_id, {"0.weight": torch.full((2, 2), 0.3)})
    patches.sync_patch_stack_to_model(model, "demo", ["patch_a"])
    original = model._kew_patch_originals["0.weight"]
    apply = patches._apply_patch_to_model
    calls = []

    def record(model, patch):
        calls.append(patch["patch_id"])
        apply(model, patch)

    monkeypatch.setattr(patches, "_apply_patch_to_model", record)
    patches.sync_patch_stack_to_model(model, "demo", ["patch_a", "patch_b"])
    assert calls == ["patch_b"]
    assert model._kew_patch_originals["0.weight"] is original


def test_baked_lineage_is_part_of_the_clean_base(patches):
    model = make_model()
    baked = snapshot(model)
    patches.create_patch("demo", "rome", [], [], patch_id="patch_baked", delta_paths=[],
                         extra_metadata={"baked_into_model": True, "lineage_only": True})
    dense_patch(patches, "patch_runtime", {"0.weight": torch.full((2, 2), 0.3)})
    patches.sync_patch_stack_to_model(model, "demo", ["patch_baked", "patch_runtime"])
    patches.sync_patch_stack_to_model(model, "demo", ["patch_baked"])
    assert_weights(model, baked)
    assert model._kew_patch_originals == {}
