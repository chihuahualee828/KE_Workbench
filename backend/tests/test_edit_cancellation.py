from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.ml.editing import easyedit_runtime as runtime
from app.ml.editing.edit_cancellation import (
    EditCanceled, EditTransaction, _run_local_edit, current_edit,
    memit_cancellation_hook, request_cancel,
)
from app.services.experiment_service import ExperimentService
from app.services.patch_service import patch_service
from app.storage.local_store import local_store


@pytest.fixture
def editing(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    hparams = SimpleNamespace(model_name="demo", layers=[0], rewrite_module_tmp="layers.{}", mom2_adjustment=False)
    state = SimpleNamespace(model=model, hparams=hparams, callback=None, calls=0, batch_calls=[], batch_args=None)

    class Editor:
        def __init__(self):
            self.model = model
            self.hparams = hparams

        def edit(self, **kwargs):
            state.calls += 1
            with torch.no_grad():
                model.layers[0].weight.add_(2)
            if state.callback:
                state.callback()
            return {}, model, {}

        def batch_edit(self, **kwargs):
            state.batch_args = kwargs
            assert kwargs["sequential_edit"] is True
            for offset in range(0, len(kwargs["prompts"]), self.hparams.batch_size):
                state.batch_calls.append(kwargs["prompts"][offset:offset + self.hparams.batch_size])
                self.apply_algo(**kwargs)
            return {}, model, {}

        apply_algo = edit

    monkeypatch.setattr(runtime, "_load_easyedit", lambda _: (
        SimpleNamespace(from_hparams=lambda _: Editor()), SimpleNamespace(from_hparams=lambda _: hparams)))
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "hf_id": "demo", "supports_rome": True, "supports_memit": True})
    monkeypatch.setattr(runtime, "_resolve_hparams_path", lambda *_: Path("unused.yaml"))
    for name in ("_configure_loaded_tokenizer", "_override_easyedit_device", "_configure_stats_dir", "_collect_runtime_memory"):
        monkeypatch.setattr(runtime, name, lambda *args: None)

    @contextmanager
    def access(_):
        yield {"model": model, "tokenizer": object(), "device": "cpu"}
    monkeypatch.setattr(runtime, "_easyedit_runtime_access", access)
    return state


def run_fact(method="memit", **kwargs):
    return runtime.run_easyedit(method, "demo", [
        {"subject": "A", "relation_prompt": "{} is", "new_object": "B"}
    ], [0], **kwargs)


def add_old_patch(editing):
    from safetensors.torch import save_file

    path = patch_service.patch_dir("patch_old") / "weight_deltas.safetensors"
    save_file({"layers.0.weight": torch.ones(2, 2), "layers.1.weight": torch.ones(2, 2) * 3}, str(path))
    patch_service.create_patch("demo", "rome", [], [0, 1], delta_paths=[str(path)], patch_id="patch_old")
    patch_service.activate_patch("patch_old")
    patch_service.sync_active_patches_to_model(editing.model, "demo")


@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("cancel_during", ["algorithm", "post_evaluation"])
def test_memit_cancel_restores_exact_prejob_weights_and_patches(editing, continuing, cancel_during):
    add_old_patch(editing)
    before = {name: value.clone() for name, value in editing.model.state_dict().items()}
    if cancel_during == "algorithm":
        editing.callback = lambda: request_cancel("job_test")

    def operation():
        result = run_fact(sequential_edit=continuing)
        request_cancel("job_test")  # Simulate cancellation during post-edit evaluation.
        return result

    assert _run_local_edit("job_test", operation, {}) == {"_edit_canceled": True}
    assert patch_service.active_patches_for_model("demo") == ["patch_old"]
    assert editing.model._kew_active_patch_ids == ("patch_old",)
    assert [patch["patch_id"] for patch in patch_service.list_patches()] == ["patch_old"]
    for name, value in editing.model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("continuing", [False, True])
def test_rome_series_cancel_keeps_only_completed_fact(editing, continuing):
    add_old_patch(editing)
    context = runtime.EasyEditSeriesContext(continue_from_active_patches=continuing)
    completed = {}

    def operation():
        result = run_fact("rome", patch_mode="series", series_context=context)
        completed.update(result)
        completed["weights"] = {name: value.clone() for name, value in editing.model.state_dict().items()}
        completed["artifact"] = (local_store.root / "patches" / result["patch_id"] / "weight_deltas.safetensors").read_bytes()
        editing.callback = lambda: request_cancel("job_test")
        return run_fact("rome", patch_mode="series", series_context=context)

    assert _run_local_edit("job_test", operation, {}) == {"_edit_canceled": True}
    expected = (["patch_old"] if continuing else []) + [completed["patch_id"]]
    assert patch_service.active_patches_for_model("demo") == expected
    patch = patch_service.get_patch(completed["patch_id"])
    assert patch["series_fact_count"] == 1
    assert len(patch["edits"]) == 1
    for name, value in editing.model.state_dict().items():
        assert torch.equal(value, completed["weights"][name])
    assert torch.equal(editing.model.layers[0].weight, torch.full((2, 2), 4. if continuing else 3.))
    assert (local_store.root / "patches" / patch["patch_id"] / "weight_deltas.safetensors").read_bytes() == completed["artifact"]


def test_cancel_before_first_fact_does_not_reset_existing_patches(editing):
    add_old_patch(editing)
    request_cancel("job_test")
    assert _run_local_edit("job_test", run_fact, {}) == {"_edit_canceled": True}
    assert editing.calls == 0
    assert patch_service.active_patches_for_model("demo") == ["patch_old"]


def test_easyedit_exception_before_return_restores_weights_even_without_job(editing):
    def fail():
        raise RuntimeError("algorithm failed after mutating weights")
    editing.callback = fail
    with pytest.raises(RuntimeError, match="algorithm failed"):
        run_fact()
    assert torch.equal(editing.model.layers[0].weight, torch.ones(2, 2))


def test_series_save_failure_restores_previous_artifact(editing, monkeypatch):
    context = runtime.EasyEditSeriesContext()
    saved = {}

    def operation():
        result = run_fact("rome", patch_mode="series", series_context=context)
        path = local_store.root / "patches" / result["patch_id"] / "weight_deltas.safetensors"
        saved.update(path=path, data=path.read_bytes())
        def fail(*args, **kwargs):
            raise RuntimeError("metadata write failed")
        monkeypatch.setattr(patch_service, "update_patch", fail)
        return run_fact("rome", patch_mode="series", series_context=context)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        _run_local_edit("job_test", operation, {})
    assert torch.equal(editing.model.layers[0].weight, torch.full((2, 2), 3.))
    assert saved["path"].read_bytes() == saved["data"]


def test_finished_edit_cannot_be_canceled(editing):
    result = _run_local_edit("job_test", run_fact, {})
    assert "patch_id" in result
    assert not request_cancel("job_test")
    assert current_edit.get() is None


def test_reset_notification_precedes_algorithm_and_finished_edit_survives_evaluation_cancel(editing):
    add_old_patch(editing)
    notified = []
    editing.callback = lambda: notified.append(("algorithm", patch_service.active_patches_for_model("demo")))
    result = _run_local_edit(
        "job_edit", run_fact, {"sequential_edit": False},
        lambda: notified.append(("reset", patch_service.active_patches_for_model("demo"))),
    )
    assert notified == [("reset", []), ("algorithm", [])]
    saved_weights = {name: value.clone() for name, value in editing.model.state_dict().items()}
    assert not request_cancel("job_edit")
    service = ExperimentService()
    experiment = service.create("evaluation", "demo", {"source": "post_edit"})
    evaluation = service.create_job(experiment["id"])
    service.cancel_job(evaluation["id"])
    with pytest.raises(runtime.EasyEditRuntimeError, match="Evaluation canceled"):
        runtime._ensure_evaluation_job_running(evaluation["id"])
    assert patch_service.active_patches_for_model("demo") == [result["patch_id"]]
    for name, value in editing.model.state_dict().items():
        assert torch.equal(value, saved_weights[name])


def test_worker_reports_patch_reset_as_structured_event(editing, monkeypatch):
    import app.services.model_runtime_service as runtime_service
    from app.services.model_worker import _dispatch
    from app.ml.editing import memit_wrapper

    worker = SimpleNamespace(exclusive_runtime_access=nullcontext)
    monkeypatch.setattr(runtime_service, "model_runtime_service", worker)
    monkeypatch.setattr(memit_wrapper, "run_memit_edit", lambda **_: run_fact(sequential_edit=False))
    events = []
    result = _dispatch(worker, "memit_edit", {"model_id": "demo", "_edit_job_id": "job_worker", "seed": 123}, events.append)
    assert result["patch_id"]
    assert patch_service.get_patch(result["patch_id"])["seed"] == 123
    assert events == [{"patch_state_changed": True}]


def test_cancel_request_stays_running_until_cleanup_acknowledged(monkeypatch, tmp_path):
    monkeypatch.setattr(local_store, "root", tmp_path)
    service = ExperimentService()
    experiment = service.create("memit_edit", "demo", {})
    job = service.create_job(experiment["id"])
    pending = service.request_edit_cancellation(job["id"])
    assert pending["cancel_requested"]
    assert pending["status"] == "running"
    service.append_job_log(job["id"], "Still in EasyEdit")
    assert service.get_job(job["id"])["cancel_requested"]
    with pytest.raises(RuntimeError, match="running experiment"):
        service.delete_experiment(experiment["id"], job["id"])
    service.cancel_job(job["id"], message=None)
    assert service.get_job(job["id"])["status"] == "canceled"
    service.delete_experiment(experiment["id"], job["id"])
    assert not (tmp_path / "experiments" / job["id"]).exists()


def test_worker_dispatch_runs_cancellation_in_model_owning_process(editing, monkeypatch):
    import app.services.model_runtime_service as runtime_service
    from app.services.model_worker import _dispatch
    from app.ml.editing import memit_wrapper

    worker_runtime = SimpleNamespace(exclusive_runtime_access=nullcontext)
    monkeypatch.setattr(runtime_service, "model_runtime_service", worker_runtime)
    monkeypatch.setattr(memit_wrapper, "run_memit_edit", lambda **_: run_fact())
    editing.callback = lambda: request_cancel("job_worker")
    result = _dispatch(worker_runtime, "memit_edit", {"model_id": "demo", "_edit_job_id": "job_worker"})
    assert result == {"_edit_canceled": True}
    assert torch.equal(editing.model.layers[0].weight, torch.ones(2, 2))
    assert current_edit.get() is None


def test_route_only_marks_canceled_after_rollback(editing, monkeypatch):
    from app.api import routes_editing
    import app.services.model_runtime_service as runtime_service

    monkeypatch.setattr(runtime_service, "model_runtime_service", SimpleNamespace(exclusive_runtime_access=nullcontext))
    service = routes_editing.experiment_service
    experiment = service.create("memit_edit", "demo", {})
    job = service.create_job(experiment["id"])

    def cancel_in_algorithm():
        pending = routes_editing.cancel_edit_job(job["id"])
        assert pending.status == "running"
        assert pending.cancel_requested
        assert torch.equal(editing.model.layers[0].weight, torch.full((2, 2), 3.))

    editing.callback = cancel_in_algorithm
    routes_editing._run_edit_job("MEMIT", job["id"], run_fact, {})
    assert service.get_job(job["id"])["status"] == "canceled"
    assert torch.equal(editing.model.layers[0].weight, torch.ones(2, 2))
    assert patch_service.list_patches() == []


def test_parent_attaches_cancellation_context_to_worker_command(monkeypatch):
    from app.ml.editing.edit_cancellation import EditTransaction
    from app.services.model_runtime_service import HardUnloadModelRuntimeService

    service = HardUnloadModelRuntimeService()
    monkeypatch.setattr(service, "require_loaded_runtime_bundle", lambda _: {})
    captured = {}
    def call(action, payload, **kwargs):
        captured.update(payload)
        return {"ok": True}
    monkeypatch.setattr(service, "_call_worker_locked", call)
    token = current_edit.set(EditTransaction("job_parent", seed=123))
    try:
        payload = {"model_id": "demo"}
        assert service.execute("memit_edit", payload) == {"ok": True}
        assert captured["_edit_job_id"] == "job_parent"
        assert captured["seed"] == 123
        assert "seed" not in payload
        assert "_edit_job_id" not in payload
    finally:
        current_edit.reset(token)


@pytest.mark.parametrize("continuing", [False, True])
def test_memit_hook_cancels_inside_edit_and_model_remains_usable(editing, continuing):
    add_old_patch(editing)
    model = editing.model
    model.forward = lambda inputs: model.layers[1](model.layers[0](inputs))
    model.layers[0].weight.requires_grad_(False)
    model.eval()
    model.layers[1].train()
    original_weights = {name: value.clone() for name, value in model.state_dict().items()}
    original_gradients = [parameter.requires_grad for parameter in model.parameters()]
    original_training = [module.training for module in model.modules()]
    forward_calls = []
    existing_hook = model.register_forward_pre_hook(lambda *_: forward_calls.append(True))
    original_hooks = dict(model._forward_pre_hooks)
    finished_steps = []

    def optimize():
        # Model the repeated forwards and parameter freezing in MEMIT compute_z.
        model.requires_grad_(False)
        model.train()
        for step in range(5):
            model(torch.ones(1, 2))
            finished_steps.append(step)
            request_cancel("job_test")
        pytest.fail("MEMIT must stop before finishing its remaining iterations")

    editing.callback = optimize
    try:
        result = _run_local_edit("job_test", lambda: run_fact(sequential_edit=continuing), {})
        assert result == {"_edit_canceled": True}
        assert finished_steps == [0]
        assert forward_calls == [True]  # Cancellation precedes existing hooks.
        assert model._forward_pre_hooks == original_hooks
        assert [parameter.requires_grad for parameter in model.parameters()] == original_gradients
        assert [module.training for module in model.modules()] == original_training
        for name, value in model.state_dict().items():
            assert torch.equal(value, original_weights[name])
        assert patch_service.active_patches_for_model("demo") == ["patch_old"]
        assert [patch["patch_id"] for patch in patch_service.list_patches()] == ["patch_old"]
        # The cancel marker still exists, but no hook leaks into later inference.
        assert torch.isfinite(model(torch.ones(1, 2))).all()
    finally:
        existing_hook.remove()


@pytest.mark.parametrize("method", ["rome", "r_rome"])
def test_other_edit_methods_do_not_get_memit_forward_hooks(editing, monkeypatch, method):
    monkeypatch.setattr(runtime.model_registry, "get", lambda _: {
        "hf_id": "demo", "supports_rome": True, "supports_r_rome": True})
    editing.model.forward = lambda inputs: editing.model.layers[0](inputs)
    finished_steps = []

    def optimize():
        request_cancel("job_test")
        assert not editing.model._forward_pre_hooks
        for step in range(3):
            editing.model(torch.ones(1, 2))
            finished_steps.append(step)

    editing.callback = optimize
    assert _run_local_edit("job_test", lambda: run_fact(method), {}) == {"_edit_canceled": True}
    assert finished_steps == [0, 1, 2]  # Existing post-EasyEdit boundary is unchanged.
    assert editing.batch_calls == []


@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("batch_size,chunk_sizes", [(1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (10, [5])])
def test_memit_batches_accumulate_in_one_replayable_patch(editing, continuing, batch_size, chunk_sizes):
    add_old_patch(editing)
    edits = [{"subject": str(i), "relation_prompt": "{} is", "new_object": "B"} for i in range(5)]
    result = _run_local_edit("job_test", lambda: runtime.run_easyedit(
        "memit", "demo", edits, [0], sequential_edit=continuing, batch_size=batch_size,
        rephrase_prompts=[f"rephrase {i}" for i in range(5)],
    ), {})
    assert [len(chunk) for chunk in editing.batch_calls] == chunk_sizes
    assert editing.batch_args["rephrase_prompts"] == [f"rephrase {i}" for i in range(5)]
    assert result["patch"]["batch_size"] == batch_size
    assert len(result["patch"]["edits"]) == 5
    expected = torch.full((2, 2), float((2 if continuing else 1) + 2 * len(chunk_sizes)))
    assert torch.equal(editing.model.layers[0].weight, expected)
    assert patch_service.active_patches_for_model("demo") == (["patch_old"] if continuing else []) + [result["patch_id"]]
    assert len(patch_service.list_patches()) == 2  # Old patch plus one whole-job patch.
    # Reloading the saved delta must give the same accumulated result.
    patch_service.sync_patch_stack_to_model(editing.model, "demo", [])
    patch_service.sync_active_patches_to_model(editing.model, "demo")
    assert torch.equal(editing.model.layers[0].weight, expected)


@pytest.mark.parametrize("continuing", [False, True])
def test_memit_cancel_in_later_batch_rolls_back_completed_batches(editing, continuing):
    add_old_patch(editing)
    before = {name: tensor.clone() for name, tensor in editing.model.state_dict().items()}
    def cancel_second_batch():
        if editing.calls == 2:
            request_cancel("job_test")
            # A forward inside the second batch aborts before a third batch.
            editing.model.layers[0].weight.requires_grad_(False)
            editing.model(torch.ones(1, 2))
    editing.model.forward = lambda inputs: editing.model.layers[0](inputs)
    editing.callback = cancel_second_batch
    edits = [{"subject": str(i), "relation_prompt": "{} is", "new_object": "B"} for i in range(5)]
    result = _run_local_edit("job_test", lambda: runtime.run_easyedit(
        "memit", "demo", edits, [0], sequential_edit=continuing, batch_size=2,
    ), {})
    assert result == {"_edit_canceled": True}
    assert len(editing.batch_calls) == 2
    for name, tensor in editing.model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert patch_service.active_patches_for_model("demo") == ["patch_old"]
    assert len(patch_service.list_patches()) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_memit_hook_restores_flags_and_hooks_on_success_or_error(editing, failure):
    model = editing.model
    model.layers[0].weight.requires_grad_(False)
    model.layers[1].eval()
    gradients = [parameter.requires_grad for parameter in model.parameters()]
    training = [module.training for module in model.modules()]

    def algorithm():
        assert len(model._forward_pre_hooks) == 1
        model.requires_grad_(False)
        model.eval()
        if failure:
            raise RuntimeError("optimizer failure")

    editing.callback = algorithm
    if failure:
        with pytest.raises(RuntimeError, match="optimizer failure"):
            _run_local_edit("job_test", run_fact, {})
    else:
        assert "patch_id" in _run_local_edit("job_test", run_fact, {})
    assert not model._forward_pre_hooks
    assert [parameter.requires_grad for parameter in model.parameters()] == gradients
    assert [module.training for module in model.modules()] == training


def test_memit_hook_does_nothing_without_a_cancellable_job(editing):
    with memit_cancellation_hook(editing.model, enabled=True):
        assert not editing.model._forward_pre_hooks


def test_cancellation_unwinds_unmodified_easyedit_trace_hooks(editing):
    import runpy
    from app.config import settings

    source = settings.easyedit_root / "easyeditor" / "util" / "nethook.py"
    if not source.exists():
        pytest.skip("Requires the EasyEdit checkout (read-only)")
    trace_dict = runpy.run_path(str(source))["TraceDict"]
    model = editing.model
    model.forward = lambda inputs: model.layers[0](inputs)
    token = current_edit.set(EditTransaction("job_test"))
    existing_hook = model.layers[0].register_forward_hook(lambda *_: None)
    original_hooks = dict(model.layers[0]._forward_hooks)
    try:
        with pytest.raises(EditCanceled):
            with memit_cancellation_hook(model, enabled=True):
                # Real EasyEdit context removes its own instrumentation when the
                # model's pre-hook raises. Existing user hooks must survive.
                with trace_dict(model, ["layers.0"]):
                    request_cancel("job_test")
                    model(torch.ones(1, 2))
        assert model.layers[0]._forward_hooks == original_hooks
        assert not model._forward_pre_hooks
    finally:
        existing_hook.remove()
        current_edit.reset(token)
