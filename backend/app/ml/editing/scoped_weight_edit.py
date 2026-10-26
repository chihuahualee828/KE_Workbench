"""Job-scoped integration for EasyEdit methods with additional runtime state.

The existing methods keep their BaseEditor path. EAMET and SPHERE expose
algorithm APIs but are not registered with BaseEditor.batch_edit in EasyEdit.
"""
from contextlib import contextmanager
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from app.ml.editing.edit_cancellation import check_edit_cancellation
from app.ml.editing.methods import COVARIANCE_MODULES


def configure_scoped_layers(method, hparams, layers):
    """Keep EAMET's layer-specific covariance weights aligned with selection."""
    if method != "eamet" or not layers:
        return
    configured = dict(zip(hparams.layers, hparams.mom2_update_weight))
    if any(layer not in configured for layer in layers):
        raise ValueError("EAMET layers must be selected from its EasyEdit profile: "
                         + ", ".join(map(str, hparams.layers)))
    hparams.mom2_update_weight = [configured[layer] for layer in layers]


@contextmanager
def _job_state(method, editor):
    """Isolate upstream globals across models, jobs, failures, and reverts.

    History accumulates within this job only. Saved patches contain weight deltas,
    not an implicit lifelong editor session. The worker already serializes edits.
    """
    import torch

    module = importlib.import_module(COVARIANCE_MODULES[method])
    names = ("CONTEXT_TEMPLATES_CACHE", "COV_CACHE", "P", "P_loaded", "cache_c", "cache_c_new")
    missing = object()
    previous = {name: getattr(module, name, missing) for name in names}
    hparams = editor.hparams
    projection_path = getattr(hparams, "P_loc", missing)
    flags = [(p, p.requires_grad) for p in editor.model.parameters()]
    training = [(m, m.training) for m in editor.model.modules()]
    try:
        module.CONTEXT_TEMPLATES_CACHE = None
        module.COV_CACHE = {}
        module.P_loaded = False
        module.cache_c_new = False
        with TemporaryDirectory(prefix="kew-projection-") as directory:
            if method in {"alphaedit", "sphere"}:
                # Use EasyEdit's own projection computation with this job's model,
                # layers and statistics, rather than a shared relative P_loc file.
                projections = None
                for index, layer in enumerate(hparams.layers):
                    check_edit_cancellation()
                    projection = module.get_project(editor.model, editor.tok, layer, hparams).cpu()
                    if projections is None:
                        projections = torch.empty((len(hparams.layers), *projection.shape), dtype=projection.dtype)
                    projections[index].copy_(projection)
                    del projection
                hparams.P_loc = str(Path(directory) / "projection.pt")
                torch.save(projections, hparams.P_loc)
                del projections
            yield
    finally:
        if projection_path is not missing:
            hparams.P_loc = projection_path
        for name, value in previous.items():
            if value is missing:
                module.__dict__.pop(name, None)
            else:
                setattr(module, name, value)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, flag in training:
            module.training = flag


def run_scoped_batches(method, editor, edit_args):
    with _job_state(method, editor):
        if method == "alphaedit":
            return editor.batch_edit(**edit_args)
        return _algorithm_batches(editor, edit_args)


def _algorithm_batches(editor, edit_args):
    """Call EasyEdit's registered algorithm and evaluation APIs for each chunk."""
    from easyeditor.editors.utils import _prepare_requests, normalize_ground_truths
    from easyeditor.editors.editor import finalize_locality_metrics
    from easyeditor.evaluate import compute_edit_quality

    requests = _prepare_requests(
        edit_args["prompts"], edit_args["target_new"],
        normalize_ground_truths(edit_args["ground_truth"], edit_args["prompts"]),
        rephrase_prompts=edit_args.get("rephrase_prompts"),
        locality_inputs=edit_args.get("locality_inputs"),
        portability_inputs=edit_args.get("portability_inputs"),
        subject=edit_args["subject"],
    )
    metrics = []
    originals = {}

    def evaluate(request):
        check_edit_cancellation()
        return compute_edit_quality(editor.model, editor.model_name, editor.hparams,
                                    editor.tok, request, editor.hparams.device)

    for start in range(0, len(requests), editor.hparams.batch_size):
        chunk = requests[start:start + editor.hparams.batch_size]
        chunk_metrics = [{"case_id": start + i, "requested_rewrite": request,
                          "pre": evaluate(request)} for i, request in enumerate(chunk)]
        check_edit_cancellation()
        started = perf_counter()
        editor.model, originals = editor.apply_algo(
            editor.model, editor.tok, chunk, editor.hparams,
            copy=False, return_orig_weights=True,
        )
        elapsed = perf_counter() - started
        check_edit_cancellation()
        for request, metric in zip(chunk, chunk_metrics):
            metric.update(time=elapsed, post=evaluate(request))
            finalize_locality_metrics(metric, request, editor.hparams, editor.model_name)
        metrics.extend(chunk_metrics)
    return metrics, editor.model, originals
