from types import SimpleNamespace

import pytest

from app.api import routes_editing
from app.ml.editing.batch_progress import track_batch_progress
from app.ml.editing.edit_cancellation import EditCanceled


@pytest.mark.parametrize("failure", [RuntimeError, EditCanceled])
def test_failed_batch_does_not_advance_progress_and_restores_callback(capsys, failure):
    calls = []

    def apply(requests):
        calls.append(requests)
        if len(calls) == 2:
            raise failure("Interrupted")
        return requests

    editor = SimpleNamespace(apply_algo=apply)
    with pytest.raises(failure):
        with track_batch_progress(editor, 5, 2):
            assert editor.apply_algo([1, 2]) == [1, 2]
            editor.apply_algo([3, 4])
    assert editor.apply_algo is apply
    assert capsys.readouterr().out.splitlines() == [
        "Applying edit batch 1/3.", "Completed edit batch 1/3.", "Applying edit batch 2/3.",
    ]


def test_batch_log_stream_preserves_counts_and_distinguishes_series(monkeypatch):
    updates = []
    monkeypatch.setattr(routes_editing.experiment_service, "append_job_log",
                        lambda job, line, **values: updates.append(values))
    stream = routes_editing._JobLogStream("job_test", None)
    # Exercise buffering used by worker output forwarding too.
    stream.write("Applying edit bat")
    assert not updates
    stream.write("ch 1/3.\nCompleted edit batch 1/3.\nApplying edit batch 2/3.\n")
    stream.write("Completed edit batch 3/3.\nCreated R-ROME sequential edit 1/3.\n")
    assert [(v["progress_current"], v["progress_total"], v["progress_phase"]) for v in updates] == [
        (0, 3, "batch_editing"), (1, 3, "batch_completed"), (1, 3, "batch_editing"),
        (3, 3, "batch_completed"), (1, 3, None),
    ]


@pytest.mark.parametrize("line", ["Applying edit batch 0/3.", "Completed edit batch 1/0.", "Completed edit batch 4/3."])
def test_invalid_batch_counters_do_not_change_progress(monkeypatch, line):
    updates = []
    monkeypatch.setattr(routes_editing.experiment_service, "append_job_log",
                        lambda job, line, **values: updates.append(values))
    routes_editing._JobLogStream("job_test", None)._emit(line)
    assert updates[0]["progress_current"] is None
    assert updates[0]["progress_phase"] is None
