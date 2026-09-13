from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from app.api import routes_editing
from app.api.routes_editing import _job_response, _sequential_edit_progress
from app.services.experiment_service import _append_job_logs


def test_job_log_window_retains_latest_lines_with_monotonic_cursor() -> None:
    job = {"logs": ["Job started"], "log_count": 1}

    _append_job_logs(job, [f"line {index}" for index in range(1, 501)])

    assert job["log_count"] == 501
    assert len(job["logs"]) == 500
    assert job["logs"][0] == "line 1"
    assert job["logs"][-1] == "line 500"


def test_wait_response_returns_only_new_logs_or_the_retained_window() -> None:
    job = {
        "id": "job_demo",
        "experiment_id": "exp_demo",
        "status": "running",
        "progress": 0.5,
        "logs": ["line 2", "line 3"],
        "log_count": 3,
    }

    current = _job_response(job, after_log_count=2)
    expired = _job_response(job, after_log_count=0)

    assert current.logs == ["line 3"]
    assert current.log_start_count == 2
    assert expired.logs == ["line 2", "line 3"]
    assert expired.log_start_count == 1


def test_sequential_edit_progress_uses_completed_fact_count() -> None:
    assert _sequential_edit_progress("Created ROME sequential edit 3/10.") == (3, 10)
    assert _sequential_edit_progress("Created R-ROME sequential edit 10/10.") == (10, 10)
    assert _sequential_edit_progress("Loaded EasyEdit hparams.") is None


def test_job_log_stream_sends_captured_output_only_to_the_job_log(monkeypatch, capsys) -> None:
    captured_logs: list[str] = []
    monkeypatch.setattr(routes_editing, "settings", SimpleNamespace(edit_log_to_terminal=False))
    monkeypatch.setattr(
        routes_editing.experiment_service,
        "append_job_log",
        lambda _job_id, message, **_kwargs: captured_logs.append(message),
    )
    terminal = StringIO()
    stream = routes_editing._JobLogStream("job_demo", terminal)

    stream.write("EasyEdit progress\n")
    stream.flush()

    assert captured_logs == ["EasyEdit progress"]
    assert terminal.getvalue() == ""
    captured_terminal = capsys.readouterr()
    assert captured_terminal.out == ""
    assert captured_terminal.err == ""


def test_job_log_stream_can_mirror_output_to_the_backend_terminal(monkeypatch) -> None:
    captured_logs: list[str] = []
    monkeypatch.setattr(
        routes_editing.experiment_service,
        "append_job_log",
        lambda _job_id, message, **_kwargs: captured_logs.append(message),
    )
    monkeypatch.setattr(routes_editing, "settings", SimpleNamespace(edit_log_to_terminal=True))
    terminal = StringIO()
    stream = routes_editing._JobLogStream("job_demo", terminal)

    stream.write("EasyEdit progress\n")
    stream.flush()

    assert terminal.getvalue() == "EasyEdit progress\n"
    assert captured_logs == ["EasyEdit progress"]
