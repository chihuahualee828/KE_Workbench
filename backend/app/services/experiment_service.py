from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.storage.local_store import local_store

_MAX_JOB_LOG_LINES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def create(self, experiment_type: str, model_id: str, input_json: dict[str, Any]) -> dict[str, Any]:
        experiment_id = f"exp_{uuid4().hex[:10]}"
        experiment = {
            "id": experiment_id,
            "type": experiment_type,
            "model_id": model_id,
            "status": "running",
            "input_json": input_json,
            "result_path": None,
            "error_message": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._condition:
            local_store.write_json(f"experiments/{experiment_id}.json", experiment)
            self._condition.notify_all()
        return experiment

    def record_completed(
        self,
        experiment_type: str,
        model_id: str,
        input_json: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        experiment = self.create(experiment_type, model_id, input_json)
        result_path = local_store.write_json(
            f"experiments/{experiment['id']}/result.json",
            result,
        )
        experiment["status"] = "completed"
        experiment["result_path"] = result_path
        experiment["updated_at"] = _now()
        local_store.write_json(f"experiments/{experiment['id']}.json", experiment)
        return experiment

    def list_experiments(self) -> list[dict[str, Any]]:
        experiments = [
            item for item in local_store.list_json("experiments")
            if item.get("id", "").startswith("exp_")
        ]
        return sorted(experiments, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return local_store.read_json(f"experiments/{experiment_id}.json")
        except FileNotFoundError:
            return None

    def get_experiment_details(self, experiment_id: str) -> dict[str, Any] | None:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return None

        job = next(
            (
                item
                for item in local_store.list_json("experiments")
                if item.get("experiment_id") == experiment_id and item.get("id", "").startswith("job_")
            ),
            None,
        )
        result = None
        result_path = experiment.get("result_path")
        if result_path:
            try:
                result = local_store.read_json(str(Path(result_path).relative_to(local_store.root)))
            except (FileNotFoundError, ValueError):
                result = None
        return {**experiment, "job": job, "result": result}

    def clear_history(self) -> dict[str, int]:
        with self._condition:
            active_count = sum(item.get("status") == "running" for item in self.list_experiments())
            if active_count:
                raise RuntimeError("Wait for running experiments to finish before clearing history.")
            experiments_root = local_store.root / "experiments"
            experiment_count = len(self.list_experiments())
            if experiments_root.exists():
                shutil.rmtree(experiments_root)
            experiments_root.mkdir(parents=True, exist_ok=True)
            self._condition.notify_all()
        return {"deleted_experiment_count": experiment_count}

    def delete_experiment(self, experiment_id: str) -> dict[str, str]:
        if not experiment_id.startswith("exp_") or Path(experiment_id).name != experiment_id:
            raise FileNotFoundError(experiment_id)

        with self._condition:
            experiment = self.get_experiment(experiment_id)
            if experiment is None:
                raise FileNotFoundError(experiment_id)
            if experiment.get("status") == "running":
                raise RuntimeError("Wait for the running experiment to finish before deleting it.")

            experiments_root = local_store.root / "experiments"
            related_jobs = [
                str(item["id"])
                for item in local_store.list_json("experiments")
                if item.get("id", "").startswith("job_") and item.get("experiment_id") == experiment_id
            ]
            for job_id in related_jobs:
                (experiments_root / f"{job_id}.json").unlink(missing_ok=True)
            (experiments_root / f"{experiment_id}.json").unlink(missing_ok=True)
            shutil.rmtree(experiments_root / experiment_id, ignore_errors=True)
            self._condition.notify_all()
        return {"experiment_id": experiment_id}

    def reconcile_interrupted_jobs(self) -> int:
        """Mark persisted running jobs as failed after a backend restart."""
        interruption_message = "Backend restarted before this job completed."
        with self._condition:
            records = local_store.list_json("experiments")
            interrupted_experiments = [
                item for item in records
                if item.get("id", "").startswith("exp_") and item.get("status") == "running"
            ]
            interrupted_ids = {item["id"] for item in interrupted_experiments}
            if not interrupted_ids:
                return 0

            for job in records:
                if job.get("id", "").startswith("job_") and job.get("experiment_id") in interrupted_ids and job.get("status") == "running":
                    job["status"] = "failed"
                    job["progress"] = 1.0
                    job["logs"] = [*job.get("logs", []), interruption_message][-500:]
                    job["updated_at"] = _now()
                    local_store.write_json(f"experiments/{job['id']}.json", job)

            for experiment in interrupted_experiments:
                experiment["status"] = "failed"
                experiment["error_message"] = interruption_message
                experiment["updated_at"] = _now()
                local_store.write_json(f"experiments/{experiment['id']}.json", experiment)

            self._condition.notify_all()
        return len(interrupted_experiments)

    def create_job(self, experiment_id: str) -> dict[str, Any]:
        job_id = f"job_{uuid4().hex[:10]}"
        job = {
            "id": job_id,
            "experiment_id": experiment_id,
            "status": "running",
            "progress": 0.0,
            "logs": ["Job started"],
            "log_count": 1,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._condition:
            experiment = local_store.read_json(f"experiments/{experiment_id}.json")
            experiment["job_id"] = job_id
            experiment["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            local_store.write_json(f"experiments/{experiment_id}.json", experiment)
            self._condition.notify_all()
        return job

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        with self._condition:
            job = local_store.read_json(f"experiments/{job_id}.json")
            if job.get("status") != "running":
                return
            experiment = local_store.read_json(f"experiments/{job['experiment_id']}.json")
            result_path = local_store.write_json(
                f"experiments/{job['experiment_id']}/result.json",
                result,
            )
            job["status"] = "completed"
            job["progress"] = 1.0
            _append_job_logs(job, ["Job completed"])
            job["updated_at"] = _now()
            experiment["status"] = "completed"
            experiment["result_path"] = result_path
            experiment["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            local_store.write_json(f"experiments/{experiment['id']}.json", experiment)
            self._condition.notify_all()

    def complete_paged_evaluation_job(
        self,
        job_id: str,
        result: dict[str, Any],
        overview: dict[str, Any],
        pages: list[dict[str, Any]],
    ) -> None:
        """Persist a large evaluation once, with small card pages for the UI."""
        with self._condition:
            job = local_store.read_json(f"experiments/{job_id}.json")
            if job.get("status") != "running":
                return
            experiment = local_store.read_json(f"experiments/{job['experiment_id']}.json")
            experiment_id = str(job["experiment_id"])
            result_path = local_store.write_json(
                f"experiments/{experiment_id}/result.json",
                result,
            )
            local_store.write_json(
                f"experiments/{experiment_id}/evaluation_overview.json",
                overview,
            )
            for page in pages:
                offset = int(page["offset"])
                local_store.write_json(
                    f"experiments/{experiment_id}/evaluation_pages/{offset}.json",
                    page,
                )
            job["status"] = "completed"
            job["progress"] = 1.0
            _append_job_logs(job, ["Job completed"])
            job["updated_at"] = _now()
            experiment["status"] = "completed"
            experiment["result_path"] = result_path
            experiment["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            local_store.write_json(f"experiments/{experiment_id}.json", experiment)
            self._condition.notify_all()

    def get_evaluation_overview(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return local_store.read_json(f"experiments/{experiment_id}/evaluation_overview.json")
        except FileNotFoundError:
            return None

    def get_evaluation_page(self, experiment_id: str, offset: int) -> dict[str, Any] | None:
        try:
            return local_store.read_json(f"experiments/{experiment_id}/evaluation_pages/{offset}.json")
        except FileNotFoundError:
            return None

    def get_evaluation_raw_result(self, experiment_id: str) -> dict[str, Any] | None:
        experiment = self.get_experiment(experiment_id)
        if experiment is None or not experiment.get("result_path"):
            return None
        try:
            return local_store.read_json(
                str(Path(str(experiment["result_path"])).relative_to(local_store.root))
            )
        except (FileNotFoundError, ValueError):
            return None

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._condition:
            job = local_store.read_json(f"experiments/{job_id}.json")
            if job.get("status") != "running":
                return
            experiment = local_store.read_json(f"experiments/{job['experiment_id']}.json")
            job["status"] = "failed"
            job["progress"] = 1.0
            _append_job_logs(job, [error_message])
            job["updated_at"] = _now()
            experiment["status"] = "failed"
            experiment["error_message"] = error_message
            experiment["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            local_store.write_json(f"experiments/{experiment['id']}.json", experiment)
            self._condition.notify_all()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            with self._lock:
                return local_store.read_json(f"experiments/{job_id}.json")
        except FileNotFoundError:
            return None

    def wait_job(self, job_id: str, after_log_count: int = 0, timeout: float = 30.0) -> dict[str, Any] | None:
        timeout = min(max(timeout, 1.0), 120.0)
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                try:
                    job = local_store.read_json(f"experiments/{job_id}.json")
                except FileNotFoundError:
                    return None
                if job["status"] != "running" or len(job.get("logs", [])) > after_log_count:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return job
                self._condition.wait(timeout=remaining)

    def append_job_log(
        self,
        job_id: str,
        message: str,
        progress: float | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        if not lines:
            return
        with self._condition:
            try:
                job = local_store.read_json(f"experiments/{job_id}.json")
            except FileNotFoundError:
                return
            _append_job_logs(job, lines)
            if progress is not None and job.get("status") == "running":
                job["progress"] = max(float(job.get("progress") or 0.0), min(float(progress), 0.99))
            if progress_current is not None and progress_total is not None and progress_total > 0:
                job["progress_current"] = max(0, min(progress_current, progress_total))
                job["progress_total"] = progress_total
            job["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            self._condition.notify_all()

    def set_job_progress(
        self,
        job_id: str,
        progress: float,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        """Persist monotonic in-flight progress without adding a synthetic log line."""
        with self._condition:
            try:
                job = local_store.read_json(f"experiments/{job_id}.json")
            except FileNotFoundError:
                return
            if job.get("status") != "running":
                return
            bounded = max(0.0, min(float(progress), 0.99))
            current = float(job.get("progress") or 0.0)
            next_current = None
            next_total = None
            if progress_current is not None and progress_total is not None and progress_total > 0:
                next_current = max(0, min(progress_current, progress_total))
                next_total = progress_total
            if bounded <= current and next_current == job.get("progress_current") and next_total == job.get("progress_total"):
                return
            job["progress"] = max(current, bounded)
            if next_current is not None and next_total is not None:
                job["progress_current"] = next_current
                job["progress_total"] = next_total
            job["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            self._condition.notify_all()


experiment_service = ExperimentService()


def _job_log_count(job: dict[str, Any]) -> int:
    return int(job.get("log_count", len(job.get("logs", []))))


def _append_job_logs(job: dict[str, Any], lines: list[str]) -> None:
    previous_count = _job_log_count(job)
    job["logs"] = [*job.get("logs", []), *lines][-_MAX_JOB_LOG_LINES:]
    job["log_count"] = previous_count + len(lines)
