from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.storage.local_store import local_store


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
        return [
            item for item in local_store.list_json("experiments")
            if item.get("id", "").startswith("exp_")
        ]

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return local_store.read_json(f"experiments/{experiment_id}.json")
        except FileNotFoundError:
            return None

    def create_job(self, experiment_id: str) -> dict[str, Any]:
        job_id = f"job_{uuid4().hex[:10]}"
        job = {
            "id": job_id,
            "experiment_id": experiment_id,
            "status": "running",
            "progress": 0.0,
            "logs": ["Job started"],
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._condition:
            local_store.write_json(f"experiments/{job_id}.json", job)
            self._condition.notify_all()
        return job

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        with self._condition:
            job = local_store.read_json(f"experiments/{job_id}.json")
            experiment = local_store.read_json(f"experiments/{job['experiment_id']}.json")
            result_path = local_store.write_json(
                f"experiments/{job['experiment_id']}/result.json",
                result,
            )
            job["status"] = "completed"
            job["progress"] = 1.0
            job["logs"].append("Job completed")
            job["logs"] = job["logs"][-500:]
            job["updated_at"] = _now()
            experiment["status"] = "completed"
            experiment["result_path"] = result_path
            experiment["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            local_store.write_json(f"experiments/{experiment['id']}.json", experiment)
            self._condition.notify_all()

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._condition:
            job = local_store.read_json(f"experiments/{job_id}.json")
            experiment = local_store.read_json(f"experiments/{job['experiment_id']}.json")
            job["status"] = "failed"
            job["progress"] = 1.0
            job["logs"].append(error_message)
            job["logs"] = job["logs"][-500:]
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

    def append_job_log(self, job_id: str, message: str) -> None:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        if not lines:
            return
        with self._condition:
            try:
                job = local_store.read_json(f"experiments/{job_id}.json")
            except FileNotFoundError:
                return
            job["logs"].extend(lines)
            job["logs"] = job["logs"][-500:]
            job["updated_at"] = _now()
            local_store.write_json(f"experiments/{job_id}.json", job)
            self._condition.notify_all()


experiment_service = ExperimentService()
