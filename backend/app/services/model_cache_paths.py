from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any

from app.config import default_hf_hub_cache_dir, settings
from app.storage.local_store import LocalStore, local_store


class ModelCachePaths:
    relative_path = "settings/model_cache.json"

    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store if store is not None else local_store
        self._lock = threading.RLock()

    def _state(self) -> dict[str, Any]:
        try:
            return self.store.read_json(self.relative_path)
        except FileNotFoundError:
            return {}

    def configuration(self) -> dict[str, Any]:
        with self._lock:
            state = self._state()
            override = state.get("download_dir")
            environment = settings.hf_hub_cache_dir
            destination = Path(override) if override else environment
            candidates = [
                destination,
                environment,
                default_hf_hub_cache_dir(),
                Path.home() / ".cache" / "huggingface" / "hub",
                *[Path(value) for value in state.get("cache_dirs", [])],
            ]
            roots = list(dict.fromkeys(str(path.expanduser().resolve()) for path in candidates))
            return {
                "download_dir": str(destination),
                "environment_dir": str(environment),
                "search_dirs": roots,
                "source": "settings" if override else "environment",
            }

    def roots(self) -> list[Path]:
        return [Path(value) for value in self.configuration()["search_dirs"]]

    def download_dir(self) -> Path:
        return Path(self.configuration()["download_dir"])

    def remember_environment(self) -> None:
        with self._lock:
            state = self._state()
            directories = state.get("cache_dirs", [])
            current = str(settings.hf_hub_cache_dir)
            if current not in directories:
                state["cache_dirs"] = [*directories, current]
                self.store.write_json(self.relative_path, state)

    def set_directory(self, value: str) -> dict[str, Any]:
        value = value.strip()
        path = Path(value).expanduser() if value else settings.hf_hub_cache_dir
        if not path.is_absolute():
            raise ValueError("Enter an absolute directory path on the backend machine.")
        path = path.resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError as exc:
            raise ValueError(f"Cache directory is not writable: {path}. {exc.strerror}") from exc
        with self._lock:
            previous = self.configuration()
            self.store.write_json(self.relative_path, {
                "download_dir": str(path) if value else None,
                "cache_dirs": previous["search_dirs"],
            })
            return self.configuration()


model_cache_paths = ModelCachePaths()
