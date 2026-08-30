from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from app.config import settings


class LocalStore:
    def __init__(self, root: Path = settings.data_dir) -> None:
        self.root = root

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(json.dumps(payload, indent=2))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            Path(temporary_path).unlink(missing_ok=True)
            raise
        return str(path)

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.root / relative_path
        raw = path.read_bytes()
        sanitized = raw.rstrip(b"\0")
        payload = json.loads(sanitized.decode("utf-8"))
        if sanitized != raw:
            # Older direct writes could leave NUL padding after a shorter JSON update.
            # Rewriting through the atomic path repairs the artifact for later requests.
            self.write_json(relative_path, payload)
        return payload

    def list_json(self, relative_dir: str) -> list[dict[str, Any]]:
        path = self.root / relative_dir
        if not path.exists():
            return []
        return [
            self.read_json(str(item.relative_to(self.root)))
            for item in sorted(path.glob("*.json"), reverse=True)
        ]


local_store = LocalStore()
