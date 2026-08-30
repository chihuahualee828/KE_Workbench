from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


class LocalStore:
    def __init__(self, root: Path = settings.data_dir) -> None:
        self.root = root

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.root / relative_path
        return json.loads(path.read_text(encoding="utf-8"))

    def list_json(self, relative_dir: str) -> list[dict[str, Any]]:
        path = self.root / relative_dir
        if not path.exists():
            return []
        return [
            json.loads(item.read_text(encoding="utf-8"))
            for item in sorted(path.glob("*.json"), reverse=True)
        ]


local_store = LocalStore()

