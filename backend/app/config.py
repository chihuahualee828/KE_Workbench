from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_backend_root = Path(__file__).resolve().parents[1]
_repo_root = _backend_root.parent
_load_env_file(_repo_root / ".env")
_load_env_file(_backend_root / ".env")

# An empty optional override must not become an empty Hub cache path.
if os.environ.get("HF_HUB_CACHE") == "":
    os.environ.pop("HF_HUB_CACHE")


def _resolve_existing_path(value: str, marker: str | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path).resolve(),
        (_repo_root / path).resolve(),
        (_backend_root / path).resolve(),
    ]
    if marker:
        for candidate in candidates:
            if (candidate / marker).exists():
                return candidate
    return candidates[0]


def _resolve_data_dir() -> Path:
    """Resolve storage once so imports do not depend on a later cwd change."""
    configured = os.getenv("DATA_DIR")
    if not configured:
        return (_backend_root / "data").resolve()
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (_backend_root / path).resolve()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_hf_hub_cache_dir() -> Path:
    explicit = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
    )
    if explicit:
        path = Path(explicit).expanduser()
    else:
        return default_hf_hub_cache_dir()
    return path.resolve() if path.is_absolute() else (_backend_root / path).resolve()


def default_hf_hub_cache_dir() -> Path:
    cache_home = Path(os.getenv("XDG_CACHE_HOME") or "~/.cache").expanduser()
    hf_home = Path(os.getenv("HF_HOME") or cache_home / "huggingface").expanduser()
    path = hf_home / "hub"
    return path.resolve() if path.is_absolute() else (_backend_root / path).resolve()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/workbench.db")
    data_dir: Path = _resolve_data_dir()
    hf_hub_cache_dir: Path = _resolve_hf_hub_cache_dir()
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    easyedit_root: Path = _resolve_existing_path(os.getenv("EASYEDIT_ROOT", "./EasyEdit"), marker="easyeditor")
    editing_runtime: str = os.getenv("EDITING_RUNTIME", "easyedit")
    edit_log_to_terminal: bool = _env_flag("EDIT_LOG_TO_TERMINAL")


settings = Settings()


def ensure_data_dirs() -> None:
    for child in ("models", "patches", "experiments", "uploads"):
        (settings.data_dir / child).mkdir(parents=True, exist_ok=True)
