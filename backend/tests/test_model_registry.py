from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import model_registry as registry_module


def test_imported_model_persists_selected_mom2_statistics_source(monkeypatch) -> None:
    registry = registry_module.ModelRegistry()
    models = [
        {
            "id": "local-abc123",
            "source": "local",
            "base_model_id": "gpt2-xl",
        }
    ]
    writes: list[list[dict]] = []
    monkeypatch.setattr(registry, "list_local_models", lambda: models)
    monkeypatch.setattr(
        registry_module,
        "_write_local_models",
        lambda value: writes.append([dict(model) for model in value]),
    )

    selected = registry.set_mom2_statistics_source("local-abc123", "model")

    assert selected["mom2_stats_source"] == "model"
    assert models[0]["mom2_stats_source"] == "model"
    assert writes == [[{"id": "local-abc123", "source": "local", "base_model_id": "gpt2-xl", "mom2_stats_source": "model"}]]


@pytest.mark.parametrize("managed_local", [False, True])
def test_catalog_hides_missing_local_models_without_removing_registrations(
    monkeypatch, tmp_path: Path, managed_local: bool
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=data_dir))
    model_dir = tmp_path / "available-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"n_layer": 2}', encoding="utf-8")
    configless_dir = tmp_path / "configless-model"
    configless_dir.mkdir()
    available = {
        "id": "local-available",
        "local_path": str(model_dir),
        "source": "local",
        "managed_local": managed_local,
        "import_metadata": {},
        "n_layers": 2,
        "supports_ft": False,
        "supports_lora": False,
    }
    missing = {
        **available,
        "id": "local-other-server",
        "local_path": str(tmp_path / "other-server" / "gpt2-xl_local6"),
    }
    configless = {**available, "id": "local-no-config", "local_path": str(configless_dir)}
    registrations = [missing, available, configless]
    registry_module._write_local_models(registrations)
    registry = registry_module.ModelRegistry()

    assert registry.list_models() == [*registry_module.SUPPORTED_MODELS, available]
    assert registry.get(available["id"]) == available
    with pytest.raises(ValueError, match="Unsupported model"):
        registry.get(missing["id"])
    assert registry.list_local_models() == registrations
    registry_path = data_dir / "models" / "local_models.json"
    assert json.loads(registry_path.read_text(encoding="utf-8")) == registrations

    # Updating another registration must preserve the hidden server's record.
    registry.set_mom2_statistics_source(available["id"], "model")
    saved = json.loads(registry_path.read_text(encoding="utf-8"))
    assert next(model for model in saved if model["id"] == missing["id"]) == missing

    # A restored model folder becomes available on the next listing.
    restored_dir = Path(missing["local_path"])
    restored_dir.mkdir(parents=True)
    (restored_dir / "config.json").write_text('{"n_layer": 2}', encoding="utf-8")
    assert missing["id"] in {model["id"] for model in registry.list_models()}


@pytest.mark.parametrize("local_path", [None, "", " ", 123, "invalid\0path"])
def test_catalog_skips_invalid_local_paths(monkeypatch, local_path) -> None:
    monkeypatch.setattr(
        registry_module, "_read_local_models",
        lambda: [{"id": "local-invalid", "local_path": local_path}],
    )
    monkeypatch.setattr(registry_module, "_migrate_managed_local_models", lambda _: False)

    assert registry_module.ModelRegistry().list_models() == registry_module.SUPPORTED_MODELS


def test_catalog_skips_inaccessible_local_folders(monkeypatch) -> None:
    registry = registry_module.ModelRegistry()
    monkeypatch.setattr(
        registry, "list_local_models",
        lambda: [{"id": "local-denied", "local_path": "/unreadable/model"}],
    )

    def denied(_path):
        raise PermissionError("Folder cannot be accessed")

    monkeypatch.setattr(Path, "is_dir", denied)

    assert registry.list_models() == registry_module.SUPPORTED_MODELS


@pytest.mark.parametrize(
    "saved_path",
    [
        "/old/wsl/backend/data/models/imported_models/local-moved",
        r"D:\old\backend\data\models\imported_models\local-moved",
        "models/imported_models/local-moved",
    ],
)
def test_managed_import_resolves_after_data_directory_moves(
    monkeypatch, tmp_path: Path, saved_path: str
) -> None:
    root = tmp_path / "configured-storage"
    relative_path = "models/imported_models/local-moved"
    model_dir = root / relative_path
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"n_layer": 2}', encoding="utf-8")
    registry_path = root / "models" / "local_models.json"
    registry_path.write_text(json.dumps([{
        "id": "local-moved",
        "source": "local",
        "managed_local": True,
        "local_path": saved_path,
        "hf_id": saved_path,
        "base_model_id": "gpt2-xl",
        "import_metadata": {},
    }]), encoding="utf-8")
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=root))
    monkeypatch.chdir(tmp_path)
    registry = registry_module.ModelRegistry()

    current = registry.get("local-moved")
    assert current["local_path"] == current["hf_id"] == str(model_dir)
    assert current["n_layers"] == 2
    assert current["supports_ft"] is True
    assert current["supports_lora"] is True
    assert current["easyedit"]["ft_hparams"] == "hparams/FT/gpt2-xl.yaml"
    assert current["easyedit"]["lora_hparams"] == "hparams/LoRA/gpt2-xl.yaml"
    saved = json.loads(registry_path.read_text(encoding="utf-8"))
    assert saved[0]["local_path"] == saved[0]["hf_id"] == relative_path

    moved_root = tmp_path / "another-server-storage"
    root.rename(moved_root)
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=moved_root))

    current = registry.get("local-moved")
    assert current["local_path"] == current["hf_id"] == str(moved_root / relative_path)
    assert current["supports_ft"] is True
    assert current["supports_lora"] is True
    assert (Path(current["hf_id"]) / "config.json").is_file()
    registry.set_mom2_statistics_source("local-moved", "model")
    saved = json.loads((moved_root / "models" / "local_models.json").read_text())
    assert saved[0]["local_path"] == saved[0]["hf_id"] == relative_path


def test_new_managed_registration_persists_relative_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=tmp_path))
    model_dir = tmp_path / "models" / "imported_models" / "local-new"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"n_layer": 2}', encoding="utf-8")
    registered = registry_module.ModelRegistry().add_local_model({
        "id": "local-new",
        "name": "My model",
        "local_path": str(model_dir),
        "model_type": "gpt2",
    }, managed_local=True)

    assert registered["local_path"] == registered["hf_id"] == str(model_dir)
    saved = json.loads((tmp_path / "models" / "local_models.json").read_text())
    assert saved[0]["local_path"] == saved[0]["hf_id"] == "models/imported_models/local-new"


def test_manual_registration_keeps_its_external_path(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "data"
    model_dir = tmp_path / "external" / "models" / "imported_models" / "local-manual"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{}', encoding="utf-8")
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=root))
    record = {
        "id": "local-manual",
        "source": "local",
        "managed_local": False,
        "local_path": str(model_dir),
        "hf_id": str(model_dir),
        "supports_ft": False,
        "supports_lora": False,
    }
    registry_module._write_local_models([record])

    assert registry_module.ModelRegistry().get("local-manual") == record
    saved = json.loads((root / "models" / "local_models.json").read_text())
    assert saved == [record]


def test_missing_managed_import_does_not_use_another_models_folder(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry_module, "settings", SimpleNamespace(data_dir=tmp_path))
    other_dir = tmp_path / "models" / "imported_models" / "local-other"
    other_dir.mkdir(parents=True)
    (other_dir / "config.json").write_text('{}', encoding="utf-8")
    registry_module._write_local_models([{
        "id": "local-missing",
        "managed_local": True,
        "local_path": "/old/data/models/imported_models/local-missing",
    }])

    assert registry_module.ModelRegistry().list_models() == registry_module.SUPPORTED_MODELS


def test_relocated_managed_import_deletes_only_current_copy(monkeypatch, tmp_path: Path) -> None:
    from app.services import model_cache_service as cache_module

    root = tmp_path / "data"
    model_dir = root / "models" / "imported_models" / "local-deleted"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    other_dir = model_dir.with_name("local-kept")
    other_dir.mkdir()
    (other_dir / "config.json").write_text('{}', encoding="utf-8")
    original_dir = tmp_path / "old-server" / "models" / "imported_models" / "local-deleted"
    original_dir.mkdir(parents=True)
    (original_dir / "config.json").write_text('{}', encoding="utf-8")
    registry_path = root / "models" / "local_models.json"
    registry_path.write_text(json.dumps([{
        "id": "local-deleted",
        "source": "local",
        "managed_local": True,
        "local_path": str(original_dir),
        "hf_id": str(original_dir),
        "import_metadata": {},
    }]), encoding="utf-8")
    settings = SimpleNamespace(data_dir=root)
    registry = registry_module.ModelRegistry()
    monkeypatch.setattr(registry_module, "settings", settings)
    monkeypatch.setattr(cache_module, "settings", settings)
    monkeypatch.setattr(cache_module, "model_registry", registry)

    service = cache_module.model_cache_service
    cache_status = service.cache_status("local-deleted")
    assert cache_status["cached"] is True
    assert cache_status["cache_path"] == str(model_dir)
    deleted = service.delete_local_model("local-deleted")

    assert deleted["cache_path"] == str(model_dir)
    assert deleted["deleted_cache_files"] is True
    assert not model_dir.exists()
    assert other_dir.is_dir()
    assert original_dir.is_dir()
    assert json.loads(registry_path.read_text(encoding="utf-8")) == []
