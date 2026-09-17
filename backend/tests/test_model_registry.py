from __future__ import annotations

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
