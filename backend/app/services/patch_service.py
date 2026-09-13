from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
from typing import Any
from uuid import uuid4

from app.config import settings
from app.storage.local_store import local_store


class PatchService:
    lineage_filename = "kew_lineage.json"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_evaluation_fact_cache: dict[str, tuple[tuple[str, ...], list[dict[str, Any]]]] = {}

    def new_patch_id(self) -> str:
        return f"patch_{uuid4().hex[:10]}"

    def create_patch(
        self,
        model_id: str,
        method: str,
        edits: list[dict[str, Any]],
        modified_layers: list[int],
        metrics: dict[str, Any] | None = None,
        delta_paths: list[str] | None = None,
        runtime: str = "native",
        extra_metadata: dict[str, Any] | None = None,
        patch_id: str | None = None,
    ) -> dict[str, Any]:
        patch_id = patch_id or self.new_patch_id()
        patch_dir = f"patches/{patch_id}"
        self.patch_dir(patch_id)
        metadata = {
            "patch_id": patch_id,
            "base_model": model_id,
            "method": method,
            "edits": edits,
            "modified_layers": modified_layers,
            "delta_paths": delta_paths or [f"data/{patch_dir}/metadata_delta.json"],
            "metrics": metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if delta_paths is None:
            local_store.write_json(
                f"{patch_dir}/metadata_delta.json",
                {"note": "No tensor deltas were supplied for this patch."},
            )
        local_store.write_json(self._metadata_relative_path(patch_id), metadata)
        return metadata

    def list_patches(self) -> list[dict[str, Any]]:
        active_by_model = self.active_patches_by_model()
        patches_root = local_store.root / "patches"
        if patches_root.exists():
            for legacy_metadata_path in patches_root.glob("patch_*.json"):
                self._migrate_flat_metadata(legacy_metadata_path.stem)
        patches = [
            local_store.read_json(str(metadata_path.relative_to(local_store.root)))
            for metadata_path in sorted(patches_root.glob("*/patch.json"), reverse=True)
        ] if patches_root.exists() else []
        for patch in patches:
            active_stack = active_by_model.get(patch.get("base_model"), [])
            patch["active"] = patch["patch_id"] in active_stack
            patch["active_index"] = active_stack.index(patch["patch_id"]) if patch["active"] else None
        patches.sort(key=lambda patch: str(patch.get("created_at") or ""), reverse=True)
        return patches

    def get_patch(self, patch_id: str) -> dict[str, Any]:
        self._migrate_flat_metadata(patch_id)
        return local_store.read_json(self._metadata_relative_path(patch_id))

    def active_patches_for_model(self, model_id: str) -> list[str]:
        return self.active_patches_by_model().get(model_id, [])

    def active_evaluation_facts(self, model_id: str) -> list[dict[str, Any]]:
        """Return de-duplicated edit facts for the currently active patch stack.

        The final patch is shown first, matching the evaluation page's previous
        ordering.  A fact's key is stable for the lifetime of its patch metadata
        and identifies the exact patch/edit pair, rather than just its prompt.
        """
        active_patch_ids = self.active_patches_for_model(model_id)
        signature = tuple(active_patch_ids)
        cached = self._active_evaluation_fact_cache.get(model_id)
        optional_prompt_fields = {
            "rephrase_prompt",
            "locality_prompt",
            "locality_ground_truth",
            "portability_prompt",
            "portability_ground_truth",
        }
        if cached and cached[0] == signature and all(optional_prompt_fields.issubset(fact) for fact in cached[1]):
            return cached[1]

        facts: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        for patch_id in reversed(active_patch_ids):
            patch = self.get_patch(patch_id)
            for index, raw_edit in enumerate(patch.get("edits", [])):
                if not isinstance(raw_edit, dict):
                    continue
                subject = str(raw_edit.get("subject") or "").strip()
                relation_prompt = str(raw_edit.get("relation_prompt") or "").strip()
                new_object = str(raw_edit.get("new_object") or "").strip()
                if not subject or not relation_prompt or not new_object:
                    continue
                prompt = relation_prompt.replace("{}", subject) if "{}" in relation_prompt else f"{subject} {relation_prompt}".strip()
                identity = (prompt, new_object)
                if identity in identities:
                    continue
                identities.add(identity)
                source_metadata = raw_edit.get("source_metadata")
                if not isinstance(source_metadata, dict):
                    source_metadata = {}

                def optional_prompt(field: str, *, allow_list: bool = False) -> str:
                    """Read an imported prompt whether it was stored directly or as metadata."""
                    values: list[Any] = [raw_edit.get(field), source_metadata.get(field)]
                    if allow_list:
                        values.extend([raw_edit.get(f"{field}s"), source_metadata.get(f"{field}s")])
                    for value in values:
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                        if allow_list and isinstance(value, list):
                            first = next((item.strip() for item in value if isinstance(item, str) and item.strip()), "")
                            if first:
                                return first
                    return ""

                facts.append(
                    {
                        "key": f"{patch_id}:{index}",
                        "subject": subject,
                        "relation_prompt": relation_prompt,
                        "prompt": prompt,
                        "new_object": new_object,
                        "ground_truth": str(raw_edit.get("ground_truth") or "").strip(),
                        "rephrase_prompt": optional_prompt("rephrase_prompt", allow_list=True),
                        "locality_prompt": optional_prompt("locality_prompt"),
                        "locality_ground_truth": optional_prompt("locality_ground_truth"),
                        "portability_prompt": optional_prompt("portability_prompt"),
                        "portability_ground_truth": optional_prompt("portability_ground_truth"),
                    }
                )

        self._active_evaluation_fact_cache[model_id] = (signature, facts)
        return facts

    def active_patches_by_model(self) -> dict[str, list[str]]:
        try:
            payload = local_store.read_json("model_state/active_patches.json")
        except FileNotFoundError:
            return {}
        return {
            str(model_id): [str(patch_id) for patch_id in patch_ids]
            for model_id, patch_ids in payload.get("models", {}).items()
            if isinstance(patch_ids, list)
        }

    def clear_active_patches(self) -> None:
        with self._lock:
            self._write_active_patches({})

    def activate_patch(self, patch_id: str, append: bool = True) -> dict[str, Any]:
        patch = self.get_patch(patch_id)
        model_id = patch["base_model"]
        with self._lock:
            active_by_model = self.active_patches_by_model()
            active_stack = active_by_model.get(model_id, []) if append else []
            if patch_id not in active_stack:
                active_stack.append(patch_id)
            active_by_model[model_id] = active_stack
            self._write_active_patches(active_by_model)
            return {"model_id": model_id, "active_patches": active_stack}

    def load_saved_patch(self, patch_id: str) -> dict[str, Any]:
        patch = self.get_patch(patch_id)
        if _is_baked_lineage(patch):
            raise ValueError(f"Patch {patch_id} is already baked into its imported model weights.")
        if not _has_tensor_deltas(patch):
            raise ValueError(f"Patch {patch_id} has no saved tensor deltas and cannot be loaded.")

        model_id = patch["base_model"]
        parent_patch_ids = patch.get("parent_patch_ids", []) if patch.get("sequential_edit") else []
        if not isinstance(parent_patch_ids, list):
            parent_patch_ids = []
        target_stack = [str(parent_patch_id) for parent_patch_id in parent_patch_ids] + [patch_id]
        target_stack = list(dict.fromkeys(target_stack))

        with self._lock:
            for candidate_id in target_stack:
                candidate = self.get_patch(candidate_id)
                if candidate.get("base_model") != model_id:
                    raise ValueError(f"Patch {candidate_id} belongs to {candidate.get('base_model')}, not {model_id}.")
                if not _has_tensor_deltas(candidate):
                    raise ValueError(f"Patch {candidate_id} has no saved tensor deltas and cannot be loaded.")
                missing_delta_paths = [
                    path_value
                    for path_value in candidate.get("delta_paths", [])
                    if str(path_value).endswith(".safetensors") and not self.resolve_delta_path(path_value).exists()
                ]
                if missing_delta_paths:
                    raise ValueError(f"Patch {candidate_id} is missing its saved tensor delta file.")
            active_by_model = self.active_patches_by_model()
            active_by_model[model_id] = target_stack
            self._write_active_patches(active_by_model)
            return {"model_id": model_id, "active_patches": target_stack}

    def discard_active_patch(self, patch_id: str) -> dict[str, Any]:
        patch = self.get_patch(patch_id)
        if _is_baked_lineage(patch):
            raise ValueError(f"Patch {patch_id} is baked into the model weights and cannot be discarded.")
        model_id = patch["base_model"]
        with self._lock:
            active_by_model = self.active_patches_by_model()
            current_stack = active_by_model.get(model_id, [])
            if patch_id not in current_stack:
                raise ValueError(f"Patch {patch_id} is not active for {model_id}.")
            # Later sequential patches were calculated on this patch's model state,
            # so discarding a point also discards its dependent suffix.
            active_stack = current_stack[: current_stack.index(patch_id)]
            if active_stack:
                active_by_model[model_id] = active_stack
            else:
                active_by_model.pop(model_id, None)
            self._write_active_patches(active_by_model)
            return {"model_id": model_id, "active_patches": active_stack}

    def revert_to_patch(self, patch_id: str) -> dict[str, Any]:
        patch = self.get_patch(patch_id)
        model_id = patch["base_model"]
        with self._lock:
            active_by_model = self.active_patches_by_model()
            active_stack = active_by_model.get(model_id, [])
            if patch_id not in active_stack:
                raise ValueError(f"Patch {patch_id} is not active for {model_id}.")
            removed_patch_ids = active_stack[active_stack.index(patch_id) + 1 :]
            baked_removed = [
                removed_patch_id
                for removed_patch_id in removed_patch_ids
                if _is_baked_lineage(self.get_patch(removed_patch_id))
            ]
            if baked_removed:
                raise ValueError(
                    "Cannot revert before imported baked lineage because those edits are already in the model weights."
                )
            active_by_model[model_id] = active_stack[: active_stack.index(patch_id) + 1]
            self._write_active_patches(active_by_model)
            return {"model_id": model_id, "active_patches": active_by_model[model_id]}

    def deactivate_all_patches(self) -> dict[str, Any]:
        with self._lock:
            active_by_model = self.active_patches_by_model()
            self._write_active_patches({})
            return {"active_patches_by_model": active_by_model, "model_ids": list(active_by_model)}

    def deactivate_runtime_patches_for_model(self, model_id: str) -> dict[str, list[str]]:
        """Remove reloadable patch deltas when their model leaves runtime memory."""
        with self._lock:
            active_by_model = self.active_patches_by_model()
            current_stack = active_by_model.get(model_id, [])
            retained_patch_ids: list[str] = []
            discarded_patch_ids: list[str] = []
            for patch_id in current_stack:
                try:
                    patch = self.get_patch(patch_id)
                except FileNotFoundError:
                    discarded_patch_ids.append(patch_id)
                    continue
                if _is_baked_lineage(patch):
                    retained_patch_ids.append(patch_id)
                else:
                    discarded_patch_ids.append(patch_id)
            if retained_patch_ids:
                active_by_model[model_id] = retained_patch_ids
            else:
                active_by_model.pop(model_id, None)
            self._write_active_patches(active_by_model)
            return {
                "discarded_patch_ids": discarded_patch_ids,
                "retained_baked_patch_ids": retained_patch_ids,
            }

    def restore_active_patches(self, active_by_model: dict[str, list[str]]) -> None:
        with self._lock:
            self._write_active_patches(active_by_model)

    def delete_all_patch_artifacts(self) -> dict[str, int]:
        patches_root = local_store.root / "patches"
        with self._lock:
            patch_count = len(list(patches_root.glob("*/patch.json"))) + len(list(patches_root.glob("patch_*.json"))) if patches_root.exists() else 0
            if patches_root.exists():
                shutil.rmtree(patches_root)
            patches_root.mkdir(parents=True, exist_ok=True)
        return {"deleted_patch_count": patch_count}

    def delete_patch_artifact(self, patch_id: str) -> dict[str, str]:
        if not patch_id.startswith("patch_") or Path(patch_id).name != patch_id:
            raise FileNotFoundError(patch_id)
        patch = self.get_patch(patch_id)
        model_id = str(patch["base_model"])
        with self._lock:
            if patch_id in self.active_patches_by_model().get(model_id, []):
                raise ValueError(f"Discard or revert patch {patch_id} before deleting its saved artifact.")

            patch_dir = local_store.root / "patches" / patch_id
            if not patch_dir.exists():
                raise FileNotFoundError(patch_id)
            shutil.rmtree(patch_dir)
        return {"patch_id": patch_id}

    def import_model_lineage(self, model: dict[str, Any]) -> dict[str, Any]:
        local_path = Path(model.get("local_path") or model.get("hf_id") or "")
        lineage_path = local_path / self.lineage_filename
        if not lineage_path.exists():
            return {"imported": 0, "active_patches": []}

        payload = json.loads(lineage_path.read_text(encoding="utf-8"))
        entries = payload.get("patch_history") or payload.get("patches") or []
        if not isinstance(entries, list):
            return {"imported": 0, "active_patches": []}

        patch_id_map: dict[str, str] = {}
        imported_patch_ids: list[str] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            original_patch_id = str(entry.get("patch_id") or f"lineage_{index + 1}")
            patch_id = self._lineage_patch_id(model["id"], original_patch_id)
            patch_id_map[original_patch_id] = patch_id
            if entry.get("workbench_patch_id"):
                patch_id_map[str(entry["workbench_patch_id"])] = patch_id
            imported_patch_ids.append(patch_id)
            if self._patch_exists(patch_id):
                continue
            metadata = {
                "patch_id": patch_id,
                "original_patch_id": original_patch_id,
                "base_model": model["id"],
                "method": str(entry.get("method") or "unknown"),
                "edits": entry.get("edits") if isinstance(entry.get("edits"), list) else _edits_from_lineage_entry(entry),
                "modified_layers": entry.get("modified_layers") or entry.get("layers") or [],
                "delta_paths": [],
                "metrics": entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {},
                "created_at": entry.get("created_at") or datetime.now(timezone.utc).isoformat(),
                "runtime": entry.get("runtime") or "imported_lineage",
                "sequential_edit": bool(entry.get("sequential_edit", True)),
                "parent_patch_ids": entry.get("parent_patch_ids") if isinstance(entry.get("parent_patch_ids"), list) else [],
                "lineage_only": True,
                "baked_into_model": True,
                "imported_from_model_path": str(local_path),
            }
            if isinstance(entry.get("mom2_adjustment"), bool):
                metadata["mom2_adjustment"] = entry["mom2_adjustment"]
            local_store.write_json(self._metadata_relative_path(patch_id), metadata)

        active_source = payload.get("active_patches")
        if isinstance(active_source, list) and active_source:
            active_patch_ids = [
                patch_id_map.get(str(patch_id), self._lineage_patch_id(model["id"], str(patch_id)))
                for patch_id in active_source
            ]
            active_patch_ids = [patch_id for patch_id in active_patch_ids if patch_id in imported_patch_ids]
        else:
            active_patch_ids = imported_patch_ids

        with self._lock:
            active_by_model = self.active_patches_by_model()
            if active_patch_ids:
                active_by_model[model["id"]] = active_patch_ids
            else:
                active_by_model.pop(model["id"], None)
            self._write_active_patches(active_by_model)

        return {"imported": len(imported_patch_ids), "active_patches": active_patch_ids}

    def write_model_lineage(self, model_id: str) -> None:
        local_model = self._local_model_for_id(model_id)
        if not local_model:
            return
        local_path = Path(local_model.get("local_path") or local_model.get("hf_id") or "")
        if not local_path.exists():
            return
        self.write_lineage_file(model_id, local_path, local_model.get("name") or model_id)

    def write_lineage_file(self, model_id: str, output_path: Path, model_name: str | None = None) -> Path:
        active_patch_ids = self.active_patches_for_model(model_id)
        patch_history = []
        for patch_id in active_patch_ids:
            try:
                patch = self.get_patch(patch_id)
            except FileNotFoundError:
                continue
            patch_history.append(_lineage_entry_from_patch(patch, baked_into_model=True))

        payload = {
            "schema_version": 1,
            "workbench_model_id": model_id,
            "model_name": model_name or model_id,
            "model_path": str(output_path),
            "active_patches": active_patch_ids,
            "patch_count": len(patch_history),
            "patch_history": patch_history,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        lineage_path = output_path / self.lineage_filename
        lineage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return lineage_path

    def export_patch_bundle(self, patch_id: str, output_path: str) -> dict[str, Any]:
        patch = self.get_patch(patch_id)
        export_path = Path(output_path).expanduser().resolve()
        export_path.mkdir(parents=True, exist_ok=True)

        delta_files = []
        deltas_dir = export_path / "deltas"
        for path_value in patch.get("delta_paths", []):
            source = self.resolve_delta_path(path_value)
            if not source.exists():
                continue
            deltas_dir.mkdir(parents=True, exist_ok=True)
            destination = deltas_dir / source.name
            shutil.copy2(source, destination)
            delta_files.append(str(destination))

        metrics_source = settings.data_dir / "patches" / patch_id / "easyedit_metrics.json"
        metrics_path = None
        if metrics_source.exists():
            metrics_path = export_path / "easyedit_metrics.json"
            shutil.copy2(metrics_source, metrics_path)

        manifest = {
            "schema_version": 1,
            "bundle_type": "kew_patch",
            "patch": patch,
            "delta_files": delta_files,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = export_path / "kew_patch.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "patch_id": patch_id,
            "output_path": str(export_path),
            "manifest_path": str(manifest_path),
            "delta_files": delta_files,
        }

    def patch_dir(self, patch_id: str) -> Path:
        path = local_store.root / "patches" / patch_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _metadata_relative_path(self, patch_id: str) -> str:
        return f"patches/{patch_id}/patch.json"

    def _migrate_flat_metadata(self, patch_id: str) -> None:
        patches_root = local_store.root / "patches"
        legacy_metadata_path = patches_root / f"{patch_id}.json"
        canonical_metadata_path = patches_root / patch_id / "patch.json"
        if canonical_metadata_path.exists():
            if legacy_metadata_path.exists():
                legacy_metadata_path.unlink()
            return
        if not legacy_metadata_path.exists():
            return
        canonical_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_metadata_path.replace(canonical_metadata_path)
        except FileNotFoundError:
            if not canonical_metadata_path.exists():
                raise

    def resolve_delta_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == "data":
            return settings.data_dir.parent / path
        return settings.data_dir / path

    @contextmanager
    def applied_to_model(
        self,
        model: Any,
        patch_id: str | None,
        model_id: str | None = None,
        use_active_patches: bool = True,
    ):
        if model_id:
            if use_active_patches:
                self.sync_active_patches_to_model(model, model_id)
            else:
                self.sync_patch_stack_to_model(model, model_id, [])

        temporary_patch = None
        if patch_id and patch_id not in getattr(model, "_kew_active_patch_ids", ()):
            temporary_patch = self.get_patch(patch_id)
            if model_id and temporary_patch.get("base_model") != model_id:
                raise ValueError(
                    f"Patch {patch_id} was created for {temporary_patch.get('base_model')}, not {model_id}."
                )
            self._apply_patch_to_model(model, temporary_patch)

        try:
            yield
        finally:
            if temporary_patch is not None:
                self._revert_patch_from_model(model, temporary_patch)
            if model_id and not use_active_patches:
                self.sync_active_patches_to_model(model, model_id)

    def sync_active_patches_to_model(self, model: Any, model_id: str) -> None:
        self.sync_patch_stack_to_model(model, model_id, self.active_patches_for_model(model_id))

    def sync_patch_stack_to_model(self, model: Any, model_id: str, patch_ids: list[str]) -> None:
        desired = tuple(patch_ids)
        current = tuple(getattr(model, "_kew_active_patch_ids", ()))
        if current == desired:
            return

        current_patches = [self.get_patch(patch_id) for patch_id in current]
        for patch in reversed(current_patches):
            self._revert_patch_from_model(model, patch)

        for patch_id in desired:
            patch = self.get_patch(patch_id)
            if patch.get("base_model") != model_id:
                raise ValueError(
                    f"Patch {patch_id} was created for {patch.get('base_model')}, not {model_id}."
                )
            self._apply_patch_to_model(model, patch)
        setattr(model, "_kew_active_patch_ids", desired)

    def _write_active_patches(self, active_by_model: dict[str, list[str]]) -> None:
        self._active_evaluation_fact_cache.clear()
        local_store.write_json(
            "model_state/active_patches.json",
            {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "models": active_by_model,
            },
        )

    def _patch_exists(self, patch_id: str) -> bool:
        try:
            self.get_patch(patch_id)
        except FileNotFoundError:
            return False
        return True

    def _lineage_patch_id(self, model_id: str, patch_id: str) -> str:
        safe_model_id = _safe_id(model_id)
        safe_patch_id = _safe_id(patch_id)
        if safe_patch_id.startswith(f"lineage_{safe_model_id}_"):
            return safe_patch_id
        return f"lineage_{safe_model_id}_{safe_patch_id}"

    def _local_model_for_id(self, model_id: str) -> dict[str, Any] | None:
        path = settings.data_dir / "models" / "local_models.json"
        if not path.exists():
            return None
        try:
            models = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        for model in models:
            if model.get("id") == model_id and model.get("source") == "local":
                return model
        return None

    def _apply_patch_to_model(self, model: Any, patch: dict[str, Any]) -> None:
        if patch.get("sparse_delta_format") == "kn_v1":
            self._apply_kn_sparse_patch(model, patch, direction=1)
            return
        for parameter, delta in self._patch_deltas(model, patch):
            parameter.data.add_(delta)

    def _revert_patch_from_model(self, model: Any, patch: dict[str, Any]) -> None:
        if patch.get("sparse_delta_format") == "kn_v1":
            self._apply_kn_sparse_patch(model, patch, direction=-1)
            return
        for parameter, delta in self._patch_deltas(model, patch):
            parameter.data.sub_(delta)

    def _apply_kn_sparse_patch(self, model: Any, patch: dict[str, Any], direction: int) -> None:
        """Apply KN's sparse MLP-output vectors without materializing full-layer deltas."""
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install torch and safetensors to apply saved KN patches.") from exc

        specs = patch.get("sparse_delta_specs")
        if not isinstance(specs, list):
            raise ValueError(f"KN patch {patch.get('patch_id')} is missing sparse delta metadata.")
        parameter_map = dict(model.named_parameters())
        for path_value in patch.get("delta_paths", []):
            path = self.resolve_delta_path(path_value)
            if path.suffix != ".safetensors":
                continue
            tensors = load_file(str(path))
            for spec in specs:
                if not isinstance(spec, dict):
                    continue
                entry = spec.get("entry")
                parameter = parameter_map.get(spec.get("parameter"))
                orientation = spec.get("orientation")
                if not isinstance(entry, str) or parameter is None or orientation not in {"row", "column"}:
                    continue
                indices = tensors.get(f"{entry}.indices")
                values = tensors.get(f"{entry}.values")
                if indices is None or values is None:
                    continue
                indices = indices.to(device=parameter.device, dtype=torch.long)
                values = values.to(device=parameter.device, dtype=parameter.dtype)
                if direction < 0:
                    values = -values
                if orientation == "row":
                    parameter.data.index_add_(0, indices, values)
                else:
                    parameter.data.index_add_(1, indices, values.transpose(0, 1))

    def _patch_deltas(self, model: Any, patch: dict[str, Any]):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install safetensors to apply saved EasyEdit patches.") from exc

        parameter_map = dict(model.named_parameters())
        for path_value in patch.get("delta_paths", []):
            path = self.resolve_delta_path(path_value)
            if path.suffix != ".safetensors":
                continue
            tensors = load_file(str(path))
            for name, delta in tensors.items():
                parameter = parameter_map.get(name)
                if parameter is None:
                    continue
                yield parameter, delta.to(device=parameter.device, dtype=parameter.dtype)


patch_service = PatchService()


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _edits_from_lineage_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if all(key in entry for key in ("subject", "relation_prompt", "new_object")):
        edit = {
            "subject": entry["subject"],
            "relation_prompt": entry["relation_prompt"],
            "new_object": entry["new_object"],
        }
        if entry.get("ground_truth") is not None:
            edit["ground_truth"] = entry["ground_truth"]
        return [edit]
    return []


def _lineage_entry_from_patch(patch: dict[str, Any], baked_into_model: bool = False) -> dict[str, Any]:
    return {
        "patch_id": patch.get("original_patch_id") or patch["patch_id"],
        "workbench_patch_id": patch["patch_id"],
        "method": patch.get("method"),
        "edits": patch.get("edits", []),
        "modified_layers": patch.get("modified_layers", []),
        "metrics": patch.get("metrics", {}),
        "created_at": patch.get("created_at"),
        "runtime": patch.get("runtime"),
        "sequential_edit": patch.get("sequential_edit", False),
        "mom2_adjustment": patch.get("mom2_adjustment"),
        "parent_patch_ids": patch.get("parent_patch_ids", []),
        "lineage_only": patch.get("lineage_only", False),
        "baked_into_model": baked_into_model or patch.get("baked_into_model", False),
    }


def _is_baked_lineage(patch: dict[str, Any]) -> bool:
    return bool(patch.get("lineage_only") or patch.get("baked_into_model"))


def _has_tensor_deltas(patch: dict[str, Any]) -> bool:
    return any(str(path).endswith(".safetensors") for path in patch.get("delta_paths", []))
