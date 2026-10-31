"""Rebuildable query index; JSON artifacts remain the source of truth.

Only changed artifacts are decoded. SQLite stores compact summaries and individual
edit rows, so a page read does not deserialize an entire patch/history. File
signatures also catch changes made by the model worker and external deletions.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from typing import Any

from app.storage.local_store import local_store

PATCH_FIELDS = (
    "patch_id", "base_model", "method", "modified_layers", "created_at", "delta_paths",
    "runtime", "sequential_edit", "mom2_adjustment", "parent_patch_ids", "layer_source",
    "lineage_only", "baked_into_model", "series_edit", "series_fact_count",
)


def patch_summary(patch: dict) -> dict:
    return {**{key: patch[key] for key in PATCH_FIELDS if key in patch},
            "edits": [], "edit_count": len(patch.get("edits") or [])}


def evaluation_fact(patch_id: str, index: int, edit: dict) -> dict | None:
    subject, relation, target = (str(edit.get(key) or "").strip()
                                 for key in ("subject", "relation_prompt", "new_object"))
    if not subject or not relation or not target:
        return None
    metadata = edit.get("source_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    def optional(field: str) -> str:
        values = [edit.get(field), metadata.get(field)]
        if field == "rephrase_prompt":
            values += [edit.get(field + "s"), metadata.get(field + "s")]
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if field == "rephrase_prompt" and isinstance(value, list):
                first = next((v.strip() for v in value if isinstance(v, str) and v.strip()), "")
                if first:
                    return first
        return ""

    return {
        "key": f"{patch_id}:{index}", "subject": subject, "relation_prompt": relation,
        "prompt": relation.replace("{}", subject) if "{}" in relation else f"{subject} {relation}",
        "new_object": target, "ground_truth": str(edit.get("ground_truth") or "").strip(),
        **{field: optional(field) for field in (
            "rephrase_prompt", "locality_prompt", "locality_ground_truth",
            "portability_prompt", "portability_ground_truth",
        )},
    }


_catalog_lock = threading.RLock()


def catalog_path() -> Path:
    # DATA_DIR can be an SMB/NFS mount, which cannot reliably host SQLite locks.
    # This disposable index contains no authoritative data and can be rebuilt.
    directory = Path(tempfile.gettempdir()) / f"kew-catalog-{os.getuid()}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = hashlib.sha256(str(local_store.root.resolve()).encode()).hexdigest()
    return directory / f"{identity}.sqlite3"


@contextmanager
def catalog():
    # Serialize index reconciliation in this API process. Model workers continue
    # to write JSON; signatures pick those changes up on the next query.
    with _catalog_lock, _open_catalog() as db:
        yield db


@contextmanager
def _open_catalog():
    db = sqlite3.connect(catalog_path(), timeout=30)
    try:
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS records (
                path TEXT PRIMARY KEY, kind TEXT, id TEXT, signature TEXT,
                created TEXT, model TEXT, status TEXT, experiment TEXT, summary TEXT
            );
            CREATE INDEX IF NOT EXISTS records_query ON records(kind, created DESC, id DESC);
            CREATE INDEX IF NOT EXISTS records_experiment ON records(experiment);
            CREATE INDEX IF NOT EXISTS records_identity ON records(kind, id);
            CREATE TABLE IF NOT EXISTS edits (
                patch TEXT, position INTEGER, data TEXT, fact TEXT, prompt TEXT,
                target TEXT, search TEXT, locality INTEGER, portability INTEGER,
                PRIMARY KEY(patch, position)
            );
            CREATE TABLE IF NOT EXISTS patch_fact_search (
                patch TEXT, position INTEGER, text TEXT,
                PRIMARY KEY(patch, position)
            );
        """)
        yield db
        db.commit()
    finally:
        db.close()


def sync_record(db: sqlite3.Connection, path: str, kind: str) -> None:
    source = local_store.root / path
    stat = source.stat()
    # Reindex existing JSON artifacts once when the searchable fact format changes.
    version = "facts-v1:" if kind == "patch" else ""
    signature = f"{version}{stat.st_mtime_ns}:{stat.st_size}:{stat.st_ino}"
    current = db.execute("SELECT signature FROM records WHERE path=?", (path,)).fetchone()
    if current and current[0] == signature:
        return
    data = local_store.read_json(path)
    if kind == "patch":
        identity = data["patch_id"]
        summary = patch_summary(data)
        db.execute("DELETE FROM edits WHERE patch=?", (identity,))
        db.execute("DELETE FROM patch_fact_search WHERE patch=?", (identity,))
        for index, edit in enumerate(data.get("edits") or []):
            fact = evaluation_fact(identity, index, edit) if isinstance(edit, dict) else None
            db.execute("INSERT INTO edits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                identity, index, json.dumps(edit), json.dumps(fact) if fact else None,
                fact["prompt"] if fact else None, fact["new_object"] if fact else None,
                "\n".join(fact[k] for k in ("subject", "ground_truth", "new_object")).lower() if fact else "",
                int(bool(fact and fact["locality_prompt"])), int(bool(fact and fact["portability_prompt"])),
            ))
            text = "\n".join(_fact_text(edit))
            if fact:
                text += "\n" + fact["prompt"]
            db.execute("INSERT INTO patch_fact_search VALUES (?, ?, ?)", (identity, index, text.casefold()))
    else:
        from app.services.experiment_service import _experiment_list_summary, _job_summary
        identity = data["id"]
        summary = _job_summary(data) if kind == "job" else _experiment_list_summary(data)
    db.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        path, kind, identity, signature, data.get("created_at", ""),
        data.get("model_id", data.get("base_model", "")), data.get("status", ""),
        data.get("experiment_id", ""), json.dumps(summary),
    ))


def sync_collection(db: sqlite3.Connection, collection: str) -> None:
    paths = ((local_store.root / "patches").glob("*/patch.json") if collection == "patch"
             else (local_store.root / "experiments").glob("*.json"))
    present = set()
    for source in paths:
        kind = "patch" if collection == "patch" else "job" if source.stem.startswith("job_") else "experiment"
        if kind == "experiment" and not source.stem.startswith("exp_"):
            continue
        path = str(source.relative_to(local_store.root))
        try:
            sync_record(db, path, kind)
            present.add(path)
        except FileNotFoundError:
            pass
    kinds = ("patch", "patch") if collection == "patch" else ("experiment", "job")
    for path, kind, identity in db.execute("SELECT path, kind, id FROM records WHERE kind IN (?, ?)", kinds).fetchall():
        if path not in present:
            db.execute("DELETE FROM records WHERE path=?", (path,))
            if kind == "patch":
                db.execute("DELETE FROM edits WHERE patch=?", (identity,))
                db.execute("DELETE FROM patch_fact_search WHERE patch=?", (identity,))


def _fact_text(value: Any):
    """Index fact values (including imported optional fields), never JSON keys."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _fact_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _fact_text(child)


def _history_filter(kind: str, model_id: str | None, status: str | None,
                    query: str, method: str | None) -> tuple[str, list[str]]:
    where, args = "kind=?", [kind]
    for column, value in (("model", model_id), ("status", status)):
        if value is not None:
            where += f" AND {column}=?"
            args.append(value)
    if kind == "patch":
        if method:
            where += " AND lower(json_extract(summary, '$.method'))=?"
            args.append(method.lower())
        if query.strip():
            where += " AND EXISTS (SELECT 1 FROM patch_fact_search f WHERE f.patch=records.id AND instr(f.text, ?)>0)"
            args.append(query.strip().casefold())
    return where, args


def history_page(kind: str, offset: int = 0, limit: int | None = 50,
                 model_id: str | None = None, status: str | None = None,
                 *, query: str = "", method: str | None = None, sort: str = "newest") -> dict[str, Any]:
    if sort not in {"oldest", "newest"}:
        raise ValueError("Unknown patch sort order")
    with catalog() as db:
        sync_collection(db, kind)
        where, args = _history_filter(kind, model_id, status, query, method)
        total = db.execute(f"SELECT COUNT(*) FROM records WHERE {where}", args).fetchone()[0]
        counts = dict(db.execute(f"SELECT status, COUNT(*) FROM records WHERE {where} GROUP BY status", args))
        direction = "ASC" if sort == "oldest" else "DESC"
        rows = db.execute(f"SELECT summary, id FROM records WHERE {where} ORDER BY created {direction}, id {direction} LIMIT ? OFFSET ?",
                          [*args, limit if limit is not None else -1, offset]).fetchall()
        items = []
        for raw, identity in rows:
            item = json.loads(raw)
            if kind == "experiment":
                job = db.execute("SELECT summary FROM records WHERE kind='job' AND experiment=? ORDER BY created DESC LIMIT 1", (identity,)).fetchone()
                if job:
                    item["job"] = json.loads(job[0])
            items.append(item)
        page = {"items": items, "total": total, "offset": offset, "limit": limit, "status_counts": counts}
        if kind == "patch":
            child_counts = dict(db.execute("""
                SELECT json_extract(summary, '$.parent_patch_ids[#-1]'), count(*) FROM records
                WHERE kind='patch' AND json_extract(summary, '$.sequential_edit')=1
                GROUP BY 1
            """))
            for item in items:
                item["lineage_child_count"] = child_counts.get(item["patch_id"], 0)
            # Global facets remain available even when the current query has no matches.
            page["methods"] = [row[0] for row in db.execute(
                "SELECT DISTINCT lower(json_extract(summary, '$.method')) FROM records WHERE kind='patch' ORDER BY 1"
            ) if row[0]]
            page["model_ids"] = [row[0] for row in db.execute(
                "SELECT DISTINCT model FROM records WHERE kind='patch' ORDER BY model"
            ) if row[0]]
        return page


def patch_selection_metadata(query: str = "", method: str | None = None, model_id: str | None = None):
    """Return matching IDs and the whole lineage, without deserializing edit facts."""
    with catalog() as db:
        sync_collection(db, "patch")
        where, args = _history_filter("patch", model_id, None, query, method)
        ids = [row[0] for row in db.execute(f"SELECT id FROM records WHERE {where} ORDER BY created DESC, id DESC", args)]
        parents = {row[0]: json.loads(row[1] or "[]") for row in db.execute(
            "SELECT id, json_extract(summary, '$.parent_patch_ids') FROM records WHERE kind='patch'"
        )}
        return ids, parents


def patch_edits_page(patch_id: str, offset: int, limit: int) -> dict:
    with catalog() as db:
        sync_record(db, f"patches/{patch_id}/patch.json", "patch")
        total = db.execute("SELECT COUNT(*) FROM edits WHERE patch=?", (patch_id,)).fetchone()[0]
        items = [json.loads(row[0]) for row in db.execute(
            "SELECT data FROM edits WHERE patch=? ORDER BY position LIMIT ? OFFSET ?", (patch_id, limit, offset))]
        return {"patch_id": patch_id, "offset": offset, "limit": limit, "total": total, "items": items}


def active_facts_page(patch_ids: list[str], offset: int, limit: int, query: str = "") -> dict:
    with catalog() as db:
        for patch_id in patch_ids:
            sync_record(db, f"patches/{patch_id}/patch.json", "patch")
        if not patch_ids:
            return {"facts": [], "total": 0, "locality_count": 0, "portability_count": 0, "matches": [], "match_total": 0}
        values = ",".join("(?, ?)" for _ in patch_ids)
        args = [value for rank, identity in enumerate(reversed(patch_ids)) for value in (identity, rank)]
        cte = f"""WITH active(patch, rank) AS (VALUES {values}), ranked AS (
            SELECT e.*, a.rank, ROW_NUMBER() OVER (PARTITION BY prompt, target ORDER BY rank, position) AS occurrence
            FROM edits e JOIN active a ON e.patch=a.patch WHERE fact IS NOT NULL
        ), facts AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY rank, position)-1 AS fact_index FROM ranked WHERE occurrence=1
        ) """
        total, locality, portability = db.execute(cte + "SELECT COUNT(*), COALESCE(SUM(locality),0), COALESCE(SUM(portability),0) FROM facts", args).fetchone()
        if query:
            match_total = db.execute(cte + "SELECT COUNT(*) FROM facts WHERE instr(search, ?)>0", [*args, query.lower()]).fetchone()[0]
            matches = [row[0] for row in db.execute(cte + "SELECT fact_index FROM facts WHERE instr(search, ?)>0 ORDER BY fact_index LIMIT ? OFFSET ?", [*args, query.lower(), limit, offset])]
            return {"facts": [], "total": total, "locality_count": locality, "portability_count": portability, "matches": matches, "match_total": match_total}
        facts = [json.loads(row[0]) for row in db.execute(cte + "SELECT fact FROM facts ORDER BY fact_index LIMIT ? OFFSET ?", [*args, limit, offset])]
        return {"facts": facts, "total": total, "locality_count": locality, "portability_count": portability, "matches": [], "match_total": 0}
