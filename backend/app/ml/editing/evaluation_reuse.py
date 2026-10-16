"""Conservative reuse of step metrics; never confuse them with final-state metrics."""
from copy import deepcopy
from typing import Any


def flatten_metrics(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [entry for item in value for entry in flatten_metrics(item)]
    if isinstance(value, dict) and isinstance(value.get("post"), dict):
        return [value]
    return []


def matching_post(entries: list[dict], index: int, record: dict, hparams: Any, model_name: str) -> dict | None:
    # Locality's step baseline can include previous patches. Final evaluation
    # uses the unpatched model, so never reuse finalized locality scores here.
    # Portability is not part of the existing step-reuse validation either.
    if index >= len(entries) or record.get("locality") or record.get("portability"):
        return None
    entry = entries[index]
    request = entry.get("requested_rewrite", {})
    for key in ("prompt", "target_new") + (("rephrase_prompt",) if record.get("rephrase_prompt") else ()):
        if (request.get(key) or "") != (record.get(key) or ""):
            return None
    try:
        from easyeditor.evaluate.metric_meta import build_lm_metric_meta
        post = entry["post"]
        for section in ("rewrite", "rephrase") if record.get("rephrase_prompt") else ("rewrite",):
            expected = build_lm_metric_meta(section, hparams, model_name)
            if post.get("metric_meta", {}).get(section) != expected or expected["result_key"] not in post:
                return None
    except (ImportError, KeyError, TypeError):
        return None
    result = deepcopy(post)
    result.pop("fluency", None)  # Run the requested final-state fluency protocol.
    result["locality"] = {}
    result["portability"] = {}
    if not record.get("rephrase_prompt"):
        for key in list(result):
            if key.startswith("rephrase_"):
                del result[key]
        result.get("metric_meta", {}).pop("rephrase", None)
    return result


def generated_text(metrics: dict, key: str) -> str | None:
    value = metrics.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return None
