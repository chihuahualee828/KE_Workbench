"""Per-metric EasyEdit protocols without mutating its source or edit profiles."""
import importlib
from copy import copy
from typing import Any

METRICS = ("reliability", "generalization", "locality", "portability")
PROTOCOLS = ("traditional", "LLM-judge", "generate-text")


def normalize_protocols(protocols: dict[str, str] | None, modes: list[str] | None = None) -> dict[str, str]:
    protocols = protocols or {}
    if any(metric not in METRICS or protocol not in PROTOCOLS for metric, protocol in protocols.items()):
        raise ValueError("Unsupported evaluation metric or protocol.")
    if any(metric != "locality" and protocol == "generate-text" for metric, protocol in protocols.items()):
        raise ValueError("generate-text is supported only for locality preservation.")
    if protocols.get("locality") == "LLM-judge":
        raise ValueError("Locality supports traditional or generate-text preservation, not LLM judging.")
    enabled = METRICS if modes is None else modes
    if any(metric not in METRICS for metric in enabled):
        raise ValueError("Unsupported evaluation metric.")
    return {metric: protocols.get(metric, "traditional") for metric in METRICS if metric in enabled}


class _CompletionTokenizer:
    """Let EasyEdit generate raw completions for tokenizers without a chat template.

    All real tokenization and stop-string handling remain with the loaded
    tokenizer. No shared tokenizer state or EasyEdit source is changed.
    """
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool = False, **kwargs: Any) -> Any:
        prompt = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return self._tokenizer.encode(prompt) if tokenize else prompt


class ProtocolEvaluator:
    """Collect just the requested metric, retaining the text used in its score."""
    def __init__(self, model: Any, tokenizer: Any, hparams: Any, model_name: str,
                 protocols: dict[str, str], include_generations: bool, check_running: Any):
        self.model, self.tokenizer, self.hparams, self.model_name = model, tokenizer, hparams, model_name
        self.protocols, self.include_generations, self.check_running = protocols, include_generations, check_running
        self.module = importlib.import_module("easyeditor.evaluate")
        self.params: dict[str, Any] = {}
        for metric, protocol in protocols.items():
            params = copy(hparams)
            params.evaluation_type = None if protocol == "traditional" else protocol
            params.api_key = None
            if protocol == "LLM-judge":
                from app.config import require_deepseek_api_key
                params.api_key = require_deepseek_api_key()
            self.params[metric] = params
        self.generation_tokenizer = tokenizer if getattr(tokenizer, "chat_template", None) else _CompletionTokenizer(tokenizer)
        if self.generation_tokenizer is tokenizer and any(protocol != "traditional" for protocol in protocols.values()):
            try:
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "Evaluation prompt."}],
                    tokenize=False, add_generation_prompt=True,
                )
            except (ValueError, TypeError):
                raise ValueError("This model's chat template is incompatible with EasyEdit generation (system/user messages).") from None

    @staticmethod
    def prompt(record: dict, metric: str) -> str | None:
        if metric == "reliability":
            return record["prompt"]
        if metric == "generalization":
            return record.get("rephrase_prompt")
        section = record.get(metric, {})
        return section.get("neighborhood" if metric == "locality" else "derived", {}).get("prompt")

    def baseline(self, record: dict) -> dict:
        result: dict[str, Any] = {"locality": {}, "portability": {}}
        texts: dict[str, list[str]] = {}
        for metric, protocol in self.protocols.items():
            prompt = self.prompt(record, metric)
            if not prompt:
                continue
            self.check_running()
            params = self.params[metric]
            if metric == "locality":
                result["locality"] = self.module.compute_locality_quality(
                    self.model, self.model_name, params,
                    self.tokenizer if protocol == "traditional" else self.generation_tokenizer,
                    "neighborhood", prompt, record["locality"]["neighborhood"]["ground_truth"], params.device,
                )
            elif protocol == "LLM-judge" and self.include_generations:
                # Original-model display needs generation, not a second paid
                # judge call. This uses the very same EasyEdit generator.
                if prompt not in texts:
                    generator = importlib.import_module("easyeditor.evaluate.evaluate_utils").generate_texts
                    texts[prompt] = generator(self.model, self.generation_tokenizer, params, prompt, params.device)
                if metric == "portability":
                    result[metric]["derived_gen_content"] = texts[prompt]
                else:
                    result["rewrite_gen_content" if metric == "reliability" else "rephrase_gen_content"] = texts[prompt]
        return result

    def evaluate(self, record: dict) -> dict:
        result: dict[str, Any] = {"locality": {}, "portability": {}, "metric_meta": {}}
        for metric, protocol in self.protocols.items():
            prompt = self.prompt(record, metric)
            if not prompt:
                continue
            self.check_running()
            params = self.params[metric]
            tok = self.tokenizer if protocol == "traditional" else self.generation_tokenizer
            try:
                if metric in ("reliability", "generalization"):
                    quality = self.module.compute_rewrite_or_rephrase_quality(
                        self.model, self.model_name, params, tok, prompt, record["target_new"],
                        device=params.device, test_rephrase=metric == "generalization",
                    )
                    result["metric_meta"].update(quality.get("metric_meta", {}))
                    result.update({key: value for key, value in quality.items() if key != "metric_meta"})
                else:
                    key = "neighborhood" if metric == "locality" else "derived"
                    evaluator = self.module.compute_locality_quality if metric == "locality" else self.module.compute_portability_quality
                    result[metric] = evaluator(
                        self.model, self.model_name, params, tok, key, prompt,
                        record[metric][key]["ground_truth"], params.device,
                    )
                    if metric == "portability":
                        # Portability's lower-level evaluator does not attach
                        # metadata in the installed EasyEdit version.
                        from easyeditor.evaluate.metric_meta import build_lm_metric_meta
                        meta = build_lm_metric_meta("portability", params, self.model_name)
                        meta["result_key"] = "portability." + key + "_acc"
                        result["metric_meta"]["portability." + key] = meta
            except Exception:
                if protocol != "LLM-judge":
                    raise
                # Provider exception bodies can echo credentials. Never put
                # them into persisted job logs or the browser's error response.
                try:
                    self.check_running()
                except Exception as canceled:
                    raise canceled from None
                raise RuntimeError("LLM judging failed. Check the backend DEEPSEEK_API_KEY, network, and provider access.") from None
        return result

    def finalize(self, metrics: dict, baseline: dict, record: dict, finalizer: Any) -> None:
        if "locality" in self.protocols and "locality" in record:
            finalizer({"pre": baseline, "post": metrics}, record, self.params["locality"], self.model_name)

    def text(self, metrics: dict, metric: str) -> str | None:
        from app.ml.editing.evaluation_reuse import generated_text
        if metric in ("reliability", "generalization"):
            return generated_text(metrics, "rewrite_gen_content" if metric == "reliability" else "rephrase_gen_content")
        return generated_text(metrics.get(metric, {}), "neighborhood_gen_content" if metric == "locality" else "derived_gen_content")
