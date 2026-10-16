"""Prepare EasyEdit's optional NLTK tokenizer data without changing EasyEdit."""
from __future__ import annotations

import threading

from app.config import settings


class FluencyResourceError(RuntimeError):
    """The optional fluency tokenizer could not be prepared."""


_preparation_lock = threading.Lock()


def ensure_fluency_tokenizer_data() -> None:
    """Reuse installed data or download the missing English tokenizer once.

    Probe the tokenizer rather than guessing from NLTK's version: older NLTK
    needs ``punkt``, while newer releases need ``punkt_tab``. The managed data
    directory survives worker/backend restarts and Docker's mounted data volume.
    """
    try:
        import nltk
    except ImportError as exc:
        raise FluencyResourceError(
            "NLTK is not installed in the backend environment. Install EasyEdit's Python requirements."
        ) from exc

    cache_dir = settings.data_dir / "nltk_data"
    with _preparation_lock:
        cache_path = str(cache_dir)
        if cache_path not in nltk.data.path:
            nltk.data.path.insert(0, cache_path)
        try:
            nltk.word_tokenize("Fluency tokenizer check.")
            return
        except LookupError as exc:
            missing_resource = str(exc)
            if "punkt_tab" in missing_resource:
                package = "punkt_tab"
            elif "punkt" in missing_resource:
                package = "punkt"
            else:
                raise FluencyResourceError(
                    "NLTK could not initialize its English tokenizer: " + missing_resource
                ) from exc
        except Exception as exc:
            raise FluencyResourceError(
                f"NLTK could not initialize its English tokenizer. Check the tokenizer data in {cache_dir}. "
                f"Details: {exc}"
            ) from exc

        print(f"Preparing fluency: downloading NLTK {package} tokenizer data to {cache_dir}.", flush=True)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            installed = nltk.download(
                package, download_dir=cache_path, quiet=True,
                halt_on_error=True, raise_on_error=True, force=True,
            )
            if not installed:
                raise RuntimeError("The NLTK downloader reported an unsuccessful download.")
            # Do not report success merely because the download returned True.
            nltk.word_tokenize("Fluency tokenizer check.")
        except Exception as exc:
            raise FluencyResourceError(
                f"Could not automatically prepare NLTK {package} tokenizer data. "
                f"Check internet access and write permission for {cache_dir}, then retry evaluation. "
                f"Details: {exc}"
            ) from exc
        print("Fluency tokenizer data is ready.", flush=True)
