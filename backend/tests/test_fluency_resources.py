from concurrent.futures import ThreadPoolExecutor
import sys
from types import SimpleNamespace

import pytest

from app.ml.editing import fluency_resources as resources


@pytest.fixture
def tokenizer_data(monkeypatch, tmp_path):
    state = SimpleNamespace(
        available=False, required="punkt", calls=[], success=True,
        error=None, becomes_ready=True, checks=0,
    )
    cache_dir = tmp_path / "data" / "nltk_data"

    def tokenize(text):
        state.checks += 1
        if not state.available:
            raise LookupError(f"Resource {state.required} not found")
        return text.split()

    def download(package, **kwargs):
        state.calls.append((package, kwargs))
        if state.error:
            raise state.error
        if state.success and state.becomes_ready:
            state.available = True
        return state.success

    nltk = SimpleNamespace(data=SimpleNamespace(path=[]), word_tokenize=tokenize, download=download)
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setattr(resources, "settings", SimpleNamespace(data_dir=cache_dir.parent))
    return state, nltk, cache_dir


def test_existing_tokenizer_data_does_not_download_or_create_a_directory(tokenizer_data):
    state, nltk, cache_dir = tokenizer_data
    state.available = True
    resources.ensure_fluency_tokenizer_data()
    assert state.calls == []
    assert str(cache_dir) in nltk.data.path
    assert not cache_dir.exists()


@pytest.mark.parametrize("package", ["punkt", "punkt_tab"])
def test_downloads_only_the_missing_resource_and_reuses_it(tokenizer_data, package, capsys):
    state, nltk, cache_dir = tokenizer_data
    state.required = package
    resources.ensure_fluency_tokenizer_data()
    resources.ensure_fluency_tokenizer_data()
    assert state.calls == [(package, {
        "download_dir": str(cache_dir), "quiet": True,
        "halt_on_error": True, "raise_on_error": True, "force": True,
    })]
    assert cache_dir.is_dir()
    assert nltk.data.path.count(str(cache_dir)) == 1
    assert "Fluency tokenizer data is ready." in capsys.readouterr().out


def test_simultaneous_requests_do_not_download_twice(tokenizer_data):
    state, _, _ = tokenizer_data
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(resources.ensure_fluency_tokenizer_data) for _ in range(8)]
        for future in futures:
            future.result(timeout=5)
    assert len(state.calls) == 1


@pytest.mark.parametrize("failure", ["false", "network", "permission", "incomplete"])
def test_failed_download_reports_an_error_and_can_retry(tokenizer_data, failure):
    state, _, cache_dir = tokenizer_data
    if failure == "false":
        state.success = False
    elif failure == "incomplete":
        state.becomes_ready = False
    else:
        state.error = PermissionError("read-only filesystem") if failure == "permission" else OSError("network unavailable")
    with pytest.raises(resources.FluencyResourceError) as captured:
        resources.ensure_fluency_tokenizer_data()
    assert "Could not automatically prepare NLTK" in str(captured.value)
    assert str(cache_dir) in str(captured.value)
    state.success = state.becomes_ready = True
    state.error = None
    resources.ensure_fluency_tokenizer_data()
    assert state.available is True
    assert len(state.calls) == 2


def test_unexpected_missing_resource_is_not_downloaded(tokenizer_data):
    state, _, _ = tokenizer_data
    state.required = "unrelated_resource"
    with pytest.raises(resources.FluencyResourceError, match="could not initialize"):
        resources.ensure_fluency_tokenizer_data()
    assert state.calls == []


def test_unreadable_existing_tokenizer_data_reports_a_resource_error(tokenizer_data):
    state, nltk, _ = tokenizer_data
    def unreadable(_):
        raise PermissionError("tokenizer data is unreadable")
    nltk.word_tokenize = unreadable
    with pytest.raises(resources.FluencyResourceError, match="tokenizer data is unreadable"):
        resources.ensure_fluency_tokenizer_data()
    assert state.calls == []


def test_missing_nltk_library_is_not_installed_at_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "nltk", None)
    with pytest.raises(resources.FluencyResourceError, match="NLTK is not installed"):
        resources.ensure_fluency_tokenizer_data()
