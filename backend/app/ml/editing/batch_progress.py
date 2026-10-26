"""Observe real EasyEdit batch calls through the existing job log stream."""
from contextlib import contextmanager
from functools import wraps

from app.ml.editing.edit_cancellation import check_edit_cancellation


@contextmanager
def track_batch_progress(editor, fact_count: int, batch_size: int, *, enabled: bool = True):
    if not enabled:
        yield
        return
    total = (fact_count + batch_size - 1) // batch_size
    completed = 0
    apply_algo = editor.apply_algo

    @wraps(apply_algo)
    def apply_with_progress(*args, **kwargs):
        nonlocal completed
        check_edit_cancellation()
        print(f"Applying edit batch {completed + 1}/{total}.")
        result = apply_algo(*args, **kwargs)
        check_edit_cancellation()
        completed += 1
        print(f"Completed edit batch {completed}/{total}.")
        return result

    editor.apply_algo = apply_with_progress
    try:
        yield
    finally:
        editor.apply_algo = apply_algo
