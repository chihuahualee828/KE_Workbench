"""Coordinate patch mutations with controller-owned edit/evaluation jobs.

The runtime has one model worker. Reservations include queued jobs and remain
held until their runner (or cancellation cleanup) actually exits.
"""
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import threading

from fastapi import HTTPException


class OperationGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._activities = 0
        self._patch_change = False

    @contextmanager
    def reserve(self, *, patch_change=False):
        with self._lock:
            if self._patch_change or (patch_change and self._activities):
                raise HTTPException(409, "Wait for editing, evaluation, or the current patch change to finish.")
            if patch_change:
                self._patch_change = True
            else:
                self._activities += 1
        try:
            yield
        finally:
            with self._lock:
                if patch_change:
                    self._patch_change = False
                else:
                    self._activities -= 1

    def protect(self, *, patch_change=False):
        def decorate(function):
            @wraps(function)
            def guarded(*args, **kwargs):
                with self.reserve(patch_change=patch_change):
                    return function(*args, **kwargs)
            # Resolve postponed annotations in the endpoint's own module for FastAPI.
            guarded.__signature__ = signature(function, eval_str=True)
            return guarded
        return decorate

    @contextmanager
    def patch_deletion(self):
        # Serialize the deletion check and file removal with new admissions and
        # patch changes. Existing jobs continue; a busy deletion must leave the
        # active stack untouched and must never wait for the model worker.
        with self._lock:
            if self._patch_change:
                raise HTTPException(409, "Wait for the current patch change to finish.")
            yield bool(self._activities)

    def start_thread(self, *, target, args=(), kwargs=None):
        # Reserve BEFORE starting: a queued runner must also exclude patch changes.
        reservation = self.reserve()
        reservation.__enter__()

        def run():
            try:
                target(*args, **(kwargs or {}))
            finally:
                reservation.__exit__(None, None, None)

        try:
            thread = threading.Thread(target=run, args=(), daemon=True)
            thread.start()
        except BaseException:
            reservation.__exit__(None, None, None)
            raise
        return thread


operation_guard = OperationGuard()
