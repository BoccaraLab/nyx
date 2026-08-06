"""Running the slow parts off the UI thread.

``compute_sleep_pca`` is minutes on a night of data. Running it on the UI
thread freezes the window; running it badly corrupts the session. This module
is the one place that decides how it runs.

``QThreadPool`` with one thread, not ``QThread``
------------------------------------------------

The work is one-shot compute jobs, which is exactly ``QRunnable``'s shape.
``QThread`` with ``moveToThread`` is for long-lived objects with their own
event loop; here it would buy nothing and add the classic *QThread: Destroyed
while thread is still running*. Capping the pool at one thread also serialises
the jobs for free, which matters: two pipeline jobs must never run at once,
because they mutate the same session.

Cancellation, honestly
----------------------

Python cannot preempt a thread, and ``compute_sleep_pca`` is a straight line
into scipy and sklearn with no callback to poll. So a running stage cannot be
stopped. What *can* be done is to stop caring: :meth:`JobRunner.abandon` bumps
the session's generation, and every result arrives tagged with the generation
it started under, so a stale one is dropped on arrival. The thread finishes and
dies; nothing is corrupted. The button therefore says "Abandon", not "Cancel".

No matplotlib here
------------------

A job returns nyx data objects and nothing else -- never a ``Figure``, never
an ``Axes``. This module imports no matplotlib at all, which makes that rule
mechanically checkable rather than a convention.
"""

from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

__all__ = ["Job", "JobSignals", "JobRunner"]


class JobSignals(QObject):
    """Signals for a :class:`Job`.

    ``QRunnable`` is not a ``QObject``, so its signals have to live on a
    separate object it owns.
    """

    #: A human-readable description of what started.
    started = Signal(str)
    #: ``(value, generation)`` -- whatever the callable returned.
    finished = Signal(object, int)
    #: ``(message, traceback, generation)``.
    failed = Signal(str, str, int)
    #: ``(list of warning messages, generation)``.
    warned = Signal(list, int)
    #: Emitted last, whatever happened, so the UI can re-enable itself.
    done = Signal(int)


class Job(QRunnable):
    """One blocking callable, run on the pool.

    Warnings raised inside are captured and re-emitted rather than printed.
    nyx warns in exactly the places a user must notice -- unnamed clusters,
    scoring without an EMG, unrecognised clustering settings -- and a notebook
    user sees those on stderr. A GUI user would not.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args,
        description: str = "",
        generation: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.description = description or getattr(fn, "__name__", "working")
        self.generation = int(generation)
        self.signals = JobSignals()

    def run(self) -> None:  # noqa: D102 - QRunnable's entry point
        self.signals.started.emit(self.description)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = self.fn(*self.args, **self.kwargs)
            messages = [str(w.message) for w in caught]
            if messages:
                self.signals.warned.emit(messages, self.generation)
            self.signals.finished.emit(value, self.generation)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            self.signals.failed.emit(
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
                self.generation,
            )
        finally:
            self.signals.done.emit(self.generation)


class JobRunner(QObject):
    """Submits jobs one at a time and reports what happens to them.

    Connect to the signals here rather than to an individual job's: the runner
    outlives any one job, so the connections can be made once.
    """

    started = Signal(str)
    finished = Signal(object, int)
    failed = Signal(str, str, int)
    warned = Signal(list, int)
    done = Signal(int)
    #: True while a job is running, so the UI can disable its inputs.
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # One at a time: concurrent jobs would race on the session.
        self._pool.setMaxThreadCount(1)
        self._running = 0

    def submit(
        self, fn: Callable[..., Any], *args, description: str = "",
        generation: int = 0, **kwargs,
    ) -> Job:
        job = Job(
            fn, *args, description=description, generation=generation, **kwargs
        )
        job.signals.started.connect(self.started)
        job.signals.finished.connect(self.finished)
        job.signals.failed.connect(self.failed)
        job.signals.warned.connect(self.warned)
        job.signals.done.connect(self._job_done)
        job.signals.done.connect(self.done)

        self._running += 1
        if self._running == 1:
            self.busy_changed.emit(True)
        self._pool.start(job)
        return job

    def _job_done(self, _generation: int) -> None:
        self._running = max(0, self._running - 1)
        if self._running == 0:
            self.busy_changed.emit(False)

    def busy(self) -> bool:
        return self._running > 0

    def wait(self, timeout_ms: int = -1) -> bool:
        """Block until the queue drains. For tests and for shutdown."""
        return bool(self._pool.waitForDone(timeout_ms))
