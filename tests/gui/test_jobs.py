"""Running work off the UI thread."""

from __future__ import annotations

import threading
import warnings

import pytest

from nyx.gui.jobs import JobRunner

pytestmark = pytest.mark.gui


@pytest.fixture
def runner(qtbot):
    return JobRunner()


def test_a_job_runs_somewhere_that_is_not_the_ui_thread(runner, qtbot):
    here = threading.get_ident()
    where = {}

    with qtbot.waitSignal(runner.finished, timeout=5000):
        runner.submit(lambda: where.setdefault("thread", threading.get_ident()))

    assert where["thread"] != here


def test_finished_carries_what_the_callable_returned(runner, qtbot):
    with qtbot.waitSignal(runner.finished, timeout=5000) as caught:
        runner.submit(lambda: 6 * 7, generation=3)

    assert caught.args == [42, 3]


def test_a_raising_job_reports_the_traceback_rather_than_vanishing(runner, qtbot):
    def explode():
        raise ValueError("no")

    with qtbot.waitSignal(runner.failed, timeout=5000) as caught:
        runner.submit(explode)

    message, tb, _generation = caught.args
    assert "ValueError: no" in message
    assert "explode" in tb


def test_a_failed_job_does_not_emit_finished(runner, qtbot):
    finished = []
    runner.finished.connect(lambda *a: finished.append(a))

    def explode():
        raise RuntimeError("no")

    with qtbot.waitSignal(runner.failed, timeout=5000):
        runner.submit(explode)

    assert finished == []


def test_warnings_raised_inside_a_job_come_back_out(runner, qtbot):
    def noisy():
        # nyx warns where a user must notice -- unnamed clusters, no EMG.
        # On the command line those reach stderr; here they would be lost.
        warnings.warn("check the per-cluster spectra", stacklevel=1)
        return 1

    with qtbot.waitSignal(runner.warned, timeout=5000) as caught:
        runner.submit(noisy)

    messages, _generation = caught.args
    assert messages == ["check the per-cluster spectra"]


def test_busy_goes_up_and_comes_back_down(runner, qtbot):
    states = []
    runner.busy_changed.connect(states.append)

    with qtbot.waitSignal(runner.done, timeout=5000):
        runner.submit(lambda: None)
    runner.wait(5000)
    qtbot.wait(50)

    assert states[0] is True
    assert states[-1] is False
    assert not runner.busy()


def test_done_is_emitted_whether_it_worked_or_not(runner, qtbot):
    def explode():
        raise RuntimeError("no")

    with qtbot.waitSignal(runner.done, timeout=5000):
        runner.submit(explode)


def test_jobs_do_not_run_at_the_same_time(runner, qtbot):
    """Two pipeline jobs at once would race on the session."""
    overlapping = []
    running = threading.Event()

    def slow(tag):
        overlapping.append(running.is_set())
        running.set()
        threading.Event().wait(0.05)
        running.clear()
        return tag

    finished = []
    runner.finished.connect(lambda value, _g: finished.append(value))

    runner.submit(slow, "a")
    runner.submit(slow, "b")
    runner.wait(10_000)
    qtbot.waitUntil(lambda: len(finished) == 2, timeout=5000)

    assert not any(overlapping)
    assert finished == ["a", "b"]
