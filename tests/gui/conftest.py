"""Qt tests: offscreen, with the binding pinned before anything imports Qt.

These are marked ``gui`` and skipped when the optional dependencies are not
installed, so ``pytest`` on a base install still passes.

Pinning matters here as much as it does in the application. PyQt5 may well be
installed alongside PySide6 -- it is what the old lab GUI used -- and loading
both into one process is a segfault rather than an ImportError. pytest-qt would
otherwise auto-detect whichever it finds first.

The root ``tests/conftest.py`` calls ``matplotlib.use("Agg")`` unconditionally,
and that is left alone: constructing a ``FigureCanvasQTAgg`` explicitly still
works and still needs a ``QApplication``, and it means the tests and the
application agree about backend state. The one thing Agg forbids is
``plt.show()``, which the GUI must never call anyway.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("PYTEST_QT_API", "pyside6")

import pytest  # noqa: E402

pytest.importorskip("PySide6", reason="needs nyx-sleep[gui]")
pytest.importorskip("pytestqt", reason="needs pytest-qt")

pytestmark = pytest.mark.gui

_exit_status = {}


def pytest_sessionfinish(session, exitstatus):
    _exit_status["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """End the process once pytest has reported, skipping Python's teardown.

    At interpreter exit Python frees whatever Qt objects are left in no
    particular order, sometimes after the QApplication has gone, and on
    Linux that segfaults -- after every test has passed, failing the run
    anyway. Nothing is left to do at that point but return the status.

    Not on Windows: there ``os._exit`` still runs the Qt DLLs' own clean-up,
    which is what crashes, and the normal exit never did.
    """
    import sys

    if "code" not in _exit_status or sys.platform == "win32":
        return

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status["code"])


@pytest.fixture
def session(synthetic, synthetic_recording, params):
    from nyx.gui.session import ScoringSession

    _eeg, _emg, _fs, truth = synthetic

    s = ScoringSession(params)
    # With the reference, so the agreement and confusion paths are exercised
    # too -- they are the ones most likely to break silently.
    s.set_recording(synthetic_recording, reference=truth)
    s.set_window((0.0, 2400.0))
    return s


@pytest.fixture
def scored(session):
    from nyx.gui.session import Stage

    session.compute_through(Stage.RESULT)
    return session
