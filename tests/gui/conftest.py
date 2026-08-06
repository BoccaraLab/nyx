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
