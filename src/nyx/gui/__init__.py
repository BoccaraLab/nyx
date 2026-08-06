"""The nyx scoring GUI: ``nyx-gui``, or ``python -m nyx.gui``.

One window with tabs down the side, one tab per decision the notebooks stop at
-- which channels, where the mains and the window are, where the EMG threshold
goes, which cluster is REM, and what to save. Every tab is reachable as soon as
its inputs exist, and each one shows whether what you are looking at is current
or stale.

Install it with::

    pip install "nyx-sleep[gui]"

The layering is worth knowing about if you are reading the code:

``session`` / ``filters`` / ``hypnogram``
    The pipeline state and its invalidation graph. **No Qt**, checked by a
    test, so it can be exercised without a display -- and so a future headless
    or real-time front end can reuse it.
``jobs`` / ``canvas`` / ``widgets`` / ``mainwindow`` / ``tabs``
    Qt. Observes the session, never analyses anything itself.
``viewers`` / ``review``
    ephyviewer, for browsing traces and correcting the hypnogram by hand.

Nothing in here re-implements any analysis: every number on screen comes from
the same functions the notebooks call.
"""

from __future__ import annotations

import os
import sys

__all__ = ["main", "run"]

#: Qt bindings that would be picked up if we did not choose one. PyQt5 and
#: PySide6 in a single process is a segfault, not an ImportError, so the choice
#: has to be made before anything imports Qt -- matplotlib, pyqtgraph and
#: ephyviewer all consult these.
_QT_ENVIRONMENT = {
    "QT_API": "pyside6",           # matplotlib
    "PYQTGRAPH_QT_LIB": "PySide6",  # pyqtgraph, and so ephyviewer
}


def _pin_qt_binding() -> None:
    for name, value in _QT_ENVIRONMENT.items():
        os.environ.setdefault(name, value)


def _missing_dependency(exc: Exception) -> str:
    return (
        f"The nyx GUI needs its optional dependencies, and one is missing:\n"
        f"    {type(exc).__name__}: {exc}\n\n"
        f"Install them with:\n"
        f"    pip install \"nyx-sleep[gui]\"\n"
    )


def run(argv: list[str] | None = None) -> int:
    """Open the GUI and run it until the window closes.

    Returns the Qt exit code. Importing this module does *not* import Qt --
    that happens here, after the binding is pinned.
    """
    _pin_qt_binding()

    try:
        from PySide6 import QtWidgets

        from nyx.gui.mainwindow import MainWindow
    except ImportError as exc:  # pragma: no cover - depends on the install
        sys.stderr.write(_missing_dependency(exc))
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        list(argv) if argv is not None else sys.argv
    )
    app.setApplicationName("nyx")

    window = MainWindow()
    window.show()
    return int(app.exec())


def main() -> int:
    """Console entry point for ``nyx-gui``."""
    return run()
