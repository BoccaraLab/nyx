"""The nyx scoring GUI: ``nyx-gui``, or ``python -m nyx.gui``.

One window with tabs down the side, one tab per decision the notebooks stop at
-- which channels, where the mains and the window are, where the EMG threshold
goes, which cluster is REM, and what to save. Every tab is reachable as soon as
its inputs exist, and each one shows whether what you are looking at is current
or stale.

Install it from a clone with::

    pip install -e ".[gui]"

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
    # From a clone rather than from PyPI: nyx is installed from the repository
    # for now, and telling someone to pip install a name that does not resolve
    # is worse than telling them nothing.
    return (
        f"The nyx GUI needs its optional dependencies, and one is missing:\n"
        f"    {type(exc).__name__}: {exc}\n\n"
        f"Install them from your clone of the repository:\n"
        f"    pip install -e \".[gui]\"\n\n"
        f"See https://github.com/BoccaraLab/nyx\n"
    )


def _parser():
    import argparse

    import nyx

    parser = argparse.ArgumentParser(
        prog="nyx-gui",
        description=(
            "Score a sleep recording: one window, one tab per decision. "
            "Runs the same pipeline as the example notebooks."
        ),
        epilog=(
            "With no arguments it opens empty; the Recording tab has a "
            "'Try the demo recording' button that needs no data."
        ),
    )
    parser.add_argument(
        "recording", nargs="?",
        help="Recording to open (.edf, .npz, or a spikeinterface folder).",
    )
    parser.add_argument(
        "--params", metavar="NAME",
        help=(
            "Parameter preset or path to a JSON file. Presets: "
            + ", ".join(nyx.available_params())
        ),
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help=(
            "Resume from a config or a previous run's run.json, with its "
            "recording, window and decisions already filled in."
        ),
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Open the synthetic demo recording straight away.",
    )
    parser.add_argument(
        "--version", action="version", version=f"nyx {nyx.__version__}"
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Open the GUI and run it until the window closes.

    Returns the Qt exit code. Importing this module does *not* import Qt --
    that happens here, after the binding is pinned, so ``--help`` and
    ``--version`` work even without the GUI extras installed.
    """
    arguments = _parser().parse_args(argv)

    _pin_qt_binding()

    try:
        from PySide6 import QtWidgets

        from nyx.gui import branding
    except ImportError as exc:  # pragma: no cover - depends on the install
        sys.stderr.write(_missing_dependency(exc))
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName("nyx")
    branding.apply_to(app)

    # Up before the heavy imports below: scipy, sklearn, spikeinterface, mne
    # and pyqtgraph together are several seconds of nothing visible happening.
    screen = branding.splash()
    if screen is not None:
        screen.show()
        app.processEvents()

    from nyx.gui.mainwindow import MainWindow
    from nyx.gui.session import ScoringSession

    session = None
    if arguments.config:
        import nyx

        session = ScoringSession.from_config(nyx.load_config(arguments.config))

    window = MainWindow(session)

    if arguments.params:
        window.tabs[0].preset.setCurrentText(arguments.params)
    if arguments.recording:
        window.tabs[0].path.setText(arguments.recording)
        window.tabs[0]._offer_channels()
    if arguments.demo:
        window.tabs[0]._load_demo()

    window.show()
    if screen is not None:
        # Closed against the window it was covering for, so it cannot outlive
        # it and sit on top of everything.
        screen.finish(window)
    return int(app.exec())


def main() -> int:
    """Console entry point for ``nyx-gui``."""
    return run()
