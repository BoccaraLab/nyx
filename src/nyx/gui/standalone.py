"""``nyx-ephyviewer``: ephyviewer's own standalone viewer, launchable.

ephyviewer 1.8.0's standalone viewer has two bugs that nyx works round here,
before handing over to ephyviewer's launcher unchanged -- so the file dialog,
the format list and the command line are all upstream's:

- The ``ephyviewer`` command cannot start: its ``scripts.py`` does
  ``from ephyviewer import __version__``, and since the pyproject migration
  (74b25d3) the package's ``__init__`` no longer defines one. Filled in from
  the installed metadata.
- Picking a folder-based format (Open Ephys, Spike2 folders, ...) and clicking
  "Select files" asks for ``QFileDialog.DirectoryOnly``, a Qt 5 name that Qt 6
  removed. nyx runs on PySide6, so it is aliased to ``FileMode.Directory``,
  which is what Qt 6 offers instead.

Upstream ``master`` has both. Once a fixed ephyviewer is released this can go,
and ``ephyviewer`` works again on its own.
"""

from __future__ import annotations

import sys

from nyx.gui import _missing_dependency, _pin_qt_binding

__all__ = ["main"]


def main() -> int:
    """Console entry point for ``nyx-ephyviewer``."""
    # Before ephyviewer imports pyqtgraph, for the same reason as nyx-gui.
    _pin_qt_binding()

    try:
        from importlib.metadata import version

        import ephyviewer

        if not hasattr(ephyviewer, "__version__"):
            ephyviewer.__version__ = version("ephyviewer")

        from PySide6.QtWidgets import QFileDialog

        if not hasattr(QFileDialog, "DirectoryOnly"):
            setattr(QFileDialog, "DirectoryOnly", QFileDialog.FileMode.Directory)  # noqa: B010

        from ephyviewer.scripts import launch_standalone_ephyviewer
    except ImportError as exc:  # pragma: no cover - depends on the install
        sys.stderr.write(_missing_dependency(exc))
        return 1

    launch_standalone_ephyviewer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
