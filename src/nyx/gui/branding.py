"""The nyx mark: the window icon, and the splash while the imports happen.

Starting the GUI is not instant -- ``import nyx`` pulls in scipy, sklearn,
spikeinterface and mne, and ephyviewer pulls in pyqtgraph on top. That is
several seconds of nothing happening, which reads as nothing *working*. The
splash says otherwise, and goes away on its own once the window is up.

The artwork is Letizia's: the Nyx cube, cut from ``Nyx_logo.svg``.
"""

from __future__ import annotations

import importlib.resources

__all__ = ["icon", "logo_path", "splash", "apply_to"]

#: Files shipped alongside this module.
LOGO = "nyx_logo.png"     # the cube, cropped to its own bounds
ICON = "nyx_icon.png"     # square, padded rather than stretched
WINDOWS_ICON = "nyx.ico"  # multi-resolution, for the taskbar


def logo_path(name: str = LOGO) -> str | None:
    """Where a bundled image lives, or ``None`` if it was not installed."""
    try:
        path = importlib.resources.files("nyx.gui") / "resources" / name
    except (ModuleNotFoundError, AttributeError):  # pragma: no cover
        return None
    return str(path) if path.is_file() else None


def icon():
    """The window and taskbar icon, or an empty one if the file is missing."""
    from PySide6.QtGui import QIcon

    for name in (WINDOWS_ICON, ICON, LOGO):
        path = logo_path(name)
        if path:
            return QIcon(path)
    return QIcon()


def splash(message: str = "Starting nyx..."):
    """A splash screen showing the logo, or ``None`` when there is no logo.

    Returned rather than shown-and-forgotten so the caller can close it on the
    window it was covering for -- a splash that outlives its window sits on
    top of everything.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    path = logo_path()
    if not path:
        return None

    pixmap = QPixmap(path)
    if pixmap.isNull():  # pragma: no cover - a corrupt install
        return None

    pixmap = pixmap.scaledToWidth(420, Qt.SmoothTransformation)
    screen = QSplashScreen(pixmap)
    screen.showMessage(
        message,
        Qt.AlignBottom | Qt.AlignHCenter,
        QColor("#402864"),   # the cube's own purple
    )
    return screen


def apply_to(app) -> None:
    """Give the application its icon.

    On Windows the taskbar groups by "application user model id" and falls
    back to the interpreter's, so without this the icon in the taskbar is
    Python's rather than nyx's however the window is set.
    """
    app.setWindowIcon(icon())

    try:  # pragma: no cover - Windows only
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "no.uio.ncmm.nyx"
        )
    except Exception:  # noqa: BLE001 - not Windows, or not permitted
        pass
