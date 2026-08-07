"""What every tab is.

Controls on the left, a dock area on the right. The dock area is ephyviewer's,
so its panels scroll together, carry a navigation toolbar, and can be dragged
side by side, tabbed, or torn off -- the layout is whatever you drag it into.

A tab owns one decision, declares the :class:`~nyx.gui.session.Stage` it
depends on, and implements:

``build_controls()``
    Return the widgets for the left-hand column. The run button is put above
    them automatically.
``build_docks()``
    Create the panels. Called once; :meth:`refresh` fills them.
``refresh()``
    Update from the session. Must be cheap and must never compute -- the
    session's accessors raise rather than compute, which keeps that honest.
``apply()``
    Push widget values into the session, before running.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nyx.gui.panels import DockHost
from nyx.gui.session import NotComputed, ScoringSession, Stage
from nyx.gui.widgets import WarningBanner

__all__ = ["Tab"]


class Tab(QWidget):
    """Base class for the tabs in the side rail."""

    title = "Tab"
    subtitle = ""
    stage: Stage = Stage.LOAD
    #: Whether the work is slow enough to need a worker and a button.
    needs_worker = True
    run_label = "Run"
    #: Width of the controls column.
    controls_width = 330

    run_requested = Signal(object)
    status = Signal(str)

    def __init__(self, session: ScoringSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._refreshing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        heading = QLabel(f"<h3 style='margin:0'>{self.title}</h3>")
        outer.addWidget(heading)
        if self.subtitle:
            note = QLabel(self.subtitle)
            note.setWordWrap(True)
            note.setStyleSheet("color: palette(mid);")
            outer.addWidget(note)

        self.warnings = WarningBanner()
        outer.addWidget(self.warnings)

        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        # -- controls, with the run button at the top where it can be seen
        column = QWidget()
        self._column = QVBoxLayout(column)
        self._column.setContentsMargins(0, 0, 8, 0)

        self.run_button = QPushButton(self.run_label)
        self.run_button.setMinimumHeight(32)
        self.run_button.setStyleSheet("font-weight: bold;")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.stage))
        if not self.needs_worker:
            self.run_button.hide()
        self._column.addWidget(self.run_button)

        for widget in self.build_controls() or []:
            if isinstance(widget, QWidget):
                self._column.addWidget(widget)
            else:
                self._column.addLayout(widget)
        self._column.addStretch(1)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(column)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setMinimumWidth(self.controls_width)
        split.addWidget(area)

        # -- panels
        self.docks = DockHost()
        self.docks.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        split.addWidget(self.docks)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([self.controls_width, 1000])

        self.build_docks()

    # -- to implement ------------------------------------------------------

    def build_controls(self) -> list:
        """Widgets for the left-hand column, top to bottom."""
        return []

    def build_docks(self) -> None:
        """Create the panels. Called once."""

    def refresh(self) -> None:
        """Update from the session. Cheap; never computes."""

    def apply(self) -> None:
        """Push widget values into the session."""

    # -- helpers -----------------------------------------------------------

    def safe_refresh(self) -> None:
        """:meth:`refresh`, tolerating a session that has not got there yet.

        Tabs are reachable before their inputs exist, so ``NotComputed`` is a
        normal state here rather than an error.
        """
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.refresh()
        except NotComputed:
            pass
        finally:
            self._refreshing = False

    def quiet(self, widget):
        """Context manager blocking a widget's signals while it is set."""
        return _Quiet(widget)

    def show_warnings(self, messages) -> None:
        self.warnings.show_messages(messages)

    def add_control(self, widget) -> None:
        """Add a widget to the controls column after construction."""
        self._column.insertWidget(self._column.count() - 1, widget)

    def shutdown(self) -> None:
        """Close the panels and stop their threads.

        Qt only delivers ``closeEvent`` to top-level windows, and the dock host
        is a child widget, so it never gets one. Its viewers own worker threads
        that they stop in *their* ``closeEvent`` -- left running against
        deleted C++ objects, they take the process down rather than raising.
        The window calls this for every tab on the way out.
        """
        self.docks.clear()

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        self.shutdown()
        super().closeEvent(event)


class _Quiet:
    def __init__(self, widget):
        self.widget = widget

    def __enter__(self):
        self._was = self.widget.blockSignals(True)
        return self.widget

    def __exit__(self, *exc):
        self.widget.blockSignals(self._was)
        return False
