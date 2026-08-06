"""What every tab is.

A tab owns one decision, declares the :class:`~nyx.gui.session.Stage` it
depends on, and implements three things:

``build()``
    Make the widgets. Called once.
``refresh()``
    Redraw from the session. Called whenever anything it depends on changes.
    Must be cheap and must never compute -- the session's accessors raise
    rather than compute, which is what keeps that honest.
``run()``
    Ask for the work. Optional: a tab whose stage is instant has no run button
    and recomputes on every edit instead.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nyx.gui.session import NotComputed, ScoringSession, Stage
from nyx.gui.widgets import WarningBanner

__all__ = ["Tab", "controls_column"]


class Tab(QWidget):
    """Base class for the tabs in the side rail."""

    #: Shown in the rail.
    title = "Tab"
    #: One line under the heading, saying what the decision is.
    subtitle = ""
    #: The stage this tab produces. The badge reads it.
    stage: Stage = Stage.LOAD
    #: Whether the work is slow enough to need a worker and a button.
    needs_worker = True
    #: Label for that button.
    run_label = "Run"

    #: Ask the window to run ``compute_through(stage)`` on the worker.
    run_requested = Signal(object)
    #: Ask the window to say something in the status bar.
    status = Signal(str)

    def __init__(self, session: ScoringSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._building = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        heading = QLabel(f"<h2 style='margin:0'>{self.title}</h2>")
        outer.addWidget(heading)
        if self.subtitle:
            note = QLabel(self.subtitle)
            note.setWordWrap(True)
            note.setStyleSheet("color: palette(mid);")
            outer.addWidget(note)

        self.warnings = WarningBanner()
        outer.addWidget(self.warnings)

        self.body = QWidget()
        outer.addWidget(self.body, 1)

        self.footer = QHBoxLayout()
        self.footer.addStretch(1)
        self.run_button = QPushButton(self.run_label)
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.stage))
        if not self.needs_worker:
            self.run_button.hide()
        self.footer.addWidget(self.run_button)
        outer.addLayout(self.footer)

        self.build()

    # -- to implement ------------------------------------------------------

    def build(self) -> None:
        """Create the widgets. Called once, from ``__init__``."""

    def refresh(self) -> None:
        """Redraw from the session. Cheap; never computes."""

    def apply(self) -> None:
        """Push widget values into the session. Called before running."""

    # -- helpers -----------------------------------------------------------

    def safe_refresh(self) -> None:
        """:meth:`refresh`, tolerating a session that has not got there yet.

        Tabs are reachable before their inputs exist, so ``NotComputed`` is a
        normal state here rather than an error.
        """
        if self._building:
            return
        self._building = True
        try:
            self.refresh()
        except NotComputed:
            pass
        finally:
            self._building = False

    def quiet(self, widget):
        """Context manager blocking a widget's signals while it is set."""
        return _Quiet(widget)

    def show_warnings(self, messages) -> None:
        self.warnings.show_messages(messages)


class _Quiet:
    def __init__(self, widget):
        self.widget = widget

    def __enter__(self):
        self._was = self.widget.blockSignals(True)
        return self.widget

    def __exit__(self, *exc):
        self.widget.blockSignals(self._was)
        return False


def controls_column(*widgets, stretch: bool = True, width: int = 320) -> QScrollArea:
    """A scrollable left-hand column of controls."""
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(0, 0, 8, 0)
    for widget in widgets:
        if isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
    if stretch:
        layout.addStretch(1)

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setMinimumWidth(width)
    area.setMaximumWidth(width + 120)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area
