"""Dockable panels, and the dock area they live in.

Every tab that shows anything is a :class:`DockHost` -- ephyviewer's
``MainViewer``, embedded rather than free-floating. That buys three things at
once, and they are the reason for using ephyviewer at all:

* **the views scroll together.** ``MainViewer`` locks its viewers to one clock,
  so dragging the EEG drags the EMG power, the PC scores and the hypnogram with
  it;
* **the navigation toolbar** -- play, seek, time width, per-view auto-scale;
* **the panels can be rearranged by dragging.** Nesting is enabled, so two
  panels can sit side by side, be tabbed together, or be torn off into their
  own window, and the layout is whatever you drag it into.

Not everything worth showing is a time series. A cluster scatter and a
per-cluster spectrum are not, and matplotlib draws them better than pyqtgraph
would. :class:`MplPanel` wraps a matplotlib figure so it satisfies the small
interface ``MainViewer`` expects -- ``name``, ``time_changed``, ``seek`` --
and so shares the same dock area, dragged and tabbed like everything else.
"""

from __future__ import annotations

from ephyviewer import MainViewer
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from nyx.gui.canvas import FigureView, PanelCanvas

__all__ = ["DockHost", "MplPanel", "TextPanel"]


class DockHost(MainViewer):
    """An embeddable dock area with ephyviewer's navigation and time locking.

    Subclassed for two reasons: to shut its viewers down deterministically
    (they own worker threads that only stop on ``closeEvent``, and a host that
    is merely garbage collected takes the process down), and to make replacing
    the contents cheap, since a tab rebuilds them whenever the data changes.
    """

    def __init__(self, parent=None, **kwargs):
        kwargs.setdefault("show_auto_scale", True)
        kwargs.setdefault("global_xsize_zoom", True)
        super().__init__(parent=parent, **kwargs)
        self.setDockNestingEnabled(True)
        # Embedded, so it must behave as a widget rather than a window.
        self.setWindowFlags(Qt.Widget)

    # -- contents ----------------------------------------------------------

    def add(self, widget, **kwargs):
        """``add_view``, tolerant of a repeated name and of panel preferences."""
        if widget.name in self.viewers:
            self.remove(widget.name)
        preferred = getattr(widget, "default_location", None)
        if preferred and not {"location", "tabify_with", "split_with"} & set(kwargs):
            kwargs["location"] = preferred
        self.add_view(widget, **kwargs)
        # A panel that was removed had close() called on it, which hides it as
        # well as stopping its threads. Re-adding the same object has to undo
        # that or the dock comes back empty.
        widget.show()
        return widget

    def remove(self, name: str) -> None:
        entry = self.viewers.pop(name, None)
        if entry is None:
            return
        entry["widget"].close()
        self.removeDockWidget(entry["dock"])
        entry["dock"].setParent(None)
        entry["dock"].deleteLater()

    def clear(self) -> None:
        """Close every panel. Called before rebuilding from new data."""
        for name in list(self.viewers):
            self.remove(name)

    def panel(self, name: str):
        entry = self.viewers.get(name)
        return entry["widget"] if entry else None

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        try:
            super().closeEvent(event)
        except Exception:  # noqa: BLE001 - settings may be unwritable
            event.accept()

    def __del__(self):
        try:
            self.clear()
        except Exception:  # noqa: BLE001 - Qt may already be gone
            pass


class _Dockable(QWidget):
    """The interface ``MainViewer`` needs from anything it docks."""

    time_changed = Signal(float)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.source = None
        self.t = 0.0

    def seek(self, t: float) -> None:
        """Called when another view scrolls. Panels that are not time series
        have nothing to do, but must not refuse the call."""
        self.t = float(t)

    def auto_scale(self) -> None:
        pass

    def set_settings(self, value) -> None:
        pass

    def get_settings(self):
        return None


class MplPanel(_Dockable):
    """A matplotlib figure that docks alongside the ephyviewer panels.

    For the things that are genuinely not time series -- the cluster scatter,
    the per-cluster spectra, the confusion matrix. Everything that *is* one
    should be a real viewer instead, so it scrolls with the rest.
    """

    def __init__(self, name: str, parent=None, toolbar: bool = True,
                 placeholder: str = ""):
        super().__init__(name, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = FigureView(toolbar=toolbar, placeholder=placeholder)
        layout.addWidget(self.view)

    def set_figure(self, figure) -> None:
        self.view.set_figure(figure)

    def show_message(self, text: str) -> None:
        self.view.show_message(text)

    def closeEvent(self, event):  # noqa: N802
        self.view.close()
        super().closeEvent(event)


class CanvasPanel(_Dockable):
    """A dockable single-panel canvas, for the ``ax``-taking report functions."""

    def __init__(self, name: str, parent=None, figsize=(6.0, 4.0)):
        super().__init__(name, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PanelCanvas(figsize=figsize)
        layout.addWidget(self.canvas)

    def draw_panel(self, panel, piece, **kwargs):
        return self.canvas.draw_panel(panel, piece, **kwargs)

    def clear(self) -> None:
        self.canvas.clear()


class TextPanel(_Dockable):
    """A dockable read-only text pane, for summaries and agreement tables.

    These belong along the bottom: they are read once and are wide rather than
    tall, and docked anywhere else they take width the traces need.
    """

    #: Where :meth:`DockHost.add` puts one unless told otherwise.
    default_location = "bottom"

    def __init__(self, name: str, parent=None, placeholder: str = ""):
        from PySide6.QtWidgets import QPlainTextEdit

        from nyx.gui.widgets import monospace

        super().__init__(name, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(monospace())
        self.text.setPlaceholderText(placeholder)
        layout.addWidget(self.text)

    def set_text(self, value: str) -> None:
        self.text.setPlainText(str(value))
