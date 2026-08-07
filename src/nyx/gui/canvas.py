"""Matplotlib inside Qt, without letting pyplot near a window.

nyx's figures split cleanly in two, and the split decides how each is hosted:

* nine panels take ``ax`` as their second positional argument and draw into it
  -- ``plot_emg_threshold``, ``plot_clusters``, ``plot_psd_per_cluster`` and so
  on. :class:`PanelCanvas` draws those, bypassing pyplot entirely.
* six composites build their own multi-panel figure and return it --
  ``plot_emg_check``, ``plot_cluster_check``, ``plot_scoring_overview``.
  :class:`FigureView` adopts one of those.

Three rules keep this safe, and they are rules rather than habits:

1. **A job never returns a figure.** Figures are built on the main thread,
   from data a job returned. ``jobs.py`` imports no matplotlib at all.
2. **The backend is Agg**, set here at import, exactly as ``tests/conftest.py``
   does. Constructing a ``FigureCanvasQTAgg`` explicitly still works and still
   needs a ``QApplication``; what Agg forbids is pyplot opening a window of its
   own.
3. **``plt.show()`` is never called.** Anywhere. It would block the event loop
   or open a second, unmanaged window.

The composites do go through ``plt.subplots``, so their figures land in
pyplot's registry and leak unless closed. :class:`FigureView` closes the one it
is replacing, and closes its own on teardown.
"""

from __future__ import annotations

import matplotlib

# Before any pyplot import. Same call, and same reason, as tests/conftest.py.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget  # noqa: E402

__all__ = ["PanelCanvas", "FigureView", "DARK", "darken"]

#: ephyviewer draws on black, and a white matplotlib panel docked beside one is
#: jarring enough to be distracting. These are applied to the *figure*, not to
#: rcParams, so nothing nyx writes to disk changes -- ``save_report`` keeps
#: producing figures on white, which is what belongs in a paper.
DARK = {
    "figure": "#1b1b1b",
    "axes": "#1b1b1b",
    "ink": "#d8d8d8",
    "grid": "#3a3a3a",
}


def darken(figure) -> None:
    """Recolour a finished figure to sit beside the ephyviewer panels.

    Deliberately applied after the fact rather than through a style context:
    the report functions choose their own colours for stages and clusters, and
    those must survive -- only the furniture changes.
    """
    figure.patch.set_facecolor(DARK["figure"])
    for ax in figure.axes:
        ax.set_facecolor(DARK["axes"])
        for spine in ax.spines.values():
            spine.set_color(DARK["grid"])
        ax.tick_params(colors=DARK["ink"], which="both")
        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            item.set_color(DARK["ink"])
        for text in ax.texts:
            if text.get_color() in ("black", "k", "#000000"):
                text.set_color(DARK["ink"])
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(DARK["axes"])
            legend.get_frame().set_edgecolor(DARK["grid"])
            for text in legend.get_texts():
                text.set_color(DARK["ink"])
    for ax in figure.axes:
        if hasattr(ax, "get_yaxis") and ax.get_label() == "<colorbar>":
            ax.tick_params(colors=DARK["ink"])


class PanelCanvas(FigureCanvasQTAgg):
    """A canvas holding one nyx panel.

    ``draw_panel(plot_clusters, clusters, stages=...)`` is the whole API. The
    call signature it relies on -- ``panel(piece, ax, **kwargs)`` -- is the one
    ``tests/test_report.py::test_panels_accept_ax_positionally`` already pins,
    so that test is load-bearing for the GUI as well as for ``save_report``.
    """

    def __init__(self, parent=None, figsize=(6.0, 4.0), dpi=100, dark: bool = True):
        figure = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        super().__init__(figure)
        self.dark = dark
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ax = None

    @property
    def ax(self):
        """The single axes, created on first use."""
        if self._ax is None:
            self._ax = self.figure.add_subplot(111)
        return self._ax

    def clear(self) -> None:
        self.figure.clear()
        self._ax = None
        self.draw_idle()

    def draw_panel(self, panel, piece, **kwargs):
        """Clear and redraw with one of nyx's ``ax``-taking panels.

        A panel that raises is reported on the axes rather than propagated: one
        broken figure should not take down the tab it lives on, the same way
        ``save_report`` keeps a run when a panel fails.
        """
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        try:
            panel(piece, self._ax, **kwargs)
        except Exception as exc:  # noqa: BLE001 - drawn, not raised
            self._ax.clear()
            self._ax.text(
                0.5, 0.5,
                f"Could not draw {getattr(panel, '__name__', 'panel')}:\n"
                f"{type(exc).__name__}: {exc}",
                ha="center", va="center", wrap=True, fontsize=9, color="crimson",
                transform=self._ax.transAxes,
            )
            self._ax.set_axis_off()
        if self.dark:
            darken(self.figure)
        self.draw_idle()
        return self._ax


class FigureView(QWidget):
    """Hosts a figure built elsewhere, with a navigation toolbar.

    For the composites, which lay out their own panels. Adopting the figure
    they return keeps their layout -- ``plot_emg_check``'s 1:2 width ratio, for
    instance -- in one place instead of duplicating it here.
    """

    def __init__(self, parent=None, toolbar: bool = True, placeholder: str = "",
                 dark: bool = True):
        super().__init__(parent)
        self.dark = dark
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._wants_toolbar = toolbar

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel(placeholder or "Nothing to show yet.")
        self._placeholder.setAlignment(
            self._placeholder.alignment().__class__.AlignCenter
        )
        self._placeholder.setStyleSheet("color: palette(mid);")
        self._layout.addWidget(self._placeholder)

    # -- content -----------------------------------------------------------

    @property
    def figure(self):
        return self._figure

    @property
    def canvas(self):
        return self._canvas

    def set_figure(self, figure) -> None:
        """Show ``figure``, closing whatever was here before."""
        self._release()
        self._placeholder.hide()

        if self.dark:
            darken(figure)
        self._figure = figure
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self._wants_toolbar:
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw_idle()

    def show_message(self, text: str) -> None:
        """Drop the figure and say why there isn't one."""
        self._release()
        self._placeholder.setText(text)
        self._placeholder.show()

    def _release(self) -> None:
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._toolbar = None
        self._canvas = None
        if self._figure is not None:
            # The composites build their figures through pyplot, so they stay
            # in its registry until closed -- twenty tabs later that is twenty
            # leaked figures and a warning about it.
            plt.close(self._figure)
            self._figure = None

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        self._release()
        super().closeEvent(event)
