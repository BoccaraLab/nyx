"""A distribution with thresholds you drag, and the trace they cut.

The wake/sleep threshold is one number, but choosing it well means watching two
things at once: where it sits in the valley of the EMG power distribution, and
whether the resulting cut tracks what the animal is obviously doing. So the
same line is drawn on both, and dragging either moves both.

This is the one view nyx needs that ephyviewer does not have. It is written in
the same idiom -- pyqtgraph, a ``name``, a ``time_changed`` signal, ``seek``
-- so it docks in a :class:`~nyx.gui.panels.DockHost` beside the real viewers
and can be dragged around with them.

One deliberate difference from the version this replaces: the bars are a single
step curve rather than one ``BarGraphItem`` per bin. The old one rebuilt fifty
graphics items on every refresh, which is why it felt heavy on a night of data.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal

from nyx.gui.panels import _Dockable

__all__ = ["HistogramViewer", "ThresholdLines"]

#: How close the pointer has to be, in pixels, to pick a line up.
GRAB_PIXELS = 8


class ThresholdLines:
    """Keeps a set of thresholds on several pyqtgraph plots at once.

    Each threshold is one value drawn in as many places as you like -- vertical
    on a distribution, horizontal on a trace. Dragging any of them moves the
    rest, because they are the same number seen more than once.
    """

    def __init__(self, values, on_change=None, on_release=None, colours=None,
                 labels=None):
        self.values = [float(v) for v in values]
        self.on_change = on_change
        self.on_release = on_release
        self.colours = list(colours or ["#dc143c", "#ff8c00", "#1e90ff"])
        self.labels = list(labels or [])
        self._lines: list[list] = [[] for _ in self.values]

    def add(self, plot, orientation: str = "v", index: int | None = None):
        """Draw the thresholds on ``plot``. ``index`` adds only one of them."""
        wanted = range(len(self.values)) if index is None else [index]
        angle = 90 if orientation == "v" else 0

        for i in wanted:
            pen = pg.mkPen(self.colours[i % len(self.colours)], width=2)
            hover = pg.mkPen("#ffd700", width=3)
            line = pg.InfiniteLine(
                pos=self.values[i], angle=angle, movable=True, pen=pen,
                hoverPen=hover,
            )
            line.setCursor(Qt.SizeHorCursor if orientation == "v" else Qt.SizeVerCursor)
            if i < len(self.labels):
                line.setToolTip(self.labels[i])
            line.sigDragged.connect(lambda _l, i=i: self._dragged(i, _l))
            line.sigPositionChangeFinished.connect(
                lambda _l, i=i: self._released(i, _l)
            )
            plot.addItem(line)
            self._lines[i].append((line, orientation))
        return self

    # -- movement ----------------------------------------------------------

    def _dragged(self, index: int, line) -> None:
        self.set(index, float(line.value()), notify=False, skip=line)
        if self.on_change is not None:
            self.on_change(list(self.values))

    def _released(self, index: int, line) -> None:
        self.set(index, float(line.value()), notify=False, skip=line)
        if self.on_release is not None:
            self.on_release(list(self.values))

    def set(self, index: int, value: float, *, notify: bool = True, skip=None) -> None:
        """Move one threshold, and every line showing it."""
        self.values[index] = float(value)
        for line, _orientation in self._lines[index]:
            if line is skip:
                continue
            was = line.blockSignals(True)
            line.setValue(float(value))
            line.blockSignals(was)
        if notify and self.on_release is not None:
            self.on_release(list(self.values))

    def set_all(self, values, *, notify: bool = False) -> None:
        for index, value in enumerate(values):
            if index < len(self.values):
                self.set(index, value, notify=False)
        if notify and self.on_release is not None:
            self.on_release(list(self.values))


class HistogramViewer(_Dockable):
    """The distribution a threshold is chosen on, with the threshold on it.

    Log counts by default: the wake mode of an EMG power distribution is often
    a hundredth the height of the sleep mode, and on a linear axis the valley
    between them -- the thing you are trying to put the line in -- is invisible.
    """

    thresholds_changed = Signal(list)

    def __init__(self, name: str = "distribution", parent=None,
                 bins: int = 120, log: bool = True):
        super().__init__(name, parent)
        self.bins = int(bins)
        self._values = np.array([])

        from PySide6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graphics = pg.GraphicsLayoutWidget()
        self.plot = self.graphics.addPlot()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "value")
        self.plot.setLabel("left", "epochs")
        self.plot.setLogMode(False, log)
        layout.addWidget(self.graphics)

        # stepMode is passed with the data, not here: it requires one more x
        # than y, which an empty curve cannot satisfy.
        self.curve = pg.PlotCurveItem(
            fillLevel=0, brush=pg.mkBrush("#7f8c9a80"), pen=pg.mkPen("#5a6570")
        )
        self.plot.addItem(self.curve)
        self.thresholds: ThresholdLines | None = None

    # -- data --------------------------------------------------------------

    def set_values(self, values, label: str = "") -> None:
        self._values = np.asarray(values, dtype="float64")
        finite = self._values[np.isfinite(self._values)]
        if finite.size == 0:
            self.curve.setData([], [])
            return

        counts, edges = np.histogram(finite, bins=self.bins)
        # A log axis cannot show an empty bin, and the tails of an EMG power
        # distribution are full of them.
        self.curve.setData(edges, np.maximum(counts, 0.5), stepMode="center")
        if label:
            self.plot.setLabel("bottom", label)
        self.plot.setXRange(float(edges[0]), float(edges[-1]), padding=0.02)

    def attach(self, thresholds: ThresholdLines) -> ThresholdLines:
        """Draw a set of thresholds here, vertically."""
        self.thresholds = thresholds.add(self.plot, "v")
        return thresholds

    def auto_scale(self) -> None:
        self.plot.enableAutoRange()
