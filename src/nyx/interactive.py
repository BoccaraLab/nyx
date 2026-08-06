"""Deciding things by hand, on a figure, wherever matplotlib is drawing.

Two decisions in the pipeline are easier made by pointing than by typing:
*which points are this cluster*, and *where does this threshold go*. Both are
here, both are pure matplotlib, and both therefore work identically in a Qt
window, in a notebook over SSH, and inside :mod:`nyx.gui` -- there is no Qt
import in this module and there should never be one.

:class:`PolygonSelector` -- draw round a group of points in PC space and get a
boolean mask back, to override the automatic cluster assignment::

    %matplotlib qt          # or %matplotlib widget over SSH
    import matplotlib.pyplot as plt
    from nyx.interactive import PolygonSelector

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(scores[:, 0], scores[:, 1], s=5)
    selector = PolygonSelector(ax, scores[:, :2])
    plt.show()

    # ...after drawing and closing the polygon:
    mask = selector.get_mask()

:class:`ThresholdSelector` -- drag a cut, shown on a distribution and on a time
course at once, instead of guessing a number and re-running the cell::

    import nyx
    from nyx.interactive import ThresholdSelector

    emg = nyx.compute_emg_features(recording, params)
    fig = nyx.plot_emg_check(emg, threshold=nyx.find_wake_sleep_threshold(emg))

    def rescore(value):
        wake_sleep = nyx.classify_wake_sleep(emg, threshold=value)
        print(f"{value:.3f}: {100 * wake_sleep.hypnogram['label'].tolist().count('WAKE')
                              / len(wake_sleep.hypnogram['label']):.1f}% wake")

    cut = ThresholdSelector(0.55, hist_ax=fig.axes[0], trace_ax=fig.axes[1],
                            on_release=rescore)
    plt.show()

    EMG_THRESHOLD = cut.value
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path

__all__ = ["PolygonSelector", "ThresholdSelector"]


class PolygonSelector:
    """Select points inside a hand-drawn polygon.

    Controls
    --------
    left click
        Add a vertex.
    right click
        Remove the last vertex.
    ``enter`` or ``space``
        Close the polygon and finish the selection.
    ``escape``
        Discard the polygon and start over.

    Parameters
    ----------
    ax
        Axes the points are plotted on.
    points
        ``(N, 2)`` array of the plotted coordinates, in the same order as the
        epochs the mask will be applied to.
    verbose
        Print each vertex as it is added.
    """

    def __init__(self, ax, points, verbose: bool = True):
        self.ax = ax
        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2); got {self.points.shape}. "
                f"Pass the two components you are plotting, e.g. scores[:, :2]."
            )

        self.polygon_points: list[list[float]] = []
        self.finished = False
        self.verbose = verbose

        self._lines: list = []
        self._dots: list = []
        self._patch = None

        canvas = ax.figure.canvas
        self._cid_click = canvas.mpl_connect("button_press_event", self._on_click)
        self._cid_key = canvas.mpl_connect("key_press_event", self._on_key)

    # -- event handling ----------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self.ax or self.finished:
            return

        if event.button == 1:  # add a vertex
            self.polygon_points.append([event.xdata, event.ydata])
            self._dots.append(self.ax.plot(event.xdata, event.ydata, "ro", markersize=8)[0])
            if len(self.polygon_points) > 1:
                previous, current = self.polygon_points[-2], self.polygon_points[-1]
                self._lines.append(
                    self.ax.plot(
                        [previous[0], current[0]],
                        [previous[1], current[1]],
                        "r-",
                        linewidth=2,
                    )[0]
                )
            self._redraw()
            if self.verbose:
                print(
                    f"Point {len(self.polygon_points)} added: "
                    f"({event.xdata:.3f}, {event.ydata:.3f})"
                )

        elif event.button == 3:  # remove the last vertex
            if self.polygon_points:
                self.polygon_points.pop()
                if self._dots:
                    self._dots.pop().remove()
                if self._lines:
                    self._lines.pop().remove()
                self._redraw()
                if self.verbose:
                    print(f"Last point removed. {len(self.polygon_points)} remaining.")

    def _on_key(self, event):
        if self.finished:
            return
        if event.key in ("enter", " "):
            self.finish()
        elif event.key == "escape":
            self.clear()
            if self.verbose:
                print("Selection cancelled. Starting over...")

    def _redraw(self):
        # draw_idle works on both interactive Qt windows and notebook backends.
        self.ax.figure.canvas.draw_idle()

    # -- public API --------------------------------------------------------

    def finish(self, *_args):
        """Close the polygon and stop listening for further clicks."""
        if self.finished or len(self.polygon_points) < 3:
            if not self.finished and self.verbose:
                print(
                    f"Need at least 3 points to close a polygon "
                    f"(have {len(self.polygon_points)})."
                )
            return

        first, last = self.polygon_points[0], self.polygon_points[-1]
        self._lines.append(
            self.ax.plot([last[0], first[0]], [last[1], first[1]], "r-", linewidth=2)[0]
        )
        self._patch = MplPolygon(
            self.polygon_points, alpha=0.3, facecolor="red", edgecolor="red", linewidth=2
        )
        self.ax.add_patch(self._patch)
        self._redraw()
        self.finished = True

        canvas = self.ax.figure.canvas
        canvas.mpl_disconnect(self._cid_click)
        canvas.mpl_disconnect(self._cid_key)

        if self.verbose:
            n_selected = int(self.get_mask().sum())
            print(
                f"Selection complete: {len(self.polygon_points)} vertices, "
                f"{n_selected} of {len(self.points)} points inside "
                f"({100 * n_selected / max(len(self.points), 1):.1f}%). "
                f"You can close the window now."
            )

    def clear(self):
        """Remove the polygon drawn so far and reset the selection."""
        for artist in (*self._lines, *self._dots):
            artist.remove()
        if self._patch is not None:
            self._patch.remove()
        self._lines, self._dots, self._patch = [], [], None
        self.polygon_points = []
        self._redraw()

    def get_mask(self) -> np.ndarray:
        """Boolean mask, ``True`` for points inside the polygon.

        Returns an all-``False`` mask if fewer than three vertices were placed,
        so an abandoned selection is simply ignored.
        """
        if len(self.polygon_points) < 3:
            return np.zeros(len(self.points), dtype=bool)
        return Path(self.polygon_points).contains_points(self.points)


class ThresholdSelector:
    """Drag a threshold line, shown on a distribution and on a time course at once.

    The two views are the two halves of :func:`nyx.plot_emg_check`: a vertical
    line on the histogram and a horizontal line on the power trace, both at the
    same value. Dragging either moves both, because they are the same number
    seen twice -- and seeing it twice is the point, since a cut that sits
    nicely in the valley of a histogram can still be obviously wrong against
    the trace.

    Nothing here knows about EMG. It is *a value, a distribution, a time
    course*, which is equally the shape of a :class:`~nyx.steps.Refinement`
    threshold on a PC score.

    Controls
    --------
    click and drag either line
        Move the threshold.
    up / down arrow
        Nudge by one ``step``. Hold shift for ten.
    ``r``
        Back to ``initial``.

    Parameters
    ----------
    initial
        Starting value. :meth:`reset` returns here.
    hist_ax
        Axes carrying the distribution; gets a vertical line. May be ``None``.
    trace_ax
        Axes carrying the value against time; gets a horizontal line. May be
        ``None``. At least one of the two is required.
    on_change
        Called with the new value on every mouse-move during a drag. Keep it
        cheap -- update a label, not a hypnogram.
    on_release
        Called with the final value when the mouse is let go, and after a
        keyboard nudge. This is where the work goes:
        :func:`nyx.classify_wake_sleep` and a redraw.
    limits
        ``(low, high)`` to clamp to. Defaults to the histogram's x-limits, or
        the trace's y-limits.
    step
        Keyboard nudge size. Defaults to a two-hundredth of the range.
    tolerance
        How close the pointer must be to grab a line, **in pixels** -- the same
        number of data units is a different distance on a 0-1 histogram than on
        a raw power trace, so a data-unit tolerance would feel wrong on one of
        the two.
    """

    def __init__(
        self,
        initial: float,
        *,
        hist_ax=None,
        trace_ax=None,
        on_change: Callable[[float], None] | None = None,
        on_release: Callable[[float], None] | None = None,
        limits: Sequence[float] | None = None,
        color: str = "crimson",
        linestyle: str = "--",
        linewidth: float = 2.0,
        label: str = "threshold",
        tolerance: float = 6.0,
        step: float | None = None,
        verbose: bool = False,
    ):
        if hist_ax is None and trace_ax is None:
            raise ValueError(
                "ThresholdSelector needs at least one axes to draw on: pass "
                "hist_ax (a distribution), trace_ax (a time course), or both."
            )

        self.initial = float(initial)
        self._value = float(initial)
        self.on_change = on_change
        self.on_release = on_release
        self.tolerance = float(tolerance)
        self.verbose = verbose
        self.label = label

        # (axes, line, orientation). Everything that moves the threshold walks
        # this list, so a third view costs nothing -- see add_axis.
        self._lines: list[tuple] = []
        self._style = dict(color=color, linestyle=linestyle, linewidth=linewidth)

        if limits is not None:
            self.limits = (float(limits[0]), float(limits[1]))
        elif hist_ax is not None:
            self.limits = tuple(float(v) for v in hist_ax.get_xlim())
        else:
            self.limits = tuple(float(v) for v in trace_ax.get_ylim())

        span = abs(self.limits[1] - self.limits[0]) or 1.0
        self.step = float(step) if step is not None else span / 200.0

        if hist_ax is not None:
            self.add_axis(hist_ax, "v")
        if trace_ax is not None:
            self.add_axis(trace_ax, "h")

        self._dragging = False
        self._background = None
        self._cids: list[tuple] = []
        self._connect()

    # -- views -------------------------------------------------------------

    def add_axis(self, ax, orientation: str = "v"):
        """Show the threshold on another axes too.

        ``orientation`` is ``"v"`` for a vertical line (the value is read off
        the x axis) or ``"h"`` for a horizontal one.
        """
        if orientation not in ("v", "h"):
            raise ValueError(f"orientation must be 'v' or 'h'; got {orientation!r}.")

        draw = ax.axvline if orientation == "v" else ax.axhline
        line = draw(self._value, **self._style)
        self._lines.append((ax, line, orientation))
        return line

    # -- the value ---------------------------------------------------------

    @property
    def value(self) -> float:
        """The threshold. Setting it moves every line."""
        return self._value

    @value.setter
    def value(self, new: float) -> None:
        self.set_value(new)

    def set_value(self, value: float, *, notify: bool = True) -> None:
        """Move the threshold to ``value``, clamped to ``limits``.

        ``notify=False`` moves the lines without calling ``on_change`` or
        ``on_release`` -- for pushing a value in from a spin box that is
        already reacting to its own edit, where notifying would loop.
        """
        self._move(value)
        self._draw()
        if notify:
            self._fire(self.on_change)
            self._fire(self.on_release)

    def reset(self) -> None:
        """Back to the value the selector was created with."""
        self.set_value(self.initial)

    def disconnect(self) -> None:
        """Stop listening. The lines stay where they are."""
        for canvas, ids in self._cids:
            for cid in ids:
                canvas.mpl_disconnect(cid)
        self._cids = []

    # -- event handling ----------------------------------------------------

    def _connect(self) -> None:
        for ax, _line, _orientation in self._lines:
            canvas = ax.figure.canvas
            if any(existing is canvas for existing, _ in self._cids):
                continue  # both panels usually share one figure
            self._cids.append((canvas, [
                canvas.mpl_connect("button_press_event", self._on_press),
                canvas.mpl_connect("motion_notify_event", self._on_motion),
                canvas.mpl_connect("button_release_event", self._on_release),
                canvas.mpl_connect("key_press_event", self._on_key),
            ]))

    def _near(self, event) -> bool:
        """Whether the pointer is within ``tolerance`` pixels of a line."""
        for ax, _line, orientation in self._lines:
            if event.inaxes is not ax:
                continue
            if orientation == "v":
                at = ax.transData.transform((self._value, 0))[0]
                return abs(event.x - at) <= self.tolerance
            at = ax.transData.transform((0, self._value))[1]
            return abs(event.y - at) <= self.tolerance
        return False

    def _axis_value(self, event) -> float | None:
        """The pointer position, read off whichever axis carries the value."""
        for ax, _line, orientation in self._lines:
            if event.inaxes is ax:
                return event.xdata if orientation == "v" else event.ydata
        return None

    def _on_press(self, event) -> None:
        if event.button != 1 or not self._near(event):
            return
        self._dragging = True
        self._grab_background()

    def _on_motion(self, event) -> None:
        if not self._dragging:
            return
        value = self._axis_value(event)
        if value is None:
            return
        self._move(value)
        self._blit()
        self._fire(self.on_change)

    def _on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._background = None
        for _ax, line, _orientation in self._lines:
            line.set_animated(False)
        self._draw()
        self._fire(self.on_release)
        if self.verbose:
            print(f"{self.label} = {self._value:.4f}")

    def _on_key(self, event) -> None:
        if event.key in ("up", "down", "shift+up", "shift+down"):
            size = self.step * (10 if event.key.startswith("shift") else 1)
            self.set_value(self._value + (size if event.key.endswith("up") else -size))
        elif event.key == "r":
            self.reset()

    # -- drawing -----------------------------------------------------------

    def _move(self, value: float) -> None:
        """Clamp and place the lines, without redrawing or notifying."""
        low, high = min(self.limits), max(self.limits)
        self._value = float(np.clip(float(value), low, high))
        for _ax, line, orientation in self._lines:
            if orientation == "v":
                line.set_xdata([self._value, self._value])
            else:
                line.set_ydata([self._value, self._value])

    def _grab_background(self) -> None:
        """Cache the figure without the lines, so a drag only redraws them.

        Without this, dragging over a night of data redraws a 20 000-point
        trace on every mouse-move. The notebook backends cannot blit, so this
        quietly gives up and :meth:`_blit` falls back to a full redraw.
        """
        canvases = []
        for ax, _line, _orientation in self._lines:
            canvas = ax.figure.canvas
            if not getattr(canvas, "supports_blit", False):
                self._background = None
                return
            if not any(existing is canvas for existing in canvases):
                canvases.append(canvas)

        # Animated artists are left out of a saved background, so hide the
        # lines, draw, and only then copy.
        for _ax, line, _orientation in self._lines:
            line.set_animated(True)
        for canvas in canvases:
            canvas.draw()
        self._background = [
            ax.figure.canvas.copy_from_bbox(ax.bbox)
            for ax, _line, _orientation in self._lines
        ]

    def _blit(self) -> None:
        if self._background is None:
            self._draw()
            return
        for (ax, line, _orientation), background in zip(
            self._lines, self._background, strict=True
        ):
            canvas = ax.figure.canvas
            canvas.restore_region(background)
            ax.draw_artist(line)
            canvas.blit(ax.bbox)

    def _draw(self) -> None:
        drawn = []
        for ax, _line, _orientation in self._lines:
            canvas = ax.figure.canvas
            if not any(existing is canvas for existing in drawn):
                drawn.append(canvas)
                canvas.draw_idle()

    def _fire(self, callback) -> None:
        if callback is not None:
            callback(self._value)
