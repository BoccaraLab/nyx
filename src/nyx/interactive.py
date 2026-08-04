"""Interactive polygon selection for refining clusters by hand.

Draw a polygon around a group of points in PC space and get back a boolean
mask, which can then be used to override the automatic cluster assignment.

This replaces the two near-identical copies that used to exist (one for a local
Qt window, one for a notebook over SSH). Both are supported by the same class:
press ``enter`` **or** ``space`` to close the polygon, so the same code works
whichever matplotlib backend is active.

Usage::

    %matplotlib qt          # or %matplotlib widget over SSH
    import matplotlib.pyplot as plt
    from nyx.interactive import PolygonSelector

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(scores[:, 0], scores[:, 1], s=5)
    selector = PolygonSelector(ax, scores[:, :2])
    plt.show()

    # ...after drawing and closing the polygon:
    mask = selector.get_mask()
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path

__all__ = ["PolygonSelector"]


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
