"""Scoring by hand, in whole epochs: the hypnogram editor behind ``nyx-manual``.

A manual scoring is one stage per fixed-length epoch -- 4 s, say. ephyviewer's
epoch encoder is built for free-form annotation: any start, any length, epochs
split at the cursor, boundaries typed into a table. Every one of those can put
an epoch off the grid, and an off-grid manual scoring cannot be compared
epoch for epoch with anything.

So here the grid is enforced twice:

* :class:`GridEpochEncoder` snaps what you *do* -- a stage key labels the
  epoch the cursor is in and moves to the next one; a selected range is widened
  to whole epochs -- and refuses what cannot be snapped: typing a start, a stop
  or a duration, and duplicating an epoch on top of itself.
* :class:`GridEpochSource` snaps what gets *stored*. Every change to the
  hypnogram, from any button, key or undo, goes through ``_clean_and_set``,
  and that puts both ends of every epoch on the grid. Whatever the editor
  lets through, nothing off the grid is ever kept.

This is the manual scorer only. The automatic GUI's editor
(:class:`~nyx.gui.viewers.NyxEpochEncoder`) is left free-form, since
correcting a scoring means moving boundaries wherever they belong.
"""

from __future__ import annotations

import numpy as np
from ephyviewer.epochencoder import DURATION_COL, START_COL, STOP_COL
from PySide6.QtCore import Signal

from nyx.gui.hypnogram import EpochGrid, merge_touching
from nyx.gui.review import NyxEpochSource
from nyx.gui.viewers import NyxEpochEncoder

__all__ = ["GridEpochSource", "GridEpochEncoder", "DEFAULT_LABELS"]

#: The stages, in the order the number keys pick them. The lab's manual
#: scorer used this order, so 3 is WAKE, 4 REM and 5 NREM here too.
DEFAULT_LABELS = ("NOSIGNAL", "UNCLASSIFIED", "WAKE", "REM", "NREM")


class GridEpochSource(NyxEpochSource):
    """A hypnogram whose every epoch sits on ``grid``.

    ``hypnogram`` is on the viewer's clock and should already be on the grid
    (see :func:`~nyx.gui.hypnogram.onto_grid`); anything that is not is
    snapped on the way in. ``save()`` hands back the scored epochs only --
    unscored stretches stay unscored rather than being filled.
    """

    def __init__(self, hypnogram: dict, grid: EpochGrid, *, labels, colors,
                 on_save=None, name: str = "manual"):
        self.grid = grid
        super().__init__(
            hypnogram, name=name, possible_labels=list(labels),
            color_labels=list(colors), on_save=on_save,
        )
        # Snap whatever came in, through the same path every edit takes.
        self._clean_and_set(
            np.asarray(self.ep_times, dtype="float64"),
            np.asarray(self.ep_durations, dtype="float64"),
            np.asarray(self.ep_labels),
            np.asarray(self.ep_ids),
        )

    def _clean_and_set(self, ep_times, ep_durations, ep_labels, ep_ids):
        times, durations = self.grid.snap(ep_times, ep_durations)
        super()._clean_and_set(times, durations, ep_labels, ep_ids)

    def add_epoch(self, t1, duration, label):
        # Scoring epoch after epoch with the same stage builds one bout, not a
        # row per epoch: merged as it goes, so there is no Merge to press.
        super().add_epoch(t1, duration, label)
        self.merge_neighbors()

    def split_epoch(self, ind, t_split):
        # At a grid line or not at all: the source's own check then refuses a
        # split that snapped onto either end.
        super().split_epoch(ind, float(self.grid.nearest(t_split)))

    def scored(self) -> dict:
        """The epochs scored so far, merged where they touch, viewer clock."""
        return merge_touching({
            "time": np.asarray(self.ep_times, dtype="float64"),
            "duration": np.asarray(self.ep_durations, dtype="float64"),
            "label": np.asarray(self.ep_labels, dtype="U16"),
        })

    def n_scored(self) -> int:
        """How many whole epochs have a stage."""
        return int(round(float(np.sum(self.ep_durations)) / self.grid.length))

    def save(self) -> None:
        if self.on_save is not None:
            self.on_save(self.scored())


class GridEpochEncoder(NyxEpochEncoder):
    """The epoch encoder, confined to whole epochs of ``source.grid``.

    Keys: the number of a stage labels the epoch under the cursor and steps to
    the next, as in the lab's scorer. With the range selector on, the range is
    widened to whole epochs first. Alt+arrows walk the stage changes.
    """

    #: Emitted after every change to the hypnogram, undo and redo included.
    edited = Signal()

    def __init__(self, source: GridEpochSource, **kargs):
        grid = source.grid
        super().__init__(source=source, rules=[], epoch_length=grid.length, **kargs)
        self.grid = grid

        # One stage per epoch: overlapping epochs are not a scoring. The
        # epoch step is the grid, and changing it here would leave the grid.
        self.params["exclusive_mode"] = True
        self.params["new_epoch_step"] = grid.length
        for name in ("exclusive_mode", "new_epoch_step"):
            self.params.param(name).setReadonly(True)
        action = getattr(self, "allow_overlap_action", None)
        if action is not None:
            action.setChecked(False)
            action.setEnabled(False)
            action.setToolTip("Manual scoring is one stage per epoch.")

        # The only navigation that makes sense is between stage changes: there
        # are no postprocessing rules in a scoring done by hand.
        for action in self.toolbar.actions():
            if action.text() in ("< To check", "To check >"):
                action.setVisible(False)

        self.t = float(max(grid.low, 0.0)) if np.isfinite(grid.low) else 0.0

    # -- writing ----------------------------------------------------------

    def on_label_shortcut(self, label, modifier_used):
        if self.range_group_box.isChecked():
            self._snap_region()
        else:
            # The epoch the cursor is in -- not one starting at the cursor.
            self.t = float(self.grid.floor(self.t))
            if not self.grid.low - 1e-6 <= self.t < self.grid.high - 1e-6:
                return  # outside the window: nothing there to score
        # Never the overlap mode, whatever the modifier.
        super().on_label_shortcut(label, False)

    def apply_region(self):
        self._snap_region()
        super().apply_region()

    def delete_region(self):
        self._snap_region()
        super().delete_region()

    def _snap_region(self) -> None:
        """Widen the range selector to the whole epochs it touches."""
        low, high = self.region.getRegion()
        low = max(float(self.grid.floor(low)), self.grid.low)
        high = min(float(self.grid.ceil(high)), self.grid.high)
        if high - low < self.grid.length - 1e-6:
            high = low + self.grid.length
        self.region.setRegion((low, high))
        self.spin_limit1.setValue(low)
        self.spin_limit2.setValue(high)

    def on_table_cell_change(self, row, col):
        if col in (START_COL, STOP_COL, DURATION_COL):
            # Typed boundaries cannot be on the grid by construction, so they
            # are not taken at all: put the table back as it was.
            self.refresh_table()
            return
        super().on_table_cell_change(row, col)

    def append_history(self):
        super().append_history()
        self.edited.emit()

    def on_undo(self):
        super().on_undo()
        self.edited.emit()

    def on_redo(self):
        super().on_redo()
        self.edited.emit()

    def on_change_label(self, ind, new_label):
        """Relabel from the table, then merge it into same-stage neighbours."""
        self.source.ep_labels[ind] = new_label
        self.source.merge_neighbors()
        self.append_history()
        self.refresh()
        self.refresh_table()

    def duplicate_selected_epoch(self, ind=None):
        """Refused: a copy of an epoch on top of itself is two stages at once."""

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        # Saving is the tab's business, and it asks itself; ephyviewer's own
        # "save before closing?" prompt would write nowhere useful.
        self.changes_since_save = 0
        super().closeEvent(event)
