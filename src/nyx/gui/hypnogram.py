"""nyx hypnograms in and out of ephyviewer's epoch format.

The two formats are nearly the same -- both are ``time`` / ``duration`` /
``label`` arrays -- which is most of why ephyviewer suits this at all. The one
thing that is *not* the same is what ``time`` is measured from, and getting
that wrong silently shifts an entire night.

The offset
----------

``Recording.time_slice`` keeps **absolute** times, so a trace source built from
a windowed recording starts at ``window[0]``. But ``staging.hypnogram["time"]``
starts at 0, relative to the window. Worse, ``time_slice`` returns the
recording unchanged when the window *is* the whole recording, so the offset is
``window[0]`` in one case and ``0.0`` in the other.

So the offset is never assumed: it is read off the source at runtime, and these
two functions are the only place it is applied. That makes the round trip a
plain unit test, with no Qt and no ephyviewer involved.

This module must not import Qt.
"""

from __future__ import annotations

import numpy as np

from nyx.io.annotations import merge_consecutive
from nyx.stages import COLORS, STAGE_ROW_ORDER

__all__ = [
    "to_epoch_dict", "from_epoch_dict", "stage_palette", "EpochGrid", "onto_grid",
    "merge_touching",
]

#: What an epoch with no label becomes on the way back in.
FILLER = "NOSIGNAL"


def to_epoch_dict(hypnogram: dict, name: str = "hypnogram",
                  t_offset: float = 0.0) -> dict:
    """A nyx hypnogram as an ephyviewer epoch dict, shifted into its clock."""
    return {
        "time": np.asarray(hypnogram["time"], dtype="float64") + float(t_offset),
        "duration": np.asarray(hypnogram["duration"], dtype="float64"),
        "label": np.asarray(hypnogram["label"], dtype="U16"),
        "name": name,
    }


def from_epoch_dict(
    epoch: dict,
    t_offset: float = 0.0,
    *,
    duration: float | None = None,
    filler: str = FILLER,
) -> dict:
    """An edited epoch dict back as a nyx hypnogram.

    Three things happen on the way back, each fixing something hand-editing
    does that the rest of nyx would not expect:

    * times are sorted and the offset removed;
    * adjacent segments carrying the same label are merged. The editor produces
      a lot of these, and without merging ``apply_rules`` sees boundaries that
      are not really there, and the CSV becomes unreadable;
    * gaps are filled with ``filler``. Deleting an epoch leaves a hole, and a
      hypnogram with holes does not span its window -- which ``evaluate`` and
      ``save_hypno_with_padding`` both assume it does.

    ``duration`` is the length of the analysis window; pass it so a hole at the
    very end is filled too.
    """
    times = np.asarray(epoch["time"], dtype="float64") - float(t_offset)
    durations = np.asarray(epoch["duration"], dtype="float64")
    labels = np.asarray(epoch["label"], dtype="U16")

    if len(times) == 0:
        if duration is None:
            return {"time": times, "duration": durations, "label": labels}
        return {
            "time": np.array([0.0]),
            "duration": np.array([float(duration)]),
            "label": np.array([filler], dtype="U16"),
        }

    order = np.argsort(times, kind="stable")
    times, durations, labels = times[order], durations[order], labels[order]

    filled_times: list[float] = []
    filled_durations: list[float] = []
    filled_labels: list[str] = []

    cursor = 0.0
    for start, length, label in zip(times, durations, labels, strict=True):
        start = max(float(start), 0.0)
        if start > cursor + 1e-9:
            filled_times.append(cursor)
            filled_durations.append(start - cursor)
            filled_labels.append(filler)
        filled_times.append(start)
        filled_durations.append(float(length))
        filled_labels.append(str(label))
        cursor = max(cursor, start + float(length))

    if duration is not None and cursor < float(duration) - 1e-9:
        filled_times.append(cursor)
        filled_durations.append(float(duration) - cursor)
        filled_labels.append(filler)

    return merge_consecutive({
        "time": np.asarray(filled_times, dtype="float64"),
        "duration": np.asarray(filled_durations, dtype="float64"),
        "label": np.asarray(filled_labels, dtype="U16"),
    })


class EpochGrid:
    """Fixed-length epochs, for scoring by hand: ``origin + k * length``.

    Manual scoring is done in whole epochs -- a stage per 4 s bin, say -- and
    an epoch that starts half-way through a bin, or lasts 5.3 s, is not a
    scoring anyone can compare with anything. Every boundary is put on this
    grid, and nothing can be scored outside ``[low, high]``: the whole epochs
    that fit in the analysis window.

    ``origin`` is where a grid line falls on the viewer's clock. The grid is
    anchored to the *recording*, not to the window, so trimming the window
    does not move it -- and a scoring saved from one window lines up with a
    reference scored on the whole recording.
    """

    def __init__(self, length: float, origin: float = 0.0,
                 low: float = -np.inf, high: float = np.inf):
        if not length > 0:
            raise ValueError(f"The epoch length must be positive, not {length!r}.")
        self.length = float(length)
        self.origin = float(origin) % self.length
        self.low = self.ceil(low) if np.isfinite(low) else -np.inf
        self.high = self.floor(high) if np.isfinite(high) else np.inf

    @classmethod
    def for_window(cls, length: float, window_start: float, window_duration: float,
                   t_offset: float = 0.0) -> EpochGrid:
        """The grid for a window of the recording, on a viewer's clock.

        The viewer shows the window starting at ``t_offset``; recording time
        ``a`` is at ``a - window_start + t_offset`` there, so the grid lines
        ``k * length`` of the recording land at ``origin = t_offset -
        window_start`` (modulo the length).
        """
        return cls(
            length,
            origin=float(t_offset) - float(window_start),
            low=float(t_offset),
            high=float(t_offset) + float(window_duration),
        )

    #: Times go through float arithmetic on the way here, and 3.9999999 has to
    #: count as the grid line at 4, not as the epoch before it.
    _TOLERANCE = 1e-4   # in epochs

    def _epochs(self, t) -> np.ndarray:
        return (np.asarray(t, dtype="float64") - self.origin) / self.length

    def floor(self, t):
        """The start of the epoch ``t`` is in."""
        return self.origin + np.floor(self._epochs(t) + self._TOLERANCE) * self.length

    def ceil(self, t):
        """The end of the epoch ``t`` is in (``t`` itself, on a grid line)."""
        return self.origin + np.ceil(self._epochs(t) - self._TOLERANCE) * self.length

    def nearest(self, t):
        """The grid line closest to ``t``, within the scoreable range."""
        return np.clip(self.origin + np.round(self._epochs(t)) * self.length,
                       self.low, self.high)

    def n_epochs(self) -> int:
        """How many whole epochs fit in the scoreable range."""
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError("An unbounded grid has no epoch count.")
        return max(int(round((self.high - self.low) / self.length)), 0)

    def snap(self, times, durations):
        """Put epochs on the grid: ``(times, durations)``, some maybe empty.

        Both ends go to the nearest grid line, so an epoch that is already on
        it is untouched, and two epochs sharing a boundary still share it.
        Epochs that end up with no whole epoch in them come back with zero
        duration, for the caller to drop.
        """
        times = np.asarray(times, dtype="float64")
        stops = times + np.asarray(durations, dtype="float64")
        start, stop = self.nearest(times), self.nearest(stops)
        return start, np.maximum(stop - start, 0.0)


def onto_grid(hypnogram: dict, grid: EpochGrid) -> dict:
    """An arbitrary hypnogram re-expressed in whole epochs of ``grid``.

    Each epoch takes the label that covers most of it -- a majority vote by
    duration, as :func:`nyx.metrics` does for agreement -- and epochs nothing
    covers are left unscored rather than invented. For opening a scoring made
    elsewhere, or at another epoch length, in the manual scorer.
    """
    times = np.asarray(hypnogram["time"], dtype="float64")
    durations = np.asarray(hypnogram["duration"], dtype="float64")
    labels = np.asarray(hypnogram["label"]).astype("U16")
    empty = {"time": np.array([], "float64"), "duration": np.array([], "float64"),
             "label": np.array([], "U16")}
    if len(times) == 0:
        return empty

    first = grid.low if np.isfinite(grid.low) else grid.floor(times.min())
    last = grid.high if np.isfinite(grid.high) else grid.ceil((times + durations).max())
    n = int(round((last - first) / grid.length))
    if n <= 0:
        return empty

    names = sorted(set(labels))
    cover = np.zeros((n, len(names)))
    for start, length, label in zip(times, durations, labels, strict=True):
        stop = start + length
        lo = max(int(np.floor((start - first) / grid.length)), 0)
        hi = min(int(np.ceil((stop - first) / grid.length)), n)
        column = names.index(label)
        for k in range(lo, hi):
            a = first + k * grid.length
            overlap = min(stop, a + grid.length) - max(start, a)
            if overlap > 0:
                cover[k, column] += overlap

    scored = cover.sum(axis=1) > 0
    winners = np.asarray(names, dtype="U16")[cover.argmax(axis=1)]
    k = np.nonzero(scored)[0]
    if len(k) == 0:
        return empty
    return merge_touching({
        "time": first + k * grid.length,
        "duration": np.full(len(k), grid.length),
        "label": winners[k],
    })


def merge_touching(hypnogram: dict) -> dict:
    """Merge neighbouring epochs with the same label -- only where they touch.

    :func:`~nyx.io.annotations.merge_consecutive` joins any two rows in a row
    that share a label, which is right for a hypnogram with no holes. A manual
    scoring in progress is mostly holes, and joining across one there would
    add the second epoch's length onto the first: moving it, not merging it.
    """
    times = np.asarray(hypnogram["time"], dtype="float64")
    durations = np.asarray(hypnogram["duration"], dtype="float64")
    labels = np.asarray(hypnogram["label"]).astype("U16")
    order = np.argsort(times, kind="stable")
    times, durations, labels = times[order], durations[order], labels[order]

    out_t: list[float] = []
    out_d: list[float] = []
    out_l: list[str] = []
    for start, length, label in zip(times, durations, labels, strict=True):
        if out_t and label == out_l[-1] and abs(out_t[-1] + out_d[-1] - start) < 1e-6:
            out_d[-1] = float(start + length - out_t[-1])
        else:
            out_t.append(float(start))
            out_d.append(float(length))
            out_l.append(str(label))
    return {
        "time": np.asarray(out_t, dtype="float64"),
        "duration": np.asarray(out_d, dtype="float64"),
        "label": np.asarray(out_l, dtype="U16"),
    }


def stage_palette(labels=None) -> tuple[list[str], list[str]]:
    """``(labels in display order, their colours)``.

    Colours come from :data:`nyx.stages.COLORS` and the order from
    :data:`nyx.stages.STAGE_ROW_ORDER`, so the editor, the report figures and
    the hypnogram plots all agree about what NREM looks like.
    """
    present = [str(label) for label in (labels if labels is not None else [])]
    ordered = [name for name in STAGE_ROW_ORDER if name in present or not present]
    for name in present:
        if name not in ordered:
            ordered.append(name)
    return ordered, [COLORS.get(name, "#888888") for name in ordered]
