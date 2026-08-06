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

__all__ = ["to_epoch_dict", "from_epoch_dict", "stage_palette"]

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
