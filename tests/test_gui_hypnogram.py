"""Hypnograms in and out of the epoch editor's format.

Qt-free and ephyviewer-free on purpose: the offset between the two clocks is
the whole bug class here, and it is worth being able to test it without a
display.
"""

from __future__ import annotations

import numpy as np
import pytest

from nyx.gui.hypnogram import from_epoch_dict, stage_palette, to_epoch_dict
from nyx.io.annotations import merge_consecutive
from nyx.stages import COLORS


def hyp(*bouts) -> dict:
    labels = [label for label, _ in bouts]
    durations = [float(seconds) for _, seconds in bouts]
    return {
        "time": np.cumsum([0.0] + durations[:-1]).astype("float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U16"),
    }


def bouts(hypnogram):
    return [
        (str(label), round(float(duration), 6))
        for label, duration in zip(
            hypnogram["label"], hypnogram["duration"], strict=True
        )
    ]


# ---------------------------------------------------------------------------
# The offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1234.5])
def test_the_round_trip_is_lossless(offset):
    hypnogram = hyp(("WAKE", 60), ("NREM", 120), ("REM", 30))

    there = to_epoch_dict(hypnogram, "nyx", offset)
    back = from_epoch_dict(there, offset)

    assert bouts(back) == bouts(merge_consecutive(hypnogram))
    assert np.allclose(back["time"], hypnogram["time"])


def test_going_out_shifts_into_the_viewer_clock():
    hypnogram = hyp(("WAKE", 60), ("NREM", 60))

    epoch = to_epoch_dict(hypnogram, "nyx", 1000.0)

    # The trace source starts at window[0]; the hypnogram starts at 0. Without
    # the shift the whole night would sit in the wrong place.
    assert list(epoch["time"]) == [1000.0, 1060.0]
    assert epoch["name"] == "nyx"


def test_coming_back_removes_it_again():
    epoch = {
        "time": np.array([1000.0, 1060.0]),
        "duration": np.array([60.0, 60.0]),
        "label": np.array(["WAKE", "NREM"], dtype="U16"),
    }

    back = from_epoch_dict(epoch, 1000.0)

    assert list(back["time"]) == [0.0, 60.0]
    assert "name" not in back


# ---------------------------------------------------------------------------
# What hand-editing does
# ---------------------------------------------------------------------------


def test_adjacent_segments_with_the_same_label_are_merged():
    # Editing produces a lot of these. Left alone, apply_rules' neighbour
    # logic sees boundaries that are not really there.
    epoch = to_epoch_dict(hyp(("NREM", 60), ("NREM", 60), ("REM", 30)))

    back = from_epoch_dict(epoch)

    assert bouts(back) == [("NREM", 120.0), ("REM", 30.0)]


def test_a_deleted_epoch_leaves_no_hole():
    epoch = {
        "time": np.array([0.0, 120.0]),
        "duration": np.array([60.0, 60.0]),
        "label": np.array(["WAKE", "REM"], dtype="U16"),
    }

    back = from_epoch_dict(epoch)

    # evaluate and save_hypno_with_padding both assume the hypnogram spans its
    # window, so the gap has to become something rather than nothing.
    assert bouts(back) == [("WAKE", 60.0), ("NOSIGNAL", 60.0), ("REM", 60.0)]


def test_a_hole_at_the_end_is_filled_when_the_window_is_known():
    epoch = to_epoch_dict(hyp(("WAKE", 60)))

    back = from_epoch_dict(epoch, duration=200.0)

    assert bouts(back) == [("WAKE", 60.0), ("NOSIGNAL", 140.0)]


def test_epochs_out_of_order_are_sorted():
    epoch = {
        "time": np.array([60.0, 0.0]),
        "duration": np.array([60.0, 60.0]),
        "label": np.array(["REM", "WAKE"], dtype="U16"),
    }

    back = from_epoch_dict(epoch)

    assert bouts(back) == [("WAKE", 60.0), ("REM", 60.0)]


def test_an_empty_hypnogram_becomes_one_filler_segment():
    empty = {
        "time": np.array([]),
        "duration": np.array([]),
        "label": np.array([], dtype="U16"),
    }

    back = from_epoch_dict(empty, duration=100.0)

    assert bouts(back) == [("NOSIGNAL", 100.0)]


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


def test_the_palette_comes_from_nyx_so_everything_agrees():
    labels, colours = stage_palette(["REM", "NREM", "WAKE"])

    assert set(labels) == {"REM", "NREM", "WAKE"}
    for label, colour in zip(labels, colours, strict=True):
        assert colour == COLORS[label]


def test_the_palette_is_in_display_order():
    labels, _ = stage_palette(["NREM", "REM", "WAKE"])

    # Shallowest first, as the hypnogram rows are drawn.
    assert labels.index("WAKE") < labels.index("REM") < labels.index("NREM")


def test_an_unknown_stage_still_gets_a_colour():
    labels, colours = stage_palette(["WAKE", "SPINDLE"])

    assert "SPINDLE" in labels
    assert colours[labels.index("SPINDLE")] == "#888888"
