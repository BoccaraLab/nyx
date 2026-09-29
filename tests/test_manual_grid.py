"""The epoch grid manual scoring is confined to. Qt-free."""

from __future__ import annotations

import numpy as np
import pytest

from nyx.gui.hypnogram import EpochGrid, merge_touching, onto_grid


def hyp(times, durations, labels) -> dict:
    return {
        "time": np.asarray(times, dtype="float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U16"),
    }


def test_the_grid_is_anchored_to_the_recording_not_the_window():
    # A window from 10 s, shown from 0 on the viewer: the recording's 4 s grid
    # lines at 12, 16... are at 2, 6... there.
    grid = EpochGrid.for_window(4.0, window_start=10.0, window_duration=1200.0)
    assert grid.origin == pytest.approx(2.0)
    assert grid.low == pytest.approx(2.0)       # the first whole epoch
    assert grid.high == pytest.approx(1198.0)   # the last one ends by 1210
    assert grid.n_epochs() == 299


def test_a_time_belongs_to_the_epoch_it_is_in():
    grid = EpochGrid(4.0)
    assert grid.floor(5.3) == 4.0
    assert grid.floor(4.0) == 4.0
    assert grid.floor(3.9999999) == 4.0          # float noise, not the epoch before
    assert grid.ceil(5.3) == 8.0


def test_snapping_moves_both_ends_to_the_nearest_line():
    grid = EpochGrid(4.0)
    times, durations = grid.snap([3.1, 10.0], [5.3, 1.0])
    assert list(times) == [4.0, 8.0]
    assert list(durations) == [4.0, 4.0]


def test_an_epoch_too_short_to_reach_a_line_snaps_to_nothing():
    grid = EpochGrid(4.0)
    _times, durations = grid.snap([4.5], [1.0])
    assert durations[0] == 0.0


def test_nothing_is_scored_outside_the_window():
    grid = EpochGrid(4.0, low=8.0, high=20.0)
    times, durations = grid.snap([0.0], [40.0])
    assert (times[0], times[0] + durations[0]) == (8.0, 20.0)


def test_a_scoring_at_another_epoch_length_takes_the_majority():
    grid = EpochGrid(4.0, low=0.0, high=12.0)
    # 0-5 WAKE, 5-12 NREM: the first epoch is WAKE, the second mostly NREM.
    out = onto_grid(hyp([0, 5], [5, 7], ["WAKE", "NREM"]), grid)
    assert list(out["time"]) == [0.0, 4.0]
    assert list(out["duration"]) == [4.0, 8.0]
    assert list(out["label"]) == ["WAKE", "NREM"]


def test_unscored_epochs_stay_unscored():
    grid = EpochGrid(4.0, low=0.0, high=16.0)
    out = onto_grid(hyp([0, 12], [4, 4], ["WAKE", "WAKE"]), grid)
    assert list(out["time"]) == [0.0, 12.0]      # not filled, not merged across
    assert list(out["duration"]) == [4.0, 4.0]


def test_merging_joins_only_epochs_that_touch():
    out = merge_touching(hyp([0, 4, 12], [4, 4, 4], ["REM", "REM", "REM"]))
    assert list(out["time"]) == [0.0, 12.0]
    assert list(out["duration"]) == [8.0, 4.0]


def test_the_epoch_length_has_to_be_positive():
    with pytest.raises(ValueError):
        EpochGrid(0.0)
