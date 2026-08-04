"""Coarser hypnograms come from merging labels, not from clustering again."""

from __future__ import annotations

import numpy as np
import pytest

from nyx.granularity import GRANULARITIES, available_granularities, collapse


def _hypno(pairs):
    """Build a hypnogram from (label, duration) pairs."""
    labels = [p[0] for p in pairs]
    durations = [float(p[1]) for p in pairs]
    times = np.cumsum([0.0] + durations[:-1])
    return {
        "time": np.asarray(times, dtype="float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U"),
    }


FIVE_STAGE = _hypno(
    [("WAKE", 30), ("NREM1", 30), ("NREM2", 60), ("NREM3", 30), ("REM", 30)]
)


def test_five_stage_is_unchanged():
    result = collapse(FIVE_STAGE, 5)
    assert list(result["label"]) == ["WAKE", "NREM1", "NREM2", "NREM3", "REM"]


def test_four_stage_folds_n1_into_nrem2():
    """The archived human runs save the merged label as NREM2, not a new name."""
    result = collapse(FIVE_STAGE, 4)

    assert list(result["label"]) == ["WAKE", "NREM2", "NREM3", "REM"]
    # N1 and N2 were adjacent, so they merge into a single 90 s interval.
    assert list(result["duration"]) == [30.0, 90.0, 30.0, 30.0]


def test_three_stage_merges_all_nrem():
    result = collapse(FIVE_STAGE, 3)

    assert list(result["label"]) == ["WAKE", "NREM", "REM"]
    assert list(result["duration"]) == [30.0, 120.0, 30.0]


def test_total_time_is_conserved():
    total = FIVE_STAGE["duration"].sum()
    for n in (5, 4, 3):
        assert collapse(FIVE_STAGE, n)["duration"].sum() == pytest.approx(total)


def test_collapsing_is_transitive():
    """5 -> 3 directly must equal 5 -> 4 -> 3."""
    direct = collapse(FIVE_STAGE, 3)
    stepwise = collapse(collapse(FIVE_STAGE, 4), 3)

    assert list(direct["label"]) == list(stepwise["label"])
    assert list(direct["duration"]) == list(stepwise["duration"])


def test_short_n1_aliases_are_handled():
    """Some references label the stages N1/N2/N3 rather than NREM1/2/3."""
    short = _hypno([("WAKE", 30), ("N1", 30), ("N2", 30)])
    assert list(collapse(short, 3)["label"]) == ["WAKE", "NREM"]


def test_rejects_an_unsupported_granularity():
    with pytest.raises(ValueError, match="must be one of"):
        collapse(FIVE_STAGE, 2)


def test_rodent_hypnogram_offers_only_three_stages():
    rodent = _hypno([("WAKE", 30), ("NREM", 30), ("REM", 30)])
    assert available_granularities(rodent) == [3]


def test_human_hypnogram_offers_all_three():
    assert available_granularities(FIVE_STAGE) == [5, 4, 3]


def test_nosignal_does_not_count_as_a_stage():
    with_gaps = _hypno([("WAKE", 30), ("NREM", 30), ("REM", 30), ("NOSIGNAL", 30)])
    assert available_granularities(with_gaps) == [3]


def test_a_bare_wake_sleep_split_has_no_standard_granularity():
    assert available_granularities(_hypno([("WAKE", 30), ("SLEEP", 30)])) == []


def test_granularity_maps_are_declarative():
    """The maps are data, so a new stage vocabulary needs no code change."""
    assert GRANULARITIES[5] == {}
    assert GRANULARITIES[4]["NREM1"] == "NREM2"
    assert set(GRANULARITIES[3].values()) == {"NREM"}
