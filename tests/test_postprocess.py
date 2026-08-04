"""Rules applied to a finished hypnogram.

The expected outcomes here are the paper's rule definitions (R1-R7 in the rule
comparison), so these tests pin nyx to what was published.
"""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.postprocess import (
    apply_rules,
    min_duration,
    rem_after_wake,
    rem_flanked_by_wake,
)


def hyp(*bouts) -> dict:
    """Build a hypnogram from ``(label, duration)`` pairs."""
    labels = [label for label, _ in bouts]
    durations = [float(seconds) for _, seconds in bouts]
    return {
        "time": np.cumsum([0.0] + durations[:-1]).astype("float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U"),
    }


def as_bouts(hypnogram) -> list[tuple[str, float]]:
    return [
        (str(label), float(duration))
        for label, duration in zip(hypnogram["label"], hypnogram["duration"])
    ]


# ---------------------------------------------------------------------------
# R1 / R2 -- REM flanked by wake
# ---------------------------------------------------------------------------


def test_rem_between_two_wake_bouts_becomes_wake():
    result = rem_flanked_by_wake(hyp(("WAKE", 60), ("REM", 10), ("WAKE", 60)))

    # Rewritten and then merged into a single bout, which is the point of
    # re-merging: three segments in, one out.
    assert as_bouts(result) == [("WAKE", 130.0)]


def test_rem_with_wake_on_only_one_side_is_left_alone():
    unchanged = hyp(("WAKE", 60), ("REM", 10), ("NREM", 60))
    assert as_bouts(rem_flanked_by_wake(unchanged)) == as_bouts(unchanged)


def test_max_duration_spares_long_rem_bouts():
    """R2: a long REM bout between two wake bouts is more likely to be real."""
    recording = hyp(("WAKE", 60), ("REM", 30), ("WAKE", 60))

    assert as_bouts(rem_flanked_by_wake(recording, max_duration=20)) == as_bouts(recording)
    assert as_bouts(rem_flanked_by_wake(recording, max_duration=None)) == [("WAKE", 150.0)]


def test_output_is_numpy_arrays():
    """Everything downstream indexes these as arrays."""
    result = rem_flanked_by_wake(hyp(("WAKE", 60), ("REM", 10), ("WAKE", 60)))

    for key in ("time", "duration", "label"):
        assert isinstance(result[key], np.ndarray), key


def test_start_times_stay_consistent_after_merging():
    result = rem_flanked_by_wake(
        hyp(("NREM", 40), ("WAKE", 60), ("REM", 10), ("WAKE", 60), ("NREM", 40))
    )

    times = result["time"]
    durations = result["duration"]
    assert np.allclose(times[1:], (times + durations)[:-1])


def test_a_hypnogram_with_fewer_than_three_bouts_is_safe():
    assert as_bouts(rem_flanked_by_wake(hyp(("WAKE", 60), ("REM", 10)))) == [
        ("WAKE", 60.0), ("REM", 10.0)
    ]


# ---------------------------------------------------------------------------
# R3 / R4 -- REM after wake
# ---------------------------------------------------------------------------


def test_rem_straight_out_of_wake_becomes_wake():
    result = rem_after_wake(hyp(("WAKE", 60), ("REM", 30), ("NREM", 60)))

    assert as_bouts(result) == [("WAKE", 90.0), ("NREM", 60.0)]


def test_rem_after_nrem_is_left_alone():
    """The normal NREM -> REM transition must survive."""
    normal = hyp(("NREM", 200), ("REM", 60), ("NREM", 200))
    assert as_bouts(rem_after_wake(normal)) == as_bouts(normal)


def test_min_wake_duration_spares_rem_after_a_brief_wake():
    """R3 only fires after a wake bout longer than 20s; R4 after any."""
    recording = hyp(("NREM", 100), ("WAKE", 10), ("REM", 30), ("NREM", 100))

    assert as_bouts(rem_after_wake(recording, min_wake_duration=20)) == as_bouts(recording)
    assert as_bouts(rem_after_wake(recording, min_wake_duration=0)) == [
        ("NREM", 100.0), ("WAKE", 40.0), ("NREM", 100.0)
    ]


def test_rewrites_do_not_cascade():
    """One left-to-right pass reading the ORIGINAL labels, as in the paper.

    Without that, the REM this rule turns into WAKE would itself become a
    'preceding wake bout' and eat the REM after it.
    """
    result = rem_after_wake(hyp(("WAKE", 60), ("REM", 30), ("NREM", 30), ("REM", 30)))

    assert as_bouts(result) == [("WAKE", 90.0), ("NREM", 30.0), ("REM", 30.0)]


# ---------------------------------------------------------------------------
# R5 -- minimum segment length
# ---------------------------------------------------------------------------


def test_short_segment_inside_one_stage_is_absorbed_by_it():
    result = min_duration(hyp(("NREM", 100), ("REM", 2), ("NREM", 100)), seconds=4)

    assert as_bouts(result) == [("NREM", 202.0)]


def test_short_segment_between_different_stages_goes_to_wake():
    result = min_duration(hyp(("NREM", 100), ("REM", 2), ("WAKE", 100)), seconds=4)

    assert as_bouts(result) == [("NREM", 100.0), ("WAKE", 102.0)]


def test_short_wake_becomes_nrem():
    """A micro-arousal too brief to score is sleep, not wake."""
    result = min_duration(hyp(("NREM", 100), ("WAKE", 2), ("REM", 100)), seconds=4)

    assert as_bouts(result) == [("NREM", 102.0), ("REM", 100.0)]


def test_nothing_shorter_than_the_threshold_survives():
    result = min_duration(
        hyp(("NREM", 100), ("REM", 2), ("NREM", 1), ("WAKE", 3), ("NREM", 100)),
        seconds=4,
    )

    assert min(duration for _, duration in as_bouts(result)) >= 4


def test_a_short_segment_at_either_end_takes_its_only_neighbour():
    """It has one neighbour, so there is nothing to decide between.

    Sending it to the fallback instead can leave it alive: a short NREM at the
    end next to REM goes to WAKE on the NREM pass and back to NREM on the WAKE
    pass, with no later pass to catch it.
    """
    assert as_bouts(min_duration(hyp(("REM", 100), ("NREM", 1)), seconds=4)) == [
        ("REM", 101.0)
    ]
    assert as_bouts(min_duration(hyp(("NREM", 1), ("REM", 100)), seconds=4)) == [
        ("REM", 101.0)
    ]


def test_a_long_recording_is_untouched_when_every_bout_is_long_enough():
    clean = hyp(("WAKE", 300), ("NREM", 600), ("REM", 120), ("NREM", 600))
    assert as_bouts(min_duration(clean, seconds=4)) == as_bouts(clean)


def test_stages_outside_the_order_are_left_alone():
    """A vocabulary this rule does not know about must not be silently rewritten."""
    result = min_duration(
        hyp(("NREM2", 100), ("TR", 2), ("NREM3", 100)), seconds=4
    )
    assert as_bouts(result) == [("NREM2", 100.0), ("TR", 2.0), ("NREM3", 100.0)]


def test_human_recordings_use_a_30s_minimum():
    result = min_duration(hyp(("NREM", 300), ("REM", 20), ("NREM", 300)), seconds=30)

    assert as_bouts(result) == [("NREM", 620.0)]


# ---------------------------------------------------------------------------
# NOSIGNAL is transparent
# ---------------------------------------------------------------------------


def test_an_artefact_gap_does_not_hide_the_flanking_wake():
    """Without this, one artefact epoch silently switches the rule off."""
    result = rem_flanked_by_wake(
        hyp(("WAKE", 60), ("NOSIGNAL", 4), ("REM", 10), ("WAKE", 60))
    )

    assert as_bouts(result) == [("WAKE", 60.0), ("NOSIGNAL", 4.0), ("WAKE", 70.0)]


def test_nosignal_is_never_relabelled():
    result = min_duration(hyp(("NREM", 100), ("NOSIGNAL", 1), ("NREM", 100)), seconds=4)

    assert ("NOSIGNAL", 1.0) in as_bouts(result)


def test_ignore_can_be_switched_off():
    blocked = hyp(("WAKE", 60), ("NOSIGNAL", 4), ("REM", 10), ("WAKE", 60))

    assert as_bouts(rem_flanked_by_wake(blocked, ignore=())) == as_bouts(blocked)


# ---------------------------------------------------------------------------
# Composition -- R6 and R7
# ---------------------------------------------------------------------------


def test_r6_and_r7_differ_by_order():
    """R5-then-R4 is a gentler REM removal than R4-then-R5.

    Running the minimum-length pass first merges the brief wake blip into NREM,
    so the REM that only followed that blip is no longer 'after wake'.
    """
    recording = hyp(("NREM", 100), ("WAKE", 2), ("REM", 60), ("NREM", 100))

    r6 = apply_rules(recording, [{"rem_after_wake": {}}, {"min_duration": {"seconds": 4}}])
    r7 = apply_rules(recording, [{"min_duration": {"seconds": 4}}, {"rem_after_wake": {}}])

    assert "REM" not in set(r6["label"])
    assert "REM" in set(r7["label"])


def test_an_empty_rule_list_returns_the_input_unchanged():
    recording = hyp(("WAKE", 60), ("REM", 10), ("WAKE", 60))
    assert apply_rules(recording, []) is recording


def test_rules_accept_the_bare_string_form():
    result = apply_rules(hyp(("WAKE", 60), ("REM", 10), ("WAKE", 60)),
                         ["rem_flanked_by_wake"])
    assert as_bouts(result) == [("WAKE", 130.0)]


def test_unknown_rule_lists_what_is_available():
    with pytest.raises(ValueError, match="min_duration"):
        apply_rules(hyp(("WAKE", 60)), [{"tidy_it_up": {}}])


def test_malformed_rule_is_reported():
    with pytest.raises(ValueError, match="Cannot read the postprocessing rule"):
        apply_rules(hyp(("WAKE", 60)), [42])


# ---------------------------------------------------------------------------
# Through the params
# ---------------------------------------------------------------------------


def test_a_typo_in_the_params_is_caught_before_the_scoring_runs():
    params = {**nyx.demo_params(), "postprocess": [{"min_lenght": {"seconds": 4}}]}

    with pytest.raises(ValueError, match="unknown rule"):
        nyx.config.validate_params(params)


def test_scoring_applies_the_params_rules(synthetic_recording):
    params = {**nyx.demo_params(),
              "postprocess": [{"min_duration": {"seconds": 60}}]}

    result = nyx.score_recording(synthetic_recording, params, verbose=False)

    durations = result.hypnogram["duration"]
    labels = result.hypnogram["label"]
    scored = durations[labels != "NOSIGNAL"]
    assert scored.min() >= 60


def test_rules_land_in_the_run_record(synthetic_recording, tmp_path):
    rules = [{"rem_after_wake": {"min_wake_duration": 20}}]
    params = {**nyx.demo_params(), "postprocess": rules}

    result = nyx.score_recording(synthetic_recording, params, verbose=False)
    nyx.save_results(result, str(tmp_path))

    record = nyx.load_json(str(tmp_path / "run.json"))
    assert record["params"]["postprocess"] == rules
