"""The review window: the viewers, and the edit finding its way home."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.gui.session import Stage

pytestmark = pytest.mark.gui

pytest.importorskip("ephyviewer", reason="needs nyx-sleep[gui]")


@pytest.fixture
def review(qtbot):
    """Build review windows and close them deterministically.

    ephyviewer views own worker threads that only stop on closeEvent, so a
    window that is merely garbage collected takes the process down. qtbot's
    teardown is not enough on its own -- these have to be closed.
    """
    built = []

    def build(*args, **kwargs):
        from nyx.gui.review import build_review_window

        window = build_review_window(*args, **kwargs)
        built.append(window)
        qtbot.addWidget(window)
        return window

    yield build

    for window in built:
        window.close()


@pytest.fixture
def opened(qtbot):
    """open_review_window, closed deterministically. See `review`."""
    built = []

    def open_it(session):
        from nyx.gui.review import open_review_window

        window = open_review_window(session)
        built.append(window)
        qtbot.addWidget(window)
        return window

    yield open_it

    for window in built:
        window.close()


@pytest.fixture
def scalogram(qtbot, scored):
    """A NyxTimeFreqViewer, closed deterministically.

    It starts a worker thread per channel and stops them in closeEvent, so it
    has the same requirement the window does.
    """
    from ephyviewer.datasource import SpikeInterfaceRecordingSource

    from nyx.gui.viewers import NyxTimeFreqViewer

    source = SpikeInterfaceRecordingSource(recording=scored.result().recording.eeg)
    viewer = NyxTimeFreqViewer(source=source, name="EEG")
    qtbot.addWidget(viewer)
    yield viewer
    viewer.close()


@pytest.fixture
def encoders(qtbot):
    """Epoch encoders, closed deterministically."""
    built = []

    def make(**kwargs):
        from nyx.gui.viewers import NyxEpochEncoder

        encoder = NyxEpochEncoder(**kwargs)
        built.append(encoder)
        qtbot.addWidget(encoder)
        return encoder

    yield make

    for encoder in built:
        encoder.changes_since_save = 0
        encoder.close()


@pytest.fixture
def window(review, scored):
    return review(
        scored.result(), reference=scored.reference, params=scored.params
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_the_window_has_the_rows_the_overview_figure_has(window):
    assert list(window.viewers) == [
        "EMG", "EMG spectrum", "EEG", "EEG spectrum", "reference", "hypnogram"
    ]


def test_a_recording_without_an_emg_simply_has_no_emg_rows(review, session):
    import dataclasses

    session.compute_through(Stage.RESULT)
    # Copied, not mutated: Recording is a mutable dataclass and the fixture
    # behind it is session-scoped, so assigning to it corrupts other tests.
    result = dataclasses.replace(
        session.result(),
        recording=dataclasses.replace(session.result().recording, emg=None),
    )

    w = review(result, params=session.params)

    assert "EMG" not in w.viewers
    assert "hypnogram" in w.viewers


def test_without_a_reference_there_is_no_reference_row(review, scored):
    w = review(scored.result(), reference=None, params=scored.params)

    assert "reference" not in w.viewers


# ---------------------------------------------------------------------------
# The epoch source
# ---------------------------------------------------------------------------


def test_the_hypnogram_arrives_intact(window, scored):
    source = window.epoch_source
    hypnogram = scored.result().hypnogram

    assert len(source.ep_times) == len(hypnogram["time"])
    assert list(source.ep_labels) == [str(x) for x in hypnogram["label"]]


def test_it_survives_the_round_trip_unedited(window, scored):
    hypnogram = scored.result().hypnogram

    back = window.epoch_source.to_nyx()

    assert np.allclose(back["time"], hypnogram["time"])
    assert list(back["label"]) == [str(x) for x in hypnogram["label"]]


def test_the_offset_is_taken_from_the_source_not_from_the_window(review, session):
    """A windowed recording must not be shifted by the length of the trim."""
    from ephyviewer.datasource import SpikeInterfaceRecordingSource

    session.set_window((600.0, 2400.0))
    session.compute_through(Stage.RESULT)
    result = session.result()

    w = review(result, params=session.params)

    source = SpikeInterfaceRecordingSource(recording=result.recording.eeg)
    assert w.epoch_source.ep_times[0] == pytest.approx(float(source.t_start))


def test_the_stage_colours_are_nyxs(window):
    from nyx.stages import COLORS

    source = window.epoch_source
    for label, colour in zip(
        source.possible_labels, source.color_labels, strict=True
    ):
        if label in COLORS:
            assert colour == COLORS[label]


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def test_saving_hands_the_hypnogram_back_rather_than_writing_a_file(review, scored):
    caught = []
    w = review(scored.result(), params=scored.params, on_save=caught.append)

    w.epoch_source.save()

    assert len(caught) == 1
    assert set(caught[0]) == {"time", "duration", "label"}


def test_an_edit_reaches_the_session_and_the_saved_result(opened, scored, tmp_path):
    w = opened(scored)

    source = w.epoch_source
    source.ep_labels[:] = ["WAKE"] * len(source.ep_labels)
    source.save()

    assert scored.was_edited()
    assert set(scored.staging().hypnogram["label"]) == {"WAKE"}

    scored.save(str(tmp_path), plots=False)
    written = (tmp_path / "hypnogram.csv").read_text(encoding="utf-8")
    assert "NREM" not in written


def test_the_edit_is_recorded_in_the_run_json(opened, scored, tmp_path):
    w = opened(scored)
    w.epoch_source.save()

    scored.save(str(tmp_path), plots=False)
    record = nyx.load_json(str(tmp_path / "run.json"))

    # A record that cannot say a human changed the scoring would claim to
    # replay something it cannot.
    assert record["decisions"]["manual_edit"]["edited"] is True


# ---------------------------------------------------------------------------
# The subclassed viewers
# ---------------------------------------------------------------------------


def test_the_scalogram_viewer_exposes_what_the_fork_added(scalogram):
    group = scalogram.params.param("timefreq")
    names = [child.name() for child in group.children()]
    assert {"decibel", "smoothing_length", "zscore"} <= set(names)


def test_the_colour_limit_has_two_ends_so_decibels_can_be_shown(scalogram):
    # Upstream plots abs(wt) on [0, clim], which cannot show dB at all.
    scalogram.by_channel_params["ch0", "clim_min"] = -40.0
    scalogram.by_channel_params["ch0", "clim_max"] = 10.0
    assert scalogram.by_channel_params["ch0", "clim_min"] == -40.0


def test_the_viewer_can_be_matched_to_the_params_it_was_scored_with(scalogram, scored):
    from nyx.gui.viewers import timefreq_params_from

    viewer = scalogram
    viewer.apply_settings(timefreq_params_from(scored.params, "EEG"))

    band = scored.params["EEG"]
    assert viewer.params["timefreq", "f_start"] == pytest.approx(band["min_freq"])
    assert viewer.params["timefreq", "f_stop"] == pytest.approx(band["max_freq"])


def test_the_worker_produces_decibels(qtbot, scalogram):
    """The point of the subclass: what you see is what nyx computed."""
    viewer = scalogram
    viewer.refresh()
    qtbot.waitUntil(lambda: bool(viewer.last_wt_maps), timeout=15_000)

    wt_map = next(iter(viewer.last_wt_maps.values()))
    # abs(wt) is non-negative; dB of a small number is not.
    assert float(np.min(wt_map)) < 0.0


# ---------------------------------------------------------------------------
# Curation navigation
# ---------------------------------------------------------------------------


def test_the_encoder_flags_what_the_rules_would_rewrite(encoders):
    from nyx.gui.review import NyxEpochSource

    hypnogram = {
        "time": np.array([0.0, 60.0, 62.0, 122.0]),
        "duration": np.array([60.0, 2.0, 60.0, 60.0]),
        "label": np.array(["WAKE", "REM", "NREM", "WAKE"], dtype="U16"),
    }
    source = NyxEpochSource(hypnogram, window_duration=182.0)
    encoder = encoders(
        source=source, name="hypnogram",
        rules=[{"min_duration": {"seconds": 4}}],
    )

    # The 2 s REM is exactly what min_duration objects to -- and the threshold
    # is the rule's, not a hardcoded 2.5 as the fork had it.
    assert 1 in encoder.flagged_indices()
    assert "min_duration" in encoder.flag_reasons(1)


def test_no_rules_means_nothing_is_flagged(encoders, scored):
    from nyx.gui.review import NyxEpochSource

    source = NyxEpochSource(scored.result().hypnogram)
    encoder = encoders(source=source, name="hypnogram", rules=[])

    assert encoder.flagged_indices() == set()


def test_state_navigation_lands_on_the_next_change(encoders):
    from nyx.gui.review import NyxEpochSource

    hypnogram = {
        "time": np.array([0.0, 60.0, 120.0, 180.0]),
        "duration": np.array([60.0, 60.0, 60.0, 60.0]),
        "label": np.array(["WAKE", "WAKE", "NREM", "REM"], dtype="U16"),
    }
    encoder = encoders(source=NyxEpochSource(hypnogram), name="hypnogram")
    encoder.t = 0.0

    encoder.go_to_next_state()

    # Past the second WAKE, which is the point: it walks changes, not epochs.
    assert encoder.t == pytest.approx(120.0)


def test_state_navigation_goes_backwards_too(encoders):
    from nyx.gui.review import NyxEpochSource

    hypnogram = {
        "time": np.array([0.0, 60.0, 120.0]),
        "duration": np.array([60.0, 60.0, 60.0]),
        "label": np.array(["WAKE", "NREM", "NREM"], dtype="U16"),
    }
    encoder = encoders(source=NyxEpochSource(hypnogram), name="hypnogram")
    encoder.t = 130.0

    encoder.go_to_previous_state()

    assert encoder.t == pytest.approx(0.0)
