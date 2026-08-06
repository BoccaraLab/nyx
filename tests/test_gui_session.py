"""The GUI's pipeline state, tested without a GUI.

Everything here runs in the ordinary test suite: no Qt, no display, no extra
install. That is the point of keeping the session layer Qt-free, and
:func:`test_the_session_layer_imports_no_qt` is what keeps it that way.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import nyx
from nyx.gui.filters import (
    annotation_filters,
    annotation_formats,
    can_list_channels,
    format_is_folder,
    recording_filters,
    recording_formats,
)
from nyx.gui.session import NotComputed, ScoringSession, Stage

WINDOW = (0.0, 2400.0)


@pytest.fixture
def session(synthetic_recording, params):
    s = ScoringSession(params)
    s.set_recording(synthetic_recording)
    s.set_window(WINDOW)
    return s


@pytest.fixture
def scored(session):
    session.compute_through(Stage.RESULT)
    return session


# ---------------------------------------------------------------------------
# It has to be the same pipeline
# ---------------------------------------------------------------------------


def test_the_session_scores_exactly_what_score_recording_scores(
    scored, synthetic_recording, params
):
    """The highest-value test here: the GUI must not be a second pipeline."""
    reference = nyx.score_recording(
        synthetic_recording, params, window=WINDOW, verbose=False
    )

    ours = scored.result().hypnogram
    theirs = reference.hypnogram

    assert list(ours["label"]) == list(theirs["label"])
    assert np.allclose(ours["time"], theirs["time"])
    assert np.allclose(ours["duration"], theirs["duration"])
    assert scored.wake_sleep().threshold == pytest.approx(
        reference.wake_sleep.threshold
    )
    assert scored.cluster_to_stage() == reference.staging.cluster_to_stage


def test_the_result_bundles_what_score_recording_bundles(scored, params):
    result = scored.result()

    assert result.window == WINDOW
    assert result.params is scored.params
    assert result.steps and len(result.steps) == len(scored.steps)
    assert result.recording is scored.windowed()


def test_postprocessing_rules_are_applied(session):
    session.set_postprocess([{"min_duration": {"seconds": 60}}])
    session.compute_through(Stage.RESULT)

    hypnogram = session.staging().hypnogram
    scored_only = hypnogram["duration"][hypnogram["label"] != "NOSIGNAL"]
    assert scored_only.min() >= 60


# ---------------------------------------------------------------------------
# Invalidation: what survives a change, by identity
# ---------------------------------------------------------------------------


def test_moving_the_threshold_keeps_the_emg_features(scored):
    emg = scored.emg()

    scored.set_emg_threshold(0.6)

    # The expensive thing survives; everything downstream of the cut does not.
    assert scored.emg() is emg
    assert not scored.has(Stage.WAKE_SLEEP)
    assert not scored.has(Stage.STEPS)


def test_changing_the_clustering_keeps_the_pca(scored):
    pca = scored.pca()

    scored.set_clustering({"method": "kmeans", "n_clusters": 2})

    # The PCA is the minutes; re-clustering it is seconds. Losing it here
    # would make every clustering tweak cost a full recompute.
    assert scored.pca() is pca
    with pytest.raises(NotComputed):
        scored.clusters()


def test_renaming_clusters_keeps_the_clustering(scored):
    clusters = scored.clusters()
    pca = scored.pca()

    scored.set_stage_order(["NREM", "REM"])

    assert scored.clusters() is clusters
    assert scored.pca() is pca


def test_renaming_clusters_actually_renames_them(scored):
    before = scored.cluster_to_stage()

    scored.set_stage_order(["NREM", "REM"])
    scored.compute_through(Stage.RESULT)

    assert scored.cluster_to_stage() == {k: _swap(v) for k, v in before.items()}


def _swap(stage):
    return {"REM": "NREM", "NREM": "REM"}.get(stage, stage)


def test_naming_a_cluster_by_id_wins_over_position(scored):
    scored.set_cluster_overrides({0: "NREM", 1: "NREM"})
    scored.compute_through(Stage.RESULT)

    assert set(scored.cluster_to_stage().values()) == {"NREM"}


def test_changing_the_window_drops_everything_below_it(scored):
    scored.set_window((0.0, 1200.0))

    assert not scored.has(Stage.PREPROCESS)
    assert not scored.has(Stage.EMG)
    assert scored.has(Stage.LOAD)


def test_changing_the_reference_keeps_the_scoring(scored):
    clusters = scored.clusters()

    scored.set_reference(None)

    assert scored.clusters() is clusters
    assert scored.has(Stage.STEPS)


def test_every_change_bumps_the_generation(session):
    generations = [session.generation()]
    for change in (
        lambda: session.set_window((0, 1200)),
        lambda: session.set_emg_threshold(0.4),
        lambda: session.set_postprocess([]),
    ):
        change()
        generations.append(session.generation())

    assert generations == sorted(set(generations))


# ---------------------------------------------------------------------------
# Accessors never compute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["windowed", "emg", "wake_sleep", "pca", "clusters", "outcome", "staging"]
)
def test_an_uncomputed_accessor_raises_rather_than_computing(session, name):
    # A repaint or a tooltip must never be able to start a five-minute PCA.
    with pytest.raises(NotComputed):
        getattr(session, name)()


def test_not_computed_says_which_stage_and_how_to_fix_it(session):
    with pytest.raises(NotComputed) as error:
        session.pca()

    assert error.value.stage is Stage.STEPS
    assert "compute_through" in str(error.value)


def test_a_session_with_no_recording_says_so():
    session = ScoringSession(nyx.demo_params())

    with pytest.raises(NotComputed):
        _ = session.recording


# ---------------------------------------------------------------------------
# Stage bookkeeping, which the tab badges read
# ---------------------------------------------------------------------------


def test_stages_become_ready_in_order(session):
    assert session.ready() is Stage.LOAD

    session.compute_through(Stage.EMG)
    assert session.has(Stage.PREPROCESS)
    assert session.has(Stage.EMG)
    assert not session.has(Stage.WAKE_SLEEP)
    assert session.ready() is Stage.EMG


def test_computing_through_skips_what_is_already_there(session):
    session.compute_through(Stage.EMG)
    emg = session.emg()

    session.compute_through(Stage.RESULT)

    assert session.emg() is emg


def test_stopping_between_stages_works(session):
    calls = []

    def should_stop():
        calls.append(len(calls))
        return len(calls) > 2

    session.compute_through(Stage.RESULT, should_stop=should_stop)

    # It gets somewhere and then stops; the point is that it stops at all.
    assert not session.has(Stage.STEPS)


def test_the_automatic_threshold_is_fitted_once(session):
    session.compute_through(Stage.EMG)
    automatic = session.auto_threshold()

    session.set_emg_threshold(0.3)
    session.compute_through(Stage.WAKE_SLEEP)

    assert session.auto_threshold() == automatic
    assert session.wake_sleep().threshold == pytest.approx(0.3)


def test_clearing_the_threshold_goes_back_to_automatic(session):
    session.compute_through(Stage.WAKE_SLEEP)
    automatic = session.auto_threshold()

    session.set_emg_threshold(0.3)
    session.set_emg_threshold(None)
    session.compute_through(Stage.WAKE_SLEEP)

    assert session.wake_sleep().threshold == pytest.approx(automatic)


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


def test_a_listener_hears_the_stage_that_changed(session):
    heard = []
    session.add_listener(heard.append)

    session.set_emg_threshold(0.5)

    assert Stage.WAKE_SLEEP in heard


def test_a_removed_listener_hears_nothing(session):
    heard = []
    session.add_listener(heard.append)
    session.remove_listener(heard.append)

    session.set_emg_threshold(0.5)

    assert heard == []


# ---------------------------------------------------------------------------
# Manual edits
# ---------------------------------------------------------------------------


def test_an_edited_hypnogram_replaces_the_scored_one(scored):
    edited = {
        "time": np.array([0.0]),
        "duration": np.array([2400.0]),
        "label": np.array(["WAKE"], dtype="U16"),
    }

    scored.set_hypnogram(edited)

    assert list(scored.staging().hypnogram["label"]) == ["WAKE"]
    assert scored.was_edited()


def test_an_edit_is_recorded_in_the_run_config(scored):
    scored.set_hypnogram(
        {
            "time": np.array([0.0]),
            "duration": np.array([2400.0]),
            "label": np.array(["WAKE"], dtype="U16"),
        }
    )

    decisions = scored.to_run_config().decisions

    # Without this a saved run.json would claim to replay a scoring that a
    # human changed by hand.
    assert decisions["manual_edit"]["edited"] is True
    assert decisions["manual_edit"]["source"] == "manual"


def test_the_run_config_records_the_decisions(scored):
    decisions = scored.to_run_config().decisions

    assert decisions["window"] == list(WINDOW)
    assert decisions["emg_threshold"] == pytest.approx(scored.wake_sleep().threshold)
    assert "manual_edit" not in decisions


def test_saving_writes_a_hypnogram_and_a_run_record(scored, tmp_path):
    scored.save(str(tmp_path), plots=False)

    assert (tmp_path / "hypnogram.csv").exists()
    assert (tmp_path / "run.json").exists()


def test_the_saved_record_can_be_loaded_back(scored, tmp_path):
    scored.save(str(tmp_path), plots=False)

    config = nyx.load_config(str(tmp_path / "run.json"))

    # Without a recording section the record cannot be loaded at all, which
    # makes it useless as a record of anything.
    assert config.recording is not None
    assert config.window == scored.window


def test_a_hand_chosen_threshold_survives_into_the_record(
    session, synthetic_recording, tmp_path
):
    session.compute_through(Stage.EMG)
    session.set_emg_threshold(0.42)
    session.compute_through(Stage.RESULT)
    session.save(str(tmp_path), plots=False)

    record = nyx.load_json(str(tmp_path / "run.json"))
    replayed = nyx.score_recording(
        synthetic_recording, record["params"],
        window=tuple(record["window"]), verbose=False,
    )

    assert replayed.wake_sleep.threshold == pytest.approx(0.42)
    assert list(replayed.hypnogram["label"]) == list(
        session.result().hypnogram["label"]
    )


# ---------------------------------------------------------------------------
# Filters follow the registries
# ---------------------------------------------------------------------------


def test_the_filters_name_every_readable_format():
    filters = recording_filters()

    for format in recording_formats():
        if format_is_folder(format):
            continue
        assert format in filters or not _has_extension(format)


def _has_extension(format):
    from nyx.gui.filters import extensions_for

    return bool(extensions_for(format))


def test_a_newly_registered_reader_shows_up_without_touching_the_gui():
    from nyx.io import RECORDING_READERS, register_recording_reader

    @register_recording_reader("test_only_format")
    def _reader(path, eeg_channel, emg_channel, **kwargs):  # pragma: no cover
        raise AssertionError("never called")

    try:
        assert "test_only_format" in recording_formats()
    finally:
        RECORDING_READERS.pop("test_only_format", None)


def test_spikeinterface_is_a_folder_and_is_kept_out_of_the_file_dialog():
    assert format_is_folder("spikeinterface")
    assert "spikeinterface" not in recording_filters()


def test_the_formats_whose_channels_can_be_listed_are_known():
    assert can_list_channels("edf")
    assert not can_list_channels("nonsense")


def test_annotation_filters_cover_the_annotation_readers():
    filters = annotation_filters()

    assert "visbrain_hyp" in filters
    assert ".hyp" in filters
    assert set(annotation_formats()) >= {"epoch_csv", "interval_csv"}


def test_every_filter_string_ends_with_all_files():
    for filters in (recording_filters(), annotation_filters()):
        assert filters.split(";;")[-1] == "All files (*)"


# ---------------------------------------------------------------------------
# The rule that keeps this file runnable
# ---------------------------------------------------------------------------


def test_the_session_layer_imports_no_qt():
    """No Qt, directly or transitively, in the pipeline-state layer.

    If this ever fails, the GUI's state has grown a dependency on the GUI and
    none of the tests above can run without a display any more.
    """
    forbidden = ("PySide", "PyQt", "pyqtgraph", "ephyviewer")

    for name in list(sys.modules):
        if name.startswith(forbidden):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("nyx.gui"):
            del sys.modules[name]

    import nyx.gui.filters  # noqa: F401
    import nyx.gui.session  # noqa: F401

    leaked = sorted({
        name.split(".")[0]
        for name in sys.modules
        if name.startswith(forbidden)
    })
    assert leaked == []
