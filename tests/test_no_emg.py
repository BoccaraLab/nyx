"""Scoring without a recorded EMG channel, and the surrogate that replaces it."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.emg_like import emg_from_lfp


@pytest.fixture(scope="module")
def wideband():
    """Multi-channel wideband signal with volume-conducted muscle."""
    return nyx.demo_wideband_recording()


@pytest.fixture
def no_emg(synthetic_recording):
    """The synthetic recording with its EMG channel taken away."""
    from dataclasses import replace

    return replace(synthetic_recording, emg=None)


NO_EMG_PARAMS = {
    "features": {"method": "spectrogram"},
    "EEG": {"min_freq": 0.5, "max_freq": 30, "binsize": 2, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": "mean"},
    "EMG": {"min_freq": 30, "max_freq": 60, "binsize": 4, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": "mean"},
    "scoring": {"pc_components": 4, "min_duration": 4},
    "steps": [
        {"name": "split_all", "within": "SLEEP", "method": "gmm", "n_clusters": 3,
         "pcs_to_use": [0, 1, 2], "use_emg": False,
         "stage_order": ["REM", "WAKE", "NREM"]},
    ],
}


# ---------------------------------------------------------------------------
# A Recording without an EMG
# ---------------------------------------------------------------------------


def test_a_recording_can_have_no_emg(no_emg):
    assert not no_emg.has_emg
    assert no_emg.emg_fs == no_emg.fs


def test_describe_says_so_rather_than_crashing(no_emg):
    text = no_emg.describe()
    assert "none" in text
    assert "less reliable" in text


def test_asking_for_the_trace_explains_the_options(no_emg):
    with pytest.raises(ValueError, match="emg_from_lfp"):
        no_emg.emg_trace()


def test_slicing_keeps_the_emg_absent(no_emg):
    assert not no_emg.time_slice(0, 100).has_emg


def test_preprocessing_skips_the_missing_channel(no_emg):
    processed = nyx.preprocess_recording(no_emg, {"EEG": {"resample": 64}})

    assert processed.fs == 64
    assert not processed.has_emg


def test_emg_features_are_none_rather_than_an_error(no_emg):
    assert nyx.compute_emg_features(no_emg, NO_EMG_PARAMS) is None


def test_signal_check_shows_the_one_channel(no_emg):
    check = nyx.check_signals(no_emg, preview=None)

    assert check.previews[0].channels == ("EEG",)
    assert all(key.startswith("EEG") for key in check.line_noise)
    check.plot()  # must not raise on the missing row


# ---------------------------------------------------------------------------
# Scoring without one
# ---------------------------------------------------------------------------


def test_scoring_without_an_emg_works_and_warns(no_emg):
    with pytest.warns(UserWarning, match="without an EMG"):
        result = nyx.score_recording(no_emg, NO_EMG_PARAMS, window=(0, 2400),
                                     verbose=False)

    assert result.emg is None
    assert set(result.hypnogram["label"]) <= {"WAKE", "NREM", "REM", "NOSIGNAL"}


def test_everything_starts_as_sleep(no_emg):
    """There is no threshold to apply, so one step has to find all three stages."""
    with pytest.warns(UserWarning):
        result = nyx.score_recording(no_emg, NO_EMG_PARAMS, window=(0, 2400),
                                     verbose=False)

    assert set(result.wake_sleep.hypnogram["label"]) == {"SLEEP"}
    assert result.wake_sleep.threshold_source == "none"


def test_use_emg_without_an_emg_says_what_to_do(no_emg):
    params = {**NO_EMG_PARAMS}
    params["steps"] = [{**NO_EMG_PARAMS["steps"][0], "use_emg": True}]

    with pytest.raises(ValueError, match="emg_from_lfp"), pytest.warns(UserWarning):
        nyx.score_recording(no_emg, params, window=(0, 2400), verbose=False)


def test_the_figures_survive_a_missing_emg(no_emg):
    import matplotlib
    matplotlib.use("Agg")

    with pytest.warns(UserWarning):
        result = nyx.score_recording(no_emg, NO_EMG_PARAMS, window=(0, 2400),
                                     verbose=False)

    nyx.plot_summary(result)
    nyx.plot_scoring_overview(result)


def test_saving_skips_the_emg_power_file(no_emg, tmp_path):
    with pytest.warns(UserWarning):
        result = nyx.score_recording(no_emg, NO_EMG_PARAMS, window=(0, 2400),
                                     verbose=False)
    nyx.save_results(result, str(tmp_path), plots=False)

    assert (tmp_path / "hypnogram.csv").exists()
    assert not (tmp_path / "emg_power.npy").exists()


def test_the_shipped_no_emg_params_load():
    params = nyx.load_params("params/mouse_no_emg.json")

    assert not any(s["method"] == "emg_threshold" for s in params["steps"])
    assert not any(s.get("use_emg") for s in params["steps"])


# ---------------------------------------------------------------------------
# The surrogate
# ---------------------------------------------------------------------------


def test_emg_from_lfp_tracks_muscle_tone(wideband):
    """High during wake, low during sleep -- which is the whole point."""
    recording, hypnogram = wideband
    emg = emg_from_lfp(recording)

    wake, sleep = [], []
    for time, duration, label in zip(
        hypnogram["time"], hypnogram["duration"], hypnogram["label"]
    ):
        start, end = int(time * emg.fs), int((time + duration) * emg.fs)
        (wake if str(label) == "WAKE" else sleep).extend(emg.power[start:end])

    assert np.mean(wake) > np.mean(sleep) + 0.3


def test_the_surrogate_looks_like_ordinary_emg_features(wideband):
    """So the threshold, the figures and use_emg all work unchanged."""
    emg = emg_from_lfp(wideband[0])

    assert emg.power.min() == pytest.approx(0.0)
    assert emg.power.max() == pytest.approx(1.0)
    assert emg.spectrogram.shape == (1, len(emg.power))
    assert len(emg.times) == len(emg.power)
    assert emg.fs == pytest.approx(10.0)  # 0.1 s bins


def test_a_threshold_can_be_found_on_it(wideband):
    emg = emg_from_lfp(wideband[0])
    wake_sleep = nyx.classify_wake_sleep(emg)

    assert {"WAKE", "SLEEP"} <= set(wake_sleep.hypnogram["label"])


def test_one_channel_is_refused(wideband):
    """The metric is how much channels agree, which is undefined for one."""
    with pytest.raises(ValueError, match="at least two channels"):
        emg_from_lfp(wideband[0], channels=["LFP1"])


def test_a_downsampled_recording_is_refused(wideband):
    """200 Hz cannot carry 275-600 Hz, and this is the usual mistake."""
    import spikeinterface.preprocessing as spre

    low = spre.resample(wideband[0], resample_rate=200)
    with pytest.raises(ValueError, match="Nyquist"):
        emg_from_lfp(low)


def test_channels_can_be_chosen_by_name(wideband):
    emg = emg_from_lfp(wideband[0], channels=["LFP1", "LFP3"])
    assert len(emg.power) > 0


def test_several_recordings_are_stacked(wideband):
    """Combining an LFP folder with an ECoG folder is the real use."""
    recording = wideband[0]
    left = recording.select_channels(recording.get_channel_ids()[:2])
    right = recording.select_channels(recording.get_channel_ids()[2:])

    together = emg_from_lfp([left, right])
    assert np.allclose(together.power, emg_from_lfp(recording).power)


def test_a_bare_array_says_what_is_missing(wideband):
    with pytest.raises(TypeError, match="sampling rate"):
        emg_from_lfp(np.zeros((4, 10_000)))


def test_scoring_with_a_surrogate(synthetic_recording, wideband):
    """The surrogate goes in through score_recording's emg= argument."""
    from dataclasses import replace

    emg = emg_from_lfp(wideband[0])
    stripped = replace(synthetic_recording, emg=None)

    result = nyx.score_recording(stripped, nyx.demo_params(), window=(0, 2400),
                                 emg=emg, verbose=False)

    assert result.emg is emg
    assert result.wake_sleep.threshold_source == "auto"
