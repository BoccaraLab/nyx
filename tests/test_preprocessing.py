"""Notch filtering and resampling, applied through spikeinterface."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.preprocessing import preprocess_channel, preprocess_recording


@pytest.fixture(scope="module")
def humming():
    """A demo recording with 50 Hz mains interference."""
    return nyx.demo_recording(mains_hz=50.0)[0]


def _line_noise(recording, freq=50.0, channel="eeg"):
    """Peak-above-flanks score. Only meaningful on an unfiltered signal."""
    check = nyx.check_signals(recording, preview=None)
    return check.line_noise[f"{channel.upper()}_{freq:g}Hz"]


def _power_at(recording, freq, channel="eeg"):
    """Absolute mean power in a narrow band, in dB."""
    check = nyx.check_signals(recording, preview=None)
    freqs, spectrum = check.previews[0].mean_spectrum(channel)
    band = np.abs(freqs - freq) <= 1.0
    return float(spectrum[band].max())


# ---------------------------------------------------------------------------
# Nothing configured
# ---------------------------------------------------------------------------


def test_no_settings_leaves_the_recording_alone(synthetic_recording):
    same = preprocess_recording(synthetic_recording, {"EEG": {}, "EMG": {}})

    assert same.fs == synthetic_recording.fs
    assert np.array_equal(same.eeg_trace(), synthetic_recording.eeg_trace())


def test_missing_sections_are_fine(synthetic_recording):
    assert preprocess_recording(synthetic_recording, {}).fs == synthetic_recording.fs


# ---------------------------------------------------------------------------
# Notch
# ---------------------------------------------------------------------------


def test_notch_removes_the_hum(humming):
    before = _line_noise(humming)
    after = _line_noise(preprocess_recording(humming, {"EEG": {"notch": 50}}))

    assert before > 10, "the demo should have obvious mains noise to remove"
    assert after < 3, f"notch left {after:.1f} dB at 50 Hz"


def test_notch_is_per_channel(humming):
    """Notching the EMG but not the EEG is a normal thing to want."""
    only_emg = preprocess_recording(humming, {"EMG": {"notch": 50}})

    assert _line_noise(only_emg, channel="EMG") < 3
    assert _line_noise(only_emg, channel="EEG") > 10


def test_notch_leaves_neighbouring_frequencies_alone(humming):
    """A notch that ate the surrounding band would take real signal with it."""
    filtered = preprocess_recording(humming, {"EEG": {"notch": 50}})

    check = nyx.check_signals(filtered, preview=None)
    freqs, spectrum = check.previews[0].mean_spectrum("eeg")
    # The theta and delta content the scoring depends on must survive.
    assert np.isfinite(spectrum[(freqs > 1) & (freqs < 20)]).all()
    assert spectrum[(freqs > 1) & (freqs < 20)].max() > spectrum[freqs > 45].max()


def test_notch_is_lazy(humming):
    """Preprocessing must not read or copy the traces until they are asked for."""
    processed = preprocess_recording(humming, {"EEG": {"notch": 50}})
    # Building it is cheap; the object is a spikeinterface recording, not an array.
    assert hasattr(processed.eeg, "get_traces")
    assert processed.eeg.get_num_samples() == humming.eeg.get_num_samples()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_resample_lowers_the_rate(synthetic_recording):
    lower = preprocess_recording(synthetic_recording, {"EEG": {"resample": 64},
                                                       "EMG": {"resample": 64}})

    assert lower.fs == 64
    assert lower.duration == pytest.approx(synthetic_recording.duration, rel=1e-2)


def test_resample_only_one_channel_records_the_two_rates(synthetic_recording):
    mixed = preprocess_recording(synthetic_recording, {"EMG": {"resample": 64}})

    assert mixed.fs == synthetic_recording.fs
    assert mixed.emg_fs == 64
    assert "64" in mixed.describe()


def test_upsampling_is_refused(synthetic_recording):
    """It would add no information, so say so rather than doing it."""
    with pytest.warns(UserWarning, match="above the recording"):
        same = preprocess_recording(synthetic_recording, {"EEG": {"resample": 10_000}})

    assert same.fs == synthetic_recording.fs


def test_channel_level_helper_returns_input_when_nothing_is_set(synthetic_recording):
    assert preprocess_channel(synthetic_recording.eeg, {}) is synthetic_recording.eeg


# ---------------------------------------------------------------------------
# The explicit chain: anything in spikeinterface.preprocessing
# ---------------------------------------------------------------------------


def test_explicit_chain_names_spikeinterface_functions(humming):
    """The whole spre catalogue is reachable without nyx wrapping each one.

    ``ignore_low_freq_error`` is a spikeinterface argument nyx knows nothing
    about, which is the point: keyword arguments pass straight through.
    """
    processed = preprocess_recording(humming, {
        "EEG": {"preprocess": [
            {"notch_filter": {"freq": 50, "q": 30}},
            {"bandpass_filter": {"freq_min": 0.5, "freq_max": 45,
                                 "ignore_low_freq_error": True}},
        ]},
    })

    # Measured as absolute power, not with line_noise_score: that compares a
    # peak to its flanks, which is meaningless once a low-pass has attenuated
    # the flanks harder than the centre.
    assert _power_at(processed, 50.0) < _power_at(humming, 50.0) - 10


def test_chain_order_is_respected(synthetic_recording):
    """Resampling first then filtering is not the same as the reverse, so the
    list order has to be the applied order."""
    first = preprocess_recording(synthetic_recording, {
        "EEG": {"preprocess": [{"resample": {"resample_rate": 64}},
                               {"highpass_filter": {"freq_min": 2.0,
                                                    "ignore_low_freq_error": True}}]},
    })
    assert first.fs == 64


def test_bare_string_step(synthetic_recording):
    processed = preprocess_recording(
        synthetic_recording, {"EEG": {"preprocess": ["common_reference"]}}
    )
    assert processed.eeg.get_num_samples() == synthetic_recording.eeg.get_num_samples()


def test_name_key_form(synthetic_recording):
    processed = preprocess_recording(synthetic_recording, {
        "EEG": {"preprocess": [{"name": "resample", "resample_rate": 64}]},
    })
    assert processed.fs == 64


def test_unknown_step_lists_what_is_available(synthetic_recording):
    with pytest.raises(ValueError, match="notch_filter"):
        preprocess_recording(
            synthetic_recording, {"EEG": {"preprocess": [{"denoise_everything": {}}]}}
        )


def test_malformed_step_is_reported(synthetic_recording):
    with pytest.raises(ValueError, match="Cannot read the preprocessing step"):
        preprocess_recording(synthetic_recording, {"EEG": {"preprocess": [42]}})


def test_explicit_chain_wins_over_shorthands(humming):
    chain = [{"resample": {"resample_rate": 64}}]

    with pytest.warns(UserWarning, match="explicit list wins"):
        both = preprocess_recording(humming, {"EEG": {"notch": 50,
                                                      "preprocess": chain}})
    only_chain = preprocess_recording(humming, {"EEG": {"preprocess": chain}})

    # The list ran, and the ignored notch left no trace: identical signal.
    assert both.fs == 64
    assert np.allclose(both.eeg_trace(), only_chain.eeg_trace())


def test_shorthands_and_explicit_chain_agree(humming):
    """The shorthands are sugar, so both routes must give the same signal."""
    shorthand = preprocess_recording(humming, {"EEG": {"notch": 50,
                                                       "notch_harmonics": False}})
    explicit = preprocess_recording(humming, {
        "EEG": {"preprocess": [{"notch_filter": {"freq": 50, "q": 30}}]},
    })

    assert np.allclose(shorthand.eeg_trace(), explicit.eeg_trace())


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------


def test_scoring_applies_the_params_preprocessing(synthetic_recording, params):
    """score_recording preprocesses once, up front, so every step and figure
    sees the same signal."""
    downsampled = dict(params)
    downsampled["EEG"] = {**params["EEG"], "resample": 64}
    downsampled["EMG"] = {**params["EMG"], "resample": 64}

    result = nyx.score_recording(synthetic_recording, downsampled,
                                 window=(0, 2400), verbose=False)

    assert result.recording.fs == 64
    assert set(result.hypnogram["label"]) <= {"WAKE", "NREM", "REM", "NOSIGNAL"}


def test_scoring_is_unchanged_when_nothing_is_configured(synthetic_recording, params):
    plain = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                verbose=False)
    explicit = dict(params)
    explicit["EEG"] = {**params["EEG"], "notch": None, "resample": None}

    same = nyx.score_recording(synthetic_recording, explicit, window=(0, 2400),
                               verbose=False)
    assert same.staging.stage_durations() == plain.staging.stage_durations()
