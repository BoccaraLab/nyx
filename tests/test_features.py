"""The two time-frequency backends, and that the pipeline runs on either."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.features import compute_time_frequency, feature_method


@pytest.fixture(scope="module")
def tone():
    """Ten seconds of a 6 Hz sine at 128 Hz, with a little noise."""
    fs = 128.0
    t = np.arange(int(10 * fs)) / fs
    rng = np.random.default_rng(0)
    return np.sin(2 * np.pi * 6 * t) + 0.05 * rng.standard_normal(len(t)), fs


SCALOGRAM = {"min_freq": 1.0, "max_freq": 20.0, "freq_resolution": 0.5}
SPECTROGRAM = {
    "min_freq": 1.0, "max_freq": 20.0, "binsize": 2, "overlapratio": 0.5,
    "time_smooth": 0, "scaling": "density", "detrend": "constant",
    "mode": "psd", "scale": "dB", "normalized": False,
}


# ---------------------------------------------------------------------------
# Shape contract: everything downstream works on (Sxx, freqs, times) alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,settings",
                         [("spectrogram", SPECTROGRAM), ("scalogram", SCALOGRAM)])
def test_both_backends_return_the_same_shape_triple(tone, method, settings):
    data, fs = tone
    Sxx, freqs, times = compute_time_frequency(data, fs, settings, method=method)

    assert Sxx.shape == (len(freqs), len(times))
    assert np.all(np.diff(times) > 0)
    # The band of interest has to be covered. It need not be the *whole* range:
    # the scalogram computes only [min_freq, max_freq), while the spectrogram
    # returns everything up to Nyquist and is cut to the band by its caller.
    band = (freqs >= settings["min_freq"]) & (freqs <= settings["max_freq"])
    assert band.sum() > 1
    assert np.isfinite(Sxx[band]).all()


@pytest.mark.parametrize("method,settings",
                         [("spectrogram", SPECTROGRAM), ("scalogram", SCALOGRAM)])
def test_both_backends_find_the_tone(tone, method, settings):
    """The point of either transform: power concentrated at the real frequency."""
    data, fs = tone
    Sxx, freqs, _ = compute_time_frequency(data, fs, settings, method=method)

    peak = freqs[np.argmax(Sxx.mean(axis=1))]
    assert abs(peak - 6.0) < 1.0, f"{method} put the peak at {peak:.1f} Hz"


def test_the_scalogram_resolves_time_far_more_finely(tone):
    """One column per sample rather than per epoch -- which is why it costs, and
    why the cluster plots decimate."""
    data, fs = tone
    _, _, wavelet_times = compute_time_frequency(data, fs, SCALOGRAM, method="scalogram")
    _, _, fourier_times = compute_time_frequency(data, fs, SPECTROGRAM, method="spectrogram")

    assert len(wavelet_times) > 20 * len(fourier_times)


def test_frequency_resolution_sets_the_number_of_rows(tone):
    data, fs = tone
    fine = compute_time_frequency(data, fs, {**SCALOGRAM, "freq_resolution": 0.25},
                                  method="scalogram")[1]
    coarse = compute_time_frequency(data, fs, {**SCALOGRAM, "freq_resolution": 1.0},
                                    method="scalogram")[1]

    assert len(fine) == pytest.approx(4 * len(coarse), rel=0.1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_spectrogram_is_the_default():
    assert feature_method({}) == "spectrogram"
    assert feature_method({"features": {"method": "scalogram"}}) == "scalogram"


def test_a_scalogram_section_missing_its_settings_says_which(tone):
    data, fs = tone
    with pytest.raises(KeyError, match="freq_resolution"):
        compute_time_frequency(data, fs, {"min_freq": 1, "max_freq": 20},
                               method="scalogram")


def test_unknown_method_is_rejected(tone):
    data, fs = tone
    with pytest.raises(ValueError, match="Unknown feature method"):
        compute_time_frequency(data, fs, SCALOGRAM, method="hilbert")


def test_normalisation_applies_to_the_scalogram_too(tone):
    data, fs = tone
    plain = compute_time_frequency(data, fs, SCALOGRAM, method="scalogram")[0]
    centred = compute_time_frequency(data, fs, {**SCALOGRAM, "normalized": "mean"},
                                     method="scalogram")[0]

    assert abs(centred.mean()) < abs(plain.mean())
    assert np.allclose(plain - plain.mean(), centred)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_scoring_runs_on_the_scalogram_backend(synthetic_recording):
    """A short window: the transform holds the whole signal in memory at once."""
    params = {
        "features": {"method": "scalogram"},
        "EEG": {"min_freq": 0.5, "max_freq": 20, "freq_resolution": 0.5,
                "time_smooth": 4},
        "EMG": {"min_freq": 30, "max_freq": 60, "freq_resolution": 2,
                "time_smooth": 4},
        "scoring": {"pc_components": 4, "min_duration": 4},
        # A fixed cluster count, as params/mouse_scalogram.json uses: hdbscan's
        # min_cluster_size is in points, and the scalogram produces one per
        # sample rather than one per epoch.
        "clustering": {"method": "gmm", "n_clusters": 2, "pcs_to_use": [0, 1]},
    }

    result = nyx.score_recording(synthetic_recording, params, window=(0, 1200),
                                 verbose=False)

    assert set(result.hypnogram["label"]) <= {"WAKE", "NREM", "REM", "NOSIGNAL"}
    assert result.pca.fs > result.recording.fs / 4, "scalogram bins are near-sample-rate"
    # It resolved both sleep stages rather than collapsing them.
    assert {"NREM", "REM"} <= set(result.hypnogram["label"])


def test_clusters_with_no_stage_name_are_left_unscored_and_warned_about(
    synthetic_recording,
):
    """They used to be folded into NREM without a word.

    This bites hard on the scalogram, where hdbscan's min_cluster_size is in
    points and there are hundreds of times more of them: a night can fragment
    into dozens of clusters, and everything past the second name silently
    became NREM.
    """
    params = {**nyx.demo_params(), "clustering": {"method": "kmeans", "n_clusters": 5}}

    with pytest.warns(UserWarning, match="no stage name"):
        result = nyx.score_recording(synthetic_recording, params,
                                     window=(0, 2400), verbose=False)

    durations = result.staging.stage_durations()
    assert durations.get("NOSIGNAL", 0) > 0, "the unnamed clusters must be unscored"
    # The two named clusters are still named; the extras keep their real ids in
    # stage_labels, which is what a following step would work from.
    assert {"C2", "C3", "C4"} <= set(result.staging.stage_labels)


def test_cluster_plot_decimates_without_changing_the_clustering(synthetic_recording):
    """The subsample is only for drawing; every point was clustered."""
    from nyx.report import _subsample, plot_clusters

    result = nyx.score_recording(synthetic_recording, nyx.demo_params(),
                                 window=(0, 2400), verbose=False)
    n = len(result.clusters.features_scaled)

    assert len(_subsample(n, 50)) == 50
    assert np.array_equal(_subsample(n, 50), _subsample(n, 50)), "must be reproducible"
    assert len(_subsample(n, n + 1)) == n

    ax = plot_clusters(result, max_points=50)
    drawn = sum(len(collection.get_offsets()) for collection in ax.collections)
    assert drawn <= 50 + len(result.clusters.centers)
