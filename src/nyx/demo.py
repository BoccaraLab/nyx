"""A synthetic recording, so nyx can be tried without downloading anything.

The signal is built to have the structure nyx looks for, and nothing else:

* EMG amplitude is high in wake and low in sleep, so a threshold separates them;
* NREM EEG is dominated by slow (delta) activity;
* REM EEG is dominated by theta with little delta.

That makes it useful for checking an installation, for the tutorial, and for
tests -- but **not** for judging how well nyx works. Real EEG is far messier,
and a method that only ever saw this would be untested. For that, use one of
the public datasets: see :func:`nyx.datasets.list_datasets`.

Usage::

    recording, reference = nyx.demo_recording()
    result = nyx.score_recording(recording, nyx.demo_params(), reference=reference)
    print(result.agreement.summary())
"""

from __future__ import annotations

import numpy as np

from nyx.io.annotations import intervals_from_labels
from nyx.types import Recording

__all__ = [
    "demo_recording",
    "demo_messy_recording",
    "demo_wideband_recording",
    "demo_params",
    "make_demo_signals",
    "DEMO_BOUTS",
]

#: Sampling rate of the synthetic signal, in Hz. Low on purpose: high enough
#: for the bands that matter, low enough to stay fast.
DEMO_FS = 128.0

#: (stage, duration in seconds). Roughly two hours with a plausible structure:
#: a long wake period, then alternating NREM and REM with REM bouts growing.
DEMO_BOUTS: list[tuple[str, float]] = [
    ("WAKE", 900),
    ("NREM", 700),
    ("REM", 300),
    ("NREM", 600),
    ("REM", 250),
    ("WAKE", 500),
    ("NREM", 500),
    ("REM", 250),
]


def make_demo_signals(
    bouts: list[tuple[str, float]] | None = None,
    fs: float = DEMO_FS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Build ``(eeg, emg, fs, hypnogram)`` for a sequence of stage bouts.

    Deterministic for a given ``seed``, so results are reproducible.
    """
    bouts = bouts or DEMO_BOUTS
    rng = np.random.default_rng(seed)

    eeg_parts, emg_parts = [], []
    labels: list[str] = []
    durations: list[float] = []

    for stage, seconds in bouts:
        n = int(seconds * fs)
        t = np.arange(n) / fs

        if stage == "WAKE":
            # Broadband, low amplitude EEG; large and variable EMG.
            eeg = 20 * rng.standard_normal(n) + 8 * np.sin(2 * np.pi * 12 * t)
            emg = 60 * rng.standard_normal(n)
        elif stage == "NREM":
            # Strong delta; quiet EMG.
            eeg = 120 * np.sin(2 * np.pi * 1.8 * t) + 15 * rng.standard_normal(n)
            emg = 6 * rng.standard_normal(n)
        elif stage == "REM":
            # Strong theta, almost no delta; quiet EMG (muscle atonia).
            eeg = 90 * np.sin(2 * np.pi * 7.0 * t) + 12 * rng.standard_normal(n)
            emg = 4 * rng.standard_normal(n)
        else:
            raise ValueError(
                f"Unknown stage {stage!r}. Expected 'WAKE', 'NREM' or 'REM'."
            )

        eeg_parts.append(eeg)
        emg_parts.append(emg)
        labels.append(stage)
        durations.append(float(seconds))

    hypnogram = {
        "time": np.cumsum([0.0] + durations[:-1]).astype("float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U"),
    }
    return np.concatenate(eeg_parts), np.concatenate(emg_parts), fs, hypnogram


def demo_recording(
    seed: int = 0,
    mains_hz: float | None = None,
    artefact_seconds: float = 0.0,
) -> tuple[Recording, dict]:
    """A synthetic :class:`~nyx.types.Recording` and its true hypnogram.

    Parameters
    ----------
    mains_hz
        Add mains interference at this frequency, with harmonics, to both
        channels. Use it to see :func:`nyx.check_signals` detect the hum, and
        a ``"notch"`` in the params remove it.
    artefact_seconds
        Add a stretch of saturated, high-amplitude signal to the start and the
        end, of the kind you get from handling the animal or a loose
        connection. Use it to practise choosing an analysis window. The added
        stretches are labelled ``NOSIGNAL`` in the returned hypnogram.

    Returns
    -------
    (Recording, dict)
        The recording, and the hypnogram it was generated from -- which can be
        passed as ``reference`` to check the scoring against.

    Examples
    --------
    A recording with both problems, as :func:`demo_messy_recording` returns::

        recording, truth = nyx.demo_recording(mains_hz=50, artefact_seconds=600)
    """
    from nyx.io.recordings import _as_single_channel

    eeg, emg, fs, hypnogram = make_demo_signals(seed=seed)
    rng = np.random.default_rng(seed + 1)

    if artefact_seconds > 0:
        eeg, emg, hypnogram = _add_artefacts(
            eeg, emg, hypnogram, fs, artefact_seconds, rng
        )

    if mains_hz:
        eeg = _add_mains(eeg, fs, mains_hz, amplitude=25.0)
        emg = _add_mains(emg, fs, mains_hz, amplitude=18.0)

    return (
        Recording(
            eeg=_as_single_channel(eeg, fs, "EEG"),
            emg=_as_single_channel(emg, fs, "EMG"),
            fs=fs,
            name="demo",
            source_path="<synthetic>",
        ),
        hypnogram,
    )


def demo_messy_recording(seed: int = 0) -> tuple[Recording, dict]:
    """A demo recording that needs both a notch filter and trimming.

    Mains hum at 50 Hz and ten minutes of saturated signal at each end, so the
    signal check has something to find. See ``examples/02_signal_check.ipynb``.
    """
    return demo_recording(seed=seed, mains_hz=50.0, artefact_seconds=600.0)


def _add_mains(trace: np.ndarray, fs: float, freq: float, amplitude: float):
    """Add mains interference: the fundamental plus decaying harmonics.

    Real mains pickup is never a pure sinusoid, which is why the notch filter
    removes harmonics too.
    """
    t = np.arange(len(trace)) / fs
    hum = np.zeros_like(trace)
    for harmonic in (1, 2, 3):
        if freq * harmonic < fs / 2:
            hum += (amplitude / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)
    return trace + hum


def _add_artefacts(eeg, emg, hypnogram, fs, seconds, rng):
    """Bracket the recording with saturated signal, as at the start of many."""
    n = int(seconds * fs)

    def burst(scale):
        # Large, broadband, and clipped -- what saturation actually looks like.
        raw = scale * rng.standard_normal(n)
        return np.clip(raw, -3 * scale, 3 * scale)

    eeg = np.concatenate([burst(600.0), eeg, burst(600.0)])
    emg = np.concatenate([burst(400.0), emg, burst(400.0)])

    times = np.concatenate([[0.0], np.asarray(hypnogram["time"]) + seconds,
                            [float(np.sum(hypnogram["duration"])) + seconds]])
    durations = np.concatenate([[seconds], np.asarray(hypnogram["duration"]), [seconds]])
    labels = np.concatenate([["NOSIGNAL"], np.asarray(hypnogram["label"]), ["NOSIGNAL"]])

    return eeg, emg, {
        "time": times.astype("float64"),
        "duration": durations.astype("float64"),
        "label": labels.astype("U"),
    }


def demo_wideband_recording(seed: int = 0, fs: float = 1500.0, n_channels: int = 4):
    """Wideband channels carrying volume-conducted muscle, for the EMG-like path.

    Returns a multi-channel spikeinterface recording sampled high enough for the
    275-600 Hz band, plus the true hypnogram. Each channel gets its own
    independent brain-band activity, and *shared* high-frequency noise during
    wake -- which is what :func:`nyx.emg_from_lfp` detects: muscle appears on
    every electrode at once, brain activity does not.

    Sampled at 1500 Hz on purpose: the default band needs at least 1200 Hz, and
    a recording downsampled below that is the usual reason the surrogate cannot
    be built.
    """
    from spikeinterface.core import NumpyRecording

    _, _, base_fs, hypnogram = make_demo_signals(seed=seed)
    rng = np.random.default_rng(seed + 7)

    n = int(float(np.sum(hypnogram["duration"])) * fs)
    t = np.arange(n) / fs

    # Muscle tone per sample, from the hypnogram: high in wake, low in sleep.
    tone = np.zeros(n)
    for time, duration, label in zip(
        hypnogram["time"], hypnogram["duration"], hypnogram["label"]
    ):
        start, end = int(time * fs), int((time + duration) * fs)
        tone[start:end] = 1.0 if str(label) == "WAKE" else 0.05

    # One shared muscle waveform, added to every channel -- that is what makes
    # them correlate. Broadband, so it lands inside the 275-600 Hz band.
    muscle = tone * rng.standard_normal(n)

    traces = np.empty((n, n_channels))
    for channel in range(n_channels):
        local = 40 * np.sin(2 * np.pi * (2.0 + channel) * t)
        traces[:, channel] = local + 8 * rng.standard_normal(n) + 30 * muscle

    recording = NumpyRecording([traces], float(fs))
    recording.set_property(
        key="channel_name", values=[f"LFP{i + 1}" for i in range(n_channels)]
    )
    return recording, hypnogram


def demo_params() -> dict:
    """Parameters matched to the demo signal's frequency content.

    Deliberately not one of the shipped species files: the synthetic signal is
    sampled at 128 Hz and has no mains noise, so the real defaults would be
    poorly matched to it.
    """
    return {
        "features": {"method": "spectrogram"},
        "EMG": {
            "min_freq": 30, "max_freq": 60, "binsize": 4, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": "mean",
        },
        "EEG": {
            "min_freq": 0.5, "max_freq": 30, "binsize": 2, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": "mean",
        },
        "scoring": {"pc_components": 4, "min_duration": 4},
        "steps": [
            {"name": "wake_sleep", "method": "emg_threshold"},
            {
                "name": "split_sleep",
                "within": "SLEEP",
                # kmeans rather than hdbscan: deterministic, so the tutorial
                # gives the same answer every time it is run.
                "method": "kmeans",
                "n_clusters": 2,
                "pcs_to_use": [0, 1],
                "outlier_z_threshold": 6,
                "stage_order": ["REM", "NREM"],
            },
        ],
    }


def demo_hypnogram_as_intervals(labels, epoch_length: float) -> dict:
    """Helper for turning a per-epoch label sequence into intervals."""
    return intervals_from_labels(labels, epoch_length)
