"""Synthetic EEG/EMG with a known hypnogram.

Lets the pipeline be tested end to end without shipping recordings. The signal
is built to have exactly the structure nyx looks for:

* EMG amplitude is high in wake and low in sleep, so a threshold separates them;
* NREM EEG is dominated by slow (delta) activity;
* REM EEG is dominated by theta with little delta,

which is what puts NREM and REM in different places in PC space.
"""

from __future__ import annotations

import numpy as np
import pytest

FS = 128.0

# (stage, duration in seconds)
DEFAULT_BOUTS = [
    ("WAKE", 900),
    ("NREM", 700),
    ("REM", 300),
    ("NREM", 600),
    ("REM", 250),
    ("WAKE", 500),
    ("NREM", 500),
    ("REM", 250),
]


def make_signals(bouts=None, fs: float = FS, seed: int = 0):
    """Build (eeg, emg, fs, hypnogram) for a sequence of stage bouts."""
    bouts = bouts or DEFAULT_BOUTS
    rng = np.random.default_rng(seed)

    eeg_parts, emg_parts = [], []
    times, durations, labels = [], [], []
    clock = 0.0

    for stage, seconds in bouts:
        n = int(seconds * fs)
        t = np.arange(n) / fs

        if stage == "WAKE":
            # Broadband, low amplitude EEG; large, variable EMG.
            eeg = 20 * rng.standard_normal(n) + 8 * np.sin(2 * np.pi * 12 * t)
            emg = 60 * rng.standard_normal(n)
        elif stage == "NREM":
            # Strong delta; quiet EMG.
            eeg = 120 * np.sin(2 * np.pi * 1.8 * t) + 15 * rng.standard_normal(n)
            emg = 6 * rng.standard_normal(n)
        elif stage == "REM":
            # Strong theta, almost no delta; quiet EMG (atonia).
            eeg = 90 * np.sin(2 * np.pi * 7.0 * t) + 12 * rng.standard_normal(n)
            emg = 4 * rng.standard_normal(n)
        else:
            raise ValueError(f"unknown stage {stage!r}")

        eeg_parts.append(eeg)
        emg_parts.append(emg)
        times.append(clock)
        durations.append(float(seconds))
        labels.append(stage)
        clock += seconds

    hypnogram = {
        "time": np.array(times, dtype="float64"),
        "duration": np.array(durations, dtype="float64"),
        "label": np.array(labels, dtype="U"),
    }
    return np.concatenate(eeg_parts), np.concatenate(emg_parts), fs, hypnogram


@pytest.fixture(scope="session")
def synthetic():
    """``(eeg, emg, fs, ground_truth_hypnogram)``."""
    return make_signals()


@pytest.fixture(scope="session")
def synthetic_recording(synthetic, tmp_path_factory):
    """The synthetic signals loaded through the real npz reader."""
    from nyx.io import read_recording

    eeg, emg, fs, _ = synthetic
    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez(path, eeg=eeg, emg=emg, fs=fs)
    return read_recording(str(path), name="synthetic")


@pytest.fixture(scope="session")
def params():
    """Parameters matching the synthetic signal's frequency content."""
    return {
        "EMG": {
            "min_freq": 30, "max_freq": 60, "binsize": 4, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": True,
        },
        "EEG": {
            "min_freq": 0.5, "max_freq": 30, "binsize": 2, "overlapratio": 0.5,
            "time_smooth": 4, "scaling": "density", "detrend": "constant",
            "mode": "psd", "scale": "dB", "normalized": True,
        },
        "scoring": {"pc_components": 4, "min_duration": 4},
        "clustering": {
            "method": "kmeans",       # deterministic, so the test does not flake
            "n_clusters": 2,
            "pcs_to_use": [0, 1],
            "use_emg": False,
            "outlier_z_threshold": 6,
        },
    }
