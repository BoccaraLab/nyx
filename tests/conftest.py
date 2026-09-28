"""Shared fixtures.

The synthetic signals live in :mod:`nyx.demo` rather than here, so the data the
tests run on is the same data users get from ``nyx.demo_recording()``. That way
the tutorial is covered by the test suite, and there is one generator to keep
correct instead of two.
"""

from __future__ import annotations

import matplotlib

# Draw to a buffer rather than a window, so the figure code is actually
# exercised by the tests instead of failing on a missing display.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nyx.demo import DEMO_FS, demo_params, make_demo_signals  # noqa: E402

FS = DEMO_FS


@pytest.fixture(scope="session")
def synthetic():
    """``(eeg, emg, fs, ground_truth_hypnogram)``."""
    return make_demo_signals()


@pytest.fixture(scope="session")
def synthetic_recording(synthetic, tmp_path_factory):
    """The synthetic signals loaded through the real npz reader.

    Round-tripping through a file rather than using ``nyx.demo_recording()``
    directly means the reader is exercised too.
    """
    from nyx.io import read_recording

    eeg, emg, fs, _ = synthetic
    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez(path, eeg=eeg, emg=emg, fs=fs)
    return read_recording(str(path), name="synthetic")


@pytest.fixture(scope="session")
def params():
    """Parameters matching the synthetic signal's frequency content."""
    return demo_params()


@pytest.fixture(scope="session")
def two_stream_edf(tmp_path_factory):
    """An EDF with EEG1 and EEG2 at 256 Hz and the EMG in a second stream at 128.

    Two streams at different rates, the way a lot of acquisition files are.
    Neo writes almost none of the formats it reads, so this is what the Neo
    path is tested on: read as ``format="neo"`` it goes the same way an Open
    Ephys folder or an Intan file would.
    """
    pyedflib = pytest.importorskip("pyedflib")

    path = tmp_path_factory.mktemp("neo") / "two_streams.edf"
    rates = {"EEG1": 256, "EEG2": 256, "EMG": 128}
    writer = pyedflib.EdfWriter(str(path), len(rates))
    try:
        writer.setSignalHeaders([
            {"label": label, "dimension": "uV", "sample_frequency": fs,
             "physical_min": -500.0, "physical_max": 500.0,
             "digital_min": -32768, "digital_max": 32767,
             "transducer": "", "prefilter": ""}
            for label, fs in rates.items()
        ])
        rng = np.random.default_rng(0)
        writer.writeSamples([rng.normal(0, 50, fs * 10) for fs in rates.values()])
    finally:
        writer.close()
    return str(path)
