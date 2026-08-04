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
