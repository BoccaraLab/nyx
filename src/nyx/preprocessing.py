"""Signal conditioning applied before features are computed.

Currently just notch filtering. Kept separate from :mod:`nyx.features` because
these operate on the raw trace, are optional, and are decided by looking at the
signal (see :func:`nyx.inspect.check_signals`) rather than by the scoring
method.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch

__all__ = ["apply_notch", "line_noise_score", "DEFAULT_MAINS_HZ"]


#: Mains frequency assumed when none is given. 50 Hz covers Europe, Asia,
#: Africa and most of South America; set 60 for North America and Japan.
DEFAULT_MAINS_HZ = 50.0


def apply_notch(
    trace: np.ndarray,
    fs: float,
    freq: float = DEFAULT_MAINS_HZ,
    harmonics: bool = True,
    quality: float = 30.0,
) -> np.ndarray:
    """Notch out mains interference and, optionally, its harmonics.

    Parameters
    ----------
    trace
        1-D signal.
    fs
        Sampling rate in Hz.
    freq
        Mains frequency. Defaults to 50 Hz (Europe); use 60 for North America.
    harmonics
        Also filter 2x, 3x, ... up to just below Nyquist. Mains pickup is rarely
        a pure sinusoid, so the harmonics usually need removing too.
    quality
        Q factor of the notch. Higher is narrower; 30 removes roughly +/-1 Hz at
        50 Hz, which is tight enough to leave neighbouring EMG power intact.

    Returns
    -------
    np.ndarray
        Filtered copy of ``trace``.
    """
    trace = np.asarray(trace, dtype=float)
    nyquist = fs / 2.0
    if not 0 < freq < nyquist:
        raise ValueError(
            f"Notch frequency {freq} Hz must be between 0 and Nyquist ({nyquist} Hz) "
            f"for a {fs} Hz recording."
        )

    targets = [freq]
    if harmonics:
        targets += [freq * k for k in range(2, int(nyquist // freq) + 1)]
    # filtfilt needs some room; a very short trace cannot be filtered stably.
    if trace.size < 3 * 27:
        return trace

    filtered = trace
    for target in targets:
        if target >= nyquist * 0.99:
            continue
        b, a = iirnotch(w0=target / nyquist, Q=quality)
        filtered = filtfilt(b, a, filtered)
    return filtered


def line_noise_score(
    freqs: np.ndarray, spectrum: np.ndarray, freq: float, bandwidth: float = 2.0
) -> float:
    """How far the spectrum sticks up at ``freq`` above its local surroundings.

    Returns the difference, in the spectrum's own units (dB, if the spectrogram
    was computed in dB), between the peak inside ``freq +/- bandwidth`` and the
    median of the flanking bands. Values above roughly 3 dB are worth a notch;
    a clean recording sits near 0.

    Returns ``nan`` if ``freq`` is outside the range covered by ``freqs``.
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    peak_band = np.abs(freqs - freq) <= bandwidth
    flank_band = (np.abs(freqs - freq) > bandwidth) & (np.abs(freqs - freq) <= 4 * bandwidth)
    if not peak_band.any() or not flank_band.any():
        return float("nan")

    return float(spectrum[peak_band].max() - np.median(spectrum[flank_band]))
