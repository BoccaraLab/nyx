"""An EMG-like signal built from wideband electrophysiology.

Not every preparation has an EMG channel. Where the recording does carry
several wideband channels -- LFP probes, ECoG screws, anything sampled high
enough to see a few hundred Hz -- muscle activity can be recovered from them
instead, and that surrogate is far better than scoring with no EMG at all.

The idea is that muscle potentials are *volume-conducted*: a contraction shows
up on every electrode at once, while genuine brain activity is local and
uncorrelated between distant sites. So band-pass well above the physiological
EEG range, then ask how much the channels agree with each other. High
correlation means muscle; low means brain. This is the method introduced by
Watson et al. (2016) as ``EMGFromLFP``.

Usage::

    emg = nyx.emg_from_lfp(recording.eeg, channels=["LFP1", "LFP2", "ECoG1"])
    result = nyx.score_recording(recording, params, emg=emg)

or, more usually, from separate files::

    import spikeinterface as si
    emg = nyx.emg_from_lfp([si.load("lfp/"), si.load("ecog/")])

What it needs
-------------

* **At least two channels**, and the further apart the better -- two adjacent
  contacts on the same probe share brain signal as well as muscle, so they
  correlate for the wrong reason.
* **A sampling rate above twice ``band[1]``.** The default 275-600 Hz band
  needs at least 1200 Hz. A 200 Hz downsample cannot carry it, and this is the
  usual reason the function refuses to run.

The output is an :class:`~nyx.types.EmgFeatures` scaled to [0, 1], exactly like
:func:`nyx.compute_emg_features`, so everything downstream -- the threshold,
the figures, ``use_emg`` in a clustering step -- works unchanged.
"""

from __future__ import annotations

import numpy as np

from nyx.types import EmgFeatures

__all__ = ["emg_from_lfp", "DEFAULT_BAND"]

#: Frequency band used to isolate muscle from brain. Above the physiological
#: EEG range and below where amplifier noise usually takes over.
DEFAULT_BAND = (275.0, 600.0)


def emg_from_lfp(
    recordings,
    channels=None,
    band: tuple[float, float] = DEFAULT_BAND,
    bin_seconds: float = 0.1,
    filter_order: int = 4,
    return_in_uV: bool = False,
) -> EmgFeatures:
    """Build an EMG-like power trace from cross-channel correlation.

    Parameters
    ----------
    recordings
        A spikeinterface recording, a list of them, or a plain
        ``(n_channels, n_samples)`` array. Several recordings are stacked, which
        is how you combine an LFP probe with ECoG screws stored separately; they
        must share a sampling rate and length.
    channels
        Channel names or positions to use. ``None`` uses all of them. Prefer
        channels that are far apart: neighbours on one probe share brain signal
        too, and correlate for the wrong reason.
    band
        Band-pass applied before correlating, in Hz. The default 275-600 Hz sits
        above physiological EEG and below the amplifier noise floor.
    bin_seconds
        Width of the bins the correlation is computed in. 0.1 s is the default;
        shorter bins are noisier, longer ones blur brief movements.
    filter_order
        Butterworth order for the band-pass.

    Returns
    -------
    EmgFeatures
        ``power`` is the mean off-diagonal correlation per bin, scaled to
        [0, 1]. ``spectrogram`` is that trace as a single row, so the figures
        that expect an EMG spectrogram still draw.

    Raises
    ------
    ValueError
        If fewer than two channels are given, or the sampling rate cannot carry
        ``band`` -- the usual cause being a recording downsampled to 200 Hz or
        so, which no longer contains the frequencies this depends on.
    """
    from scipy.signal import butter, sosfiltfilt
    from sklearn.preprocessing import MinMaxScaler

    traces, fs = _stack(recordings, channels, return_in_uV=return_in_uV)
    n_channels = traces.shape[0]

    if n_channels < 2:
        raise ValueError(
            f"An EMG-like signal needs at least two channels to correlate; got "
            f"{n_channels}. It measures how much the channels agree with each "
            f"other, which is undefined for one."
        )

    low, high = float(band[0]), float(band[1])
    nyquist = fs / 2.0
    if low >= nyquist:
        raise ValueError(
            f"The {low:g}-{high:g} Hz band is above this recording's Nyquist "
            f"frequency ({nyquist:g} Hz, from {fs:g} Hz sampling), so the "
            f"frequencies this depends on are not in the data. Load the "
            f"wideband recording rather than a downsampled copy, or lower "
            f"band= if you know muscle shows up further down."
        )
    high = min(high, nyquist - 1e-3)

    sos = butter(filter_order, [low / nyquist, high / nyquist],
                 btype="band", output="sos")
    filtered = sosfiltfilt(sos, traces, axis=-1)

    window = max(int(bin_seconds * fs), 2)
    n_bins = filtered.shape[1] // window
    if n_bins < 2:
        raise ValueError(
            f"{filtered.shape[1] / fs:.1f}s of signal is not enough for "
            f"{bin_seconds:g}s bins. Lower bin_seconds or use a longer window."
        )
    binned = filtered[:, : n_bins * window].reshape(n_channels, n_bins, window)

    # Mean of the off-diagonal correlations: how much the channels agree.
    # Muscle is volume-conducted and shows up on all of them at once; brain
    # activity is local, so it does not.
    lower = np.tril(np.ones((n_channels, n_channels)), k=-1)
    n_pairs = lower.sum()

    score = np.empty(n_bins, dtype=float)
    for b in range(n_bins):
        correlations = np.corrcoef(binned[:, b, :])
        score[b] = np.nansum(correlations * lower) / n_pairs

    # A bin where a channel is flat gives NaN correlations. Treat those as no
    # detectable muscle rather than dropping them, which would shift every
    # later bin in time.
    score = np.nan_to_num(score, nan=float(np.nanmin(score)) if np.isfinite(score).any() else 0.0)

    power = MinMaxScaler().fit_transform(score.reshape(-1, 1)).ravel()
    times = (np.arange(n_bins) + 0.5) * bin_seconds

    return EmgFeatures(
        # One row, at the centre of the band: the figures expect a spectrogram,
        # and this trace does not come from one.
        spectrogram=power[np.newaxis, :],
        freqs=np.array([(low + high) / 2.0]),
        times=times,
        power=power,
        fs=1.0 / bin_seconds,
    )


def _stack(recordings, channels, return_in_uV: bool = False):
    """Gather the requested channels into ``(n_channels, n_samples)`` and a rate."""
    if isinstance(recordings, np.ndarray):
        raise TypeError(
            "emg_from_lfp needs the sampling rate, which a bare array does not "
            "carry. Wrap it first: "
            "spikeinterface.core.NumpyRecording([traces.T], fs)."
        )

    if not isinstance(recordings, (list, tuple)):
        recordings = [recordings]

    rates = {float(rec.get_sampling_frequency()) for rec in recordings}
    if len(rates) > 1:
        raise ValueError(
            f"The recordings have different sampling rates ({sorted(rates)}). "
            f"Resample them to a common rate first -- spikeinterface's "
            f"`resample` does it lazily."
        )
    fs = rates.pop()

    parts = []
    for rec in recordings:
        selected = rec if channels is None else rec.select_channels(
            [_resolve(rec, spec) for spec in _for(rec, channels)]
        )
        parts.append(selected.get_traces(return_in_uV=return_in_uV).T.astype(float))

    lengths = {part.shape[1] for part in parts}
    if len(lengths) > 1:
        shortest = min(lengths)
        parts = [part[:, :shortest] for part in parts]

    return np.vstack(parts), fs


def _for(rec, channels):
    """The subset of ``channels`` this recording actually has.

    Combining an LFP folder with an ECoG folder means naming channels from
    both, and each recording only holds its own.
    """
    available = {str(cid) for cid in rec.get_channel_ids()}
    try:
        names = {str(n) for n in rec.get_property("channel_name")}
    except Exception:  # noqa: BLE001 - the property is optional
        names = set()

    wanted = [
        spec for spec in channels
        if isinstance(spec, (int, np.integer))
        or str(spec) in available or str(spec) in names
    ]
    if not wanted:
        raise KeyError(
            f"None of {list(channels)} are in this recording. "
            f"Available: {sorted(available | names)}."
        )
    return wanted


def _resolve(rec, spec):
    from nyx.io.recordings import _resolve_channel

    return _resolve_channel(rec, spec)
