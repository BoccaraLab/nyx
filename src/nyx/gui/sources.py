"""nyx's data as things ephyviewer can scroll.

Everything the pipeline produces that varies over time -- the raw traces, the
EMG power that the wake/sleep cut is made on, the principal component scores,
the hypnogram at every stage -- becomes an ephyviewer source here. That is what
makes them scrollable, zoomable and, because a
:class:`~ephyviewer.MainViewer` locks its viewers to one clock, scrollable
*together*: drag the EEG and the EMG power, the PC scores and the hypnogram all
follow.

The alternative is a static picture of the same numbers, which is what a
notebook already gives you.

Two clocks
----------

``Recording.time_slice`` keeps absolute times;
``SpikeInterfaceRecordingSource`` restarts at zero; and the hypnogram is
relative to the window. So the offset is read off the trace source at runtime
-- :func:`trace_sources` returns it -- and everything else is built against
that. Never assume ``window[0]``.
"""

from __future__ import annotations

import numpy as np
from ephyviewer.datasource import (
    InMemoryAnalogSignalSource,
    InMemoryEpochSource,
    SpikeInterfaceRecordingSource,
)

from nyx.gui.hypnogram import to_epoch_dict

__all__ = [
    "trace_sources",
    "power_source",
    "component_source",
    "epoch_source",
    "epoch_sources",
]


def trace_sources(recording):
    """``(eeg_source, emg_source_or_None, t_offset)`` for a recording.

    Two sources, not one: the EMG may be sampled at a different rate from the
    EEG, and a single source cannot carry two rates.
    """
    eeg = SpikeInterfaceRecordingSource(recording=recording.eeg)
    emg = (
        SpikeInterfaceRecordingSource(recording=recording.emg)
        if recording.emg is not None else None
    )
    return eeg, emg, float(getattr(eeg, "t_start", 0.0) or 0.0)


def power_source(emg, t_offset: float = 0.0):
    """The EMG power trace the wake/sleep threshold is drawn on.

    One value per epoch rather than per sample, so ``emg.fs`` is bins per
    second, not the recording's sampling rate.
    """
    if emg is None:
        return None
    power = np.asarray(emg.power, dtype="float32").reshape(-1, 1)
    return InMemoryAnalogSignalSource(
        power, float(emg.fs), float(t_offset), channel_names=["EMG power"]
    )


def component_source(pca, n: int | None = None, t_offset: float = 0.0):
    """Principal component scores over time, so they can be read against the trace.

    ``pca.signal`` is NaN outside the epochs the step covered, which is what a
    reader wants to see: the gaps *are* the epochs this step did not touch.
    """
    if pca is None or getattr(pca, "signal", None) is None:
        return None
    signal = np.asarray(pca.signal, dtype="float32")
    if signal.ndim == 1:
        signal = signal[:, None]
    if n is not None:
        signal = signal[:, :n]
    names = [f"PC{i + 1}" for i in range(signal.shape[1])]
    return InMemoryAnalogSignalSource(
        signal, float(pca.fs), float(t_offset), channel_names=names
    )


def epoch_source(hypnogram, name: str = "hypnogram", t_offset: float = 0.0):
    """One hypnogram as a read-only epoch source."""
    if hypnogram is None:
        return None
    return InMemoryEpochSource(all_epochs=[to_epoch_dict(hypnogram, name, t_offset)])


def epoch_sources(named, t_offset: float = 0.0):
    """Several hypnograms in one source, drawn as separate rows.

    For showing a scoring against the reference, or against what it looked like
    before a rule was applied.
    """
    epochs = [
        to_epoch_dict(hypnogram, name, t_offset)
        for name, hypnogram in named
        if hypnogram is not None
    ]
    return InMemoryEpochSource(all_epochs=epochs) if epochs else None
