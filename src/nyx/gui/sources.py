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
    "autoscale",
    "colour_channels",
    "fix_range",
    "stream_source",
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


class _NamedRecordingSource(SpikeInterfaceRecordingSource):
    """Labels each trace ``name`` rather than by id.

    Raw acquisition files number their channels ``0``, ``1``, ``2``... and keep
    the meaningful name -- ``EMG``, ``CH12``, ``A-031`` -- in a property, which
    is the one worth reading when deciding which trace is which.
    """

    def __init__(self, recording, names):
        super().__init__(recording=recording)
        self._names = [str(n) for n in names]

    def get_channel_name(self, chan=0):
        return self._names[chan]


def stream_source(recording, names=None):
    """Every channel of a multi-channel recording, for looking at before picking.

    ``names`` defaults to the ids. Unlike :func:`trace_sources`, nothing has
    been chosen yet: this is the whole stream.
    """
    if names is None:
        names = [str(c) for c in recording.get_channel_ids()]
    return _NamedRecordingSource(recording, names)


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


def autoscale(viewer):
    """Scale a raw trace to its own amplitude, once, when it first appears.

    ephyviewer opens every trace on a fixed range, which for a recording in
    volts rather than microvolts is a flat line down the middle. Auto-scaling
    reads the amplitude off the data instead.

    It has to fetch a chunk first: ``auto_scale`` measures what the viewer has
    already drawn, so calling it on a viewer that has never refreshed measures
    nothing.

    Only for signals whose units are unknown. The EMG power is scaled to
    ``[0, 1]`` and the component scores have a meaningful spread of their own;
    both are pinned with :func:`fix_range` instead.
    """
    try:
        viewer.refresh()
        viewer.auto_scale()
    except Exception:  # noqa: BLE001 - not every viewer scales
        pass
    return viewer


def colour_channels(viewer, cmap: str = "tab10"):
    """Give each channel of a trace viewer its own colour.

    ephyviewer draws every channel in the same green, which is fine for one
    trace and makes a stack of them hard to tell apart. This is what its
    "automatic color" button does, done up front.
    """
    import matplotlib

    colours = matplotlib.colormaps[cmap]
    n = viewer.source.nb_channel
    for i in range(n):
        rgb = colours(i % colours.N)[:3]
        viewer.by_channel_params[f"ch{i}", "color"] = [int(255 * v) for v in rgb]
    return viewer


def fix_range(viewer, low: float, high: float):
    """Pin a trace viewer's y range instead of letting it auto-scale."""
    try:
        viewer.params["ylim_min"] = float(low)
        viewer.params["ylim_max"] = float(high)
        for i in range(viewer.source.nb_channel):
            viewer.by_channel_params[f"ch{i}", "gain"] = 1.0
            viewer.by_channel_params[f"ch{i}", "offset"] = 0.0
    except Exception:  # noqa: BLE001 - parameter names vary between releases
        pass
    return viewer
