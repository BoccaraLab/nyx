"""ephyviewer views, adapted for sleep scoring.

Both classes here re-apply, as subclasses of upstream ephyviewer, changes that
previously lived in a fork of it (``BoccaraLab/ephyviewer@sleepscoring``). They
have to be subclasses: PyPI rejects direct git-URL dependencies, so
``pip install "nyx-sleep[gui]"`` could never resolve a fork. Upstream exposes
``_default_params``, ``_default_by_channel_params`` and ``_ControllerClass`` as
class attributes, which is most of what is needed.

:class:`NyxTimeFreqViewer`
    A Morlet scalogram in **dB**, optionally smoothed in time and z-scored,
    with a two-ended colour limit. Upstream plots ``abs(wt)`` on a ``[0, clim]``
    scale, which cannot show dB at all -- dB values are negative.

    This is not a cosmetic change: ``nyx.features`` already computes its
    scalogram in dB with the same Gaussian time-smoothing, and its Morlet code
    is a port of ephyviewer's. Making the viewer match means what you scroll
    through is the transform the scoring actually used, rather than a
    lookalike.

:class:`NyxEpochEncoder`
    Adds two kinds of navigation to the hypnogram editor. *State* jumps to the
    next label change. *To curate* jumps to the next epoch worth a second
    look -- and rather than hardcoding what that means, it asks
    :func:`nyx.flag_rules` which epochs the postprocessing rules would rewrite.
    The fork hardcoded "shorter than 2.5 s, or REM after WAKE", which is
    exactly ``min_duration`` and ``rem_after_wake``; driving it from the rules
    means the threshold comes from the params that were actually used, and a
    rule added to nyx gets curation navigation for free.
"""

from __future__ import annotations

import copy

import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.stats
from ephyviewer.myqt import QT
from ephyviewer.spectrogramviewer import SpectrogramViewer, SpectrogramWorker
from ephyviewer.timefreqviewer import (
    TimeFreqViewer,
    TimeFreqViewer_ParamController,
    TimeFreqWorker,
    generate_wavelet_fourier,
)
from ephyviewer.timefreqviewer import (
    default_by_channel_params as upstream_channel_params,
)
from ephyviewer.timefreqviewer import default_params as upstream_params
from scipy.ndimage import gaussian_filter

from ephyviewer import EpochEncoder  # isort: skip

__all__ = [
    "NyxTimeFreqViewer",
    "NyxSpectrogramViewer",
    "NyxEpochEncoder",
    "timefreq_params_from",
    "spectrogram_params_from",
    "make_timefreq_viewer",
]

def _with_nyx_timefreq_params() -> list:
    """Upstream's parameter tree, plus the three nyx adds.

    Derived rather than transcribed. Upstream's list carries settings the
    viewer reads directly -- ``xratio`` and ``xsize`` among them -- and a
    hand-written copy that misses one fails at the first repaint rather than at
    import, which is a bad way to find out.
    """
    params = copy.deepcopy(upstream_params)
    for entry in params:
        if entry.get("name") == "timefreq":
            entry["children"] = list(entry["children"]) + [
                {"name": "decibel", "type": "bool", "value": True},
                {"name": "smoothing_length", "type": "float", "value": 0.0,
                 "step": 0.1},
                {"name": "zscore", "type": "bool", "value": False},
            ]
    return params


def _with_two_ended_clim() -> list:
    """Upstream's per-channel parameters, with ``clim`` split in two.

    Upstream plots ``abs(wt)`` on ``[0, clim]``. dB values are negative, so a
    single upper limit cannot show them at all.
    """
    params = []
    for entry in copy.deepcopy(upstream_channel_params):
        if entry.get("name") == "clim":
            params.append(
                {"name": "clim_min", "type": "float", "value": 0.0, "step": 0.5}
            )
            params.append(
                {"name": "clim_max", "type": "float", "value": 30.0, "step": 0.5}
            )
        else:
            params.append(entry)
    return params


def _emit(signal, *args) -> None:
    """Emit, unless the viewer on the other end has already been destroyed.

    Closing a viewer tells its threads to quit, but a request already in
    flight still runs to completion -- and by then the C++ object it reports
    to may be gone, which raises rather than being ignored.
    """
    try:
        signal.emit(*args)
    except RuntimeError:
        pass


NYX_TIMEFREQ_PARAMS = _with_nyx_timefreq_params()
NYX_TIMEFREQ_CHANNEL_PARAMS = _with_two_ended_clim()


def timefreq_params_from(params: dict, channel: str = "EEG") -> dict:
    """Viewer settings matching the params a recording was scored with.

    So that the spectrogram on screen is the transform the scoring used.
    ``time_smooth`` and ``f0`` are read from the scalogram settings when the
    params declare that backend; the spectrogram backend has no ``f0``, so its
    default is left alone.
    """
    section = params.get(channel, {}) or {}
    settings = {
        "f_start": float(section.get("min_freq", 0.5)),
        "f_stop": float(section.get("max_freq", 40.0)),
    }
    if section.get("freq_resolution"):
        settings["deltafreq"] = float(section["freq_resolution"])
    if section.get("f0"):
        settings["f0"] = float(section["f0"])
    if section.get("time_smooth"):
        settings["smoothing_length"] = float(section["time_smooth"])
    return settings


class NyxTimeFreqWorker(TimeFreqWorker):
    """Upstream's worker, with the scalogram converted to dB and smoothed.

    ``on_request_data`` is copied from upstream rather than extended, because
    the transform has to happen *before* the map is cut to ``plot_length``:
    smoothing after the cut would reflect at the edge instead of using the real
    samples beyond it, leaving a visible band at the right-hand side of every
    redraw. The lines that differ from upstream are marked.
    """

    def on_request_data(self, chan, t, t_start, t_stop, visible_channels, worker_params):
        if chan != self.chan or not visible_channels[chan]:
            return
        if self.viewer.t != t:
            return  # the viewer has moved on already

        ds_ratio = worker_params["downsample_ratio"]
        sig_chunk_size = worker_params["sig_chunk_size"]
        filter_sos = worker_params["filter_sos"]
        wavelet_fourrier = worker_params["wavelet_fourrier"]
        plot_length = worker_params["plot_length"]
        # -- nyx
        decibel = worker_params.get("decibel", True)
        smoothing_length = worker_params.get("smoothing_length", 0.0)
        zscore = worker_params.get("zscore", False)
        sub_sample_rate = worker_params["sub_sample_rate"]

        i_start = self.source.time_to_index(t_start)
        if ds_ratio > 1:
            i_start = i_start - (i_start % ds_ratio)
        i_start = max(0, min(i_start, self.source.get_length()))
        if ds_ratio > 1:
            i_start = i_start - (i_start % ds_ratio)

        i_stop = min(i_start + sig_chunk_size, self.source.get_length())

        sigs_chunk = self.source.get_chunk(i_start=i_start, i_stop=i_stop)
        # -- nyx: some readers hand back float64 or int16; the FFT below wants
        #    one dtype and the filter refuses integers.
        if sigs_chunk.dtype != "float32":
            sigs_chunk = sigs_chunk.astype("float32")
        sig = sigs_chunk[:, chan]

        if ds_ratio > 1:
            small_sig = scipy.signal.sosfiltfilt(filter_sos, sig)
            small_sig = small_sig[::ds_ratio].copy()  # ensure continuity
        else:
            small_sig = sig.copy()

        left_pad = 0
        if small_sig.shape[0] != wavelet_fourrier.shape[0]:
            padded = np.zeros(wavelet_fourrier.shape[0], dtype=small_sig.dtype)
            left_pad = wavelet_fourrier.shape[0] - small_sig.shape[0]
            padded[: small_sig.shape[0]] = small_sig
            small_sig = padded

        small_sig -= small_sig.mean()  # avoid border effects

        small_sig_f = scipy.fftpack.fft(small_sig)
        if small_sig_f.shape[0] != wavelet_fourrier.shape[0]:
            return
        wt_tmp = scipy.fftpack.ifft(
            small_sig_f[:, np.newaxis] * wavelet_fourrier, axis=0
        )
        wt = scipy.fftpack.fftshift(wt_tmp, axes=[0])
        wt = np.abs(wt).astype("float32")

        # -- nyx: dB, as nyx.features computes it. The floor keeps log10 off
        #    zero, which happens on a flat or disconnected channel.
        if decibel:
            wt = 10 * np.log10(np.maximum(wt, 1e-12))

        if left_pad > 0:
            wt = wt[:-left_pad]

        # -- nyx: smooth in time, before the cut to plot_length
        if smoothing_length:
            wt = gaussian_filter(
                wt, sigma=[smoothing_length * sub_sample_rate, 0], mode="reflect"
            )
        if zscore:
            wt = scipy.stats.zscore(wt, axis=1)

        wt_map = wt[:plot_length]

        t1 = self.source.index_to_time(i_start)
        t2 = self.source.index_to_time(i_start + wt_map.shape[0] * ds_ratio)
        _emit(self.data_ready, chan, t, t_start, t_stop, t1, t2, wt_map)


class NyxTimeFreqController(TimeFreqViewer_ParamController):
    """Auto-scaling for a colour limit that has two ends rather than one."""

    def clim_zoom(self, factor):
        self.viewer.by_channel_params.blockSignals(True)
        for parameter in self.viewer.by_channel_params.children():
            for name in ("clim_min", "clim_max"):
                parameter.param(name).setValue(parameter.param(name).value() * factor)
        self.viewer.by_channel_params.blockSignals(False)
        self.some_clim_changed.emit()

    def compute_auto_clim(self):
        self.viewer.by_channel_params.blockSignals(True)
        highs, lows = [], []
        visible, = np.nonzero(self.visible_channels)

        for chan in visible:
            wt_map = self.viewer.last_wt_maps.get(chan)
            if wt_map is None:
                continue
            high, low = float(np.max(wt_map)), float(np.min(wt_map))
            if self.viewer.params["scale_mode"] == "by_channel":
                self.viewer.by_channel_params[f"ch{chan}", "clim_max"] = high
                self.viewer.by_channel_params[f"ch{chan}", "clim_min"] = low
            else:
                highs.append(high)
                lows.append(low)

        if self.viewer.params["scale_mode"] == "same_for_all" and highs:
            for chan in visible:
                self.viewer.by_channel_params[f"ch{chan}", "clim_max"] = max(highs)
                self.viewer.by_channel_params[f"ch{chan}", "clim_min"] = min(lows)

        self.viewer.by_channel_params.blockSignals(False)
        self.some_clim_changed.emit()


class NyxTimeFreqViewer(TimeFreqViewer):
    """A Morlet scalogram in dB, matching what nyx computed.

    Slower than :class:`~ephyviewer.SpectrogramViewer`, which is an ordinary
    short-time Fourier transform -- a wavelet transform per redraw is real
    work. It is also far easier to read on sleep data, which is why both are
    offered and this one is the default when the params say ``scalogram``.
    """

    _default_params = NYX_TIMEFREQ_PARAMS
    _default_by_channel_params = NYX_TIMEFREQ_CHANNEL_PARAMS
    _ControllerClass = NyxTimeFreqController

    def __init__(self, **kargs):
        # Copied from TimeFreqViewer.__init__ so the worker class can be
        # swapped. Upstream constructs TimeFreqWorker by name rather than
        # holding it as a class attribute; a two-line `_WorkerClass` hook there
        # would make this whole method unnecessary.
        from ephyviewer.base import BaseMultiChannelViewer

        BaseMultiChannelViewer.__init__(self, **kargs)

        self.make_params()

        self.by_channel_params.blockSignals(True)
        for c in range(self.source.nb_channel):
            self.by_channel_params["ch" + str(c), "visible"] = c == 0
        self.by_channel_params.blockSignals(False)

        self.make_param_controller()
        self.params_controller.some_clim_changed.connect(self.refresh)

        self.set_layout()
        self.change_color_scale()
        self.create_grid()
        self.initialize_time_freq()

        self.last_wt_maps = {}
        self.threads = []
        self.timefreq_makers = []
        for c in range(self.source.nb_channel):
            thread = QT.QThread(parent=self)
            self.threads.append(thread)
            worker = NyxTimeFreqWorker(self.source, self, c)  # <- the one change
            self.timefreq_makers.append(worker)
            worker.moveToThread(thread)
            thread.start()
            worker.data_ready.connect(self.on_data_ready)
            self.request_data.connect(worker.on_request_data)

        self.params.param("xsize").setLimits((0, np.inf))

    # -- the three extra worker parameters ----------------------------------

    def initialize_time_freq(self):
        tfr = self.params.param("timefreq")
        sample_rate = self.source.sample_rate

        wanted = tfr["f_stop"] * 4
        sub_rate = wanted if wanted < sample_rate else sample_rate

        d = self.worker_params = {}
        d["wanted_size"] = self.params["xsize"]
        length = d["len_wavelet"] = int(
            2 ** np.ceil(np.log(d["wanted_size"] * sub_rate) / np.log(2))
        )
        d["sig_chunk_size"] = d["wanted_size"] * self.source.sample_rate
        d["downsample_ratio"] = int(np.ceil(d["sig_chunk_size"] / length))
        d["sig_chunk_size"] = d["downsample_ratio"] * length
        d["sub_sample_rate"] = self.source.sample_rate / d["downsample_ratio"]
        d["plot_length"] = int(d["wanted_size"] * d["sub_sample_rate"])
        d["wavelet_fourrier"] = generate_wavelet_fourier(
            d["len_wavelet"], tfr["f_start"], tfr["f_stop"], tfr["deltafreq"],
            d["sub_sample_rate"], tfr["f0"], tfr["normalisation"],
        )
        d["decibel"] = tfr["decibel"]
        d["smoothing_length"] = tfr["smoothing_length"]
        d["zscore"] = tfr["zscore"]

        if d["downsample_ratio"] > 1:
            d["filter_sos"] = scipy.signal.cheby1(
                8, 0.05, 0.8 / d["downsample_ratio"], output="sos"
            )
        else:
            d["filter_sos"] = None

    def on_data_ready(self, chan, t, t_start, t_stop, t1, t2, wt_map):
        if not self.params_controller.visible_channels[chan]:
            return
        if self.images[chan] is None:
            return

        self.last_wt_maps[chan] = wt_map
        f_start = self.params["timefreq", "f_start"]
        f_stop = self.params["timefreq", "f_stop"]

        image = self.images[chan]
        low = self.by_channel_params[f"ch{chan}", "clim_min"]
        high = self.by_channel_params[f"ch{chan}", "clim_max"]
        image.setImage(wt_map, lut=self.lut, levels=[low, high])
        image.setRect(QT.QRectF(t1, f_start, t2 - t1, f_stop - f_start))

        self.vlines[chan].setPos(t)
        self.vlines[chan].setPen(self.params["vline_color"])

    # -- convenience --------------------------------------------------------

    def apply_settings(self, settings: dict) -> None:
        """Set scalogram parameters from :func:`timefreq_params_from`."""
        group = self.params.param("timefreq")
        for name, value in settings.items():
            try:
                group.param(name).setValue(value)
            except Exception:  # noqa: BLE001 - an unknown key is not fatal
                continue
        self.initialize_time_freq()
        self.refresh()


class NyxEpochEncoder(EpochEncoder):
    """The hypnogram editor, with somewhere to go.

    Scrolling a night of sleep one screen at a time to find the parts worth
    checking is most of the work of curating a hypnogram. Two shortcuts do that
    walking instead:

    ``alt`` + left / right
        The previous or next *change* of stage.
    ``ctrl`` + left / right
        The previous or next epoch a postprocessing rule objects to. Those
        epochs are also drawn faded, so they are visible without navigating.

    Which epochs those are comes from :func:`nyx.flag_rules`, given the rules
    the run used -- so the duration threshold is the params' and not a constant,
    and a rule added to :data:`nyx.postprocess.RULES` is picked up here without
    any change.
    """

    def __init__(self, rules=None, **kargs):
        self._rules = list(rules or [])
        self._flagged: set[int] = set()
        super().__init__(**kargs)
        self._add_navigation()
        self.refresh_flags()

    # -- setup -------------------------------------------------------------

    def _add_navigation(self) -> None:
        self.toolbar.addSeparator()

        for text, slot, key, tip in (
            ("< State", self.go_to_previous_state, "Alt+Left",
             "The previous change of stage"),
            ("State >", self.go_to_next_state, "Alt+Right",
             "The next change of stage"),
            ("< To check", self.go_to_previous_flagged, "Ctrl+Left",
             "The previous epoch a postprocessing rule would rewrite"),
            ("To check >", self.go_to_next_flagged, "Ctrl+Right",
             "The next epoch a postprocessing rule would rewrite"),
        ):
            action = self.toolbar.addAction(text, slot)
            action.setShortcut(QT.QKeySequence(key))
            action.setToolTip(f"{tip}  ({key})")

    def set_rules(self, rules) -> None:
        self._rules = list(rules or [])
        self.refresh_flags()
        self.refresh()

    # -- flagging ----------------------------------------------------------

    def refresh_flags(self) -> None:
        """Recompute which epochs a rule would rewrite.

        Asks rather than tells: :func:`nyx.flag_rules` reports what each rule
        *would* change without changing it, which is the difference between
        curation and automatic postprocessing.
        """
        from nyx.postprocess import flag_rules

        self._flagged = set()
        self._flag_reasons = {}
        if not self._rules:
            return

        hypnogram = {
            "time": np.asarray(self.source.ep_times, dtype="float64"),
            "duration": np.asarray(self.source.ep_durations, dtype="float64"),
            "label": np.asarray(self.source.ep_labels, dtype="U16"),
        }
        if len(hypnogram["time"]) == 0:
            return

        try:
            flags = flag_rules(hypnogram, self._rules)
        except Exception:  # noqa: BLE001 - navigation is not worth a crash
            return

        for name, mask in flags.items():
            for index in np.nonzero(np.asarray(mask))[0]:
                self._flagged.add(int(index))
                self._flag_reasons.setdefault(int(index), []).append(name)

    def flagged_indices(self) -> set[int]:
        """Indices of the epochs a rule objects to."""
        return set(self._flagged)

    def flag_reasons(self, index: int) -> list[str]:
        """Which rules objected to this epoch."""
        return list(getattr(self, "_flag_reasons", {}).get(int(index), []))

    def on_data_ready(self, t_start, t_stop, visibles, data):
        super().on_data_ready(t_start, t_stop, visibles, data)
        if not self._flagged:
            return

        # Fade the flagged epochs, so they can be seen without navigating to
        # them. RectItem.paint reads self.fill on every repaint, so replacing
        # it after the fact is enough.
        flagged_ids = {
            int(self.source.ep_ids[i])
            for i in self._flagged
            if i < len(self.source.ep_ids)
        }
        for item in self.rect_items:
            if int(item.id) in flagged_ids:
                colour = QT.QColor(item.fill)
                colour.setAlpha(50)
                item.fill = colour
                item.update()

    # -- navigation --------------------------------------------------------

    def _current_index(self) -> int:
        """Which epoch the cursor is in, or the one the table has selected."""
        selected = self.table_widget.selectedIndexes()
        if selected:
            return selected[0].row()

        times = np.asarray(self.source.ep_times, dtype="float64")
        if len(times) == 0:
            return 0
        after = np.searchsorted(times, float(self.t), side="right") - 1
        return int(np.clip(after, 0, len(times) - 1))

    def _seek(self, index: int) -> None:
        if 0 <= index < len(self.source.ep_times):
            self.on_seek_table(index)

    def go_to_next_state(self) -> None:
        labels = list(self.source.ep_labels)
        start = self._current_index()
        if start >= len(labels):
            return
        current = labels[start]
        for index in range(start + 1, len(labels)):
            if labels[index] != current:
                self._seek(index)
                return

    def go_to_previous_state(self) -> None:
        labels = list(self.source.ep_labels)
        start = self._current_index()
        if not labels:
            return
        current = labels[min(start, len(labels) - 1)]
        for index in range(start - 1, -1, -1):
            if labels[index] != current:
                self._seek(index)
                return

    def go_to_next_flagged(self) -> None:
        start = self._current_index()
        for index in sorted(self._flagged):
            if index > start:
                self._seek(index)
                return

    def go_to_previous_flagged(self) -> None:
        start = self._current_index()
        for index in sorted(self._flagged, reverse=True):
            if index < start:
                self._seek(index)
                return


# ---------------------------------------------------------------------------
# The Fourier view
# ---------------------------------------------------------------------------


def spectrogram_params_from(params: dict, channel: str = "EEG") -> dict:
    """Spectrogram settings matching the params a recording is scored with.

    Upstream defaults to ``binsize = 0.01`` s, which at a rodent sampling rate
    is a **one-sample window** -- the transform is meaningless and the panel
    looks empty. nyx already knows the epoch length and overlap it computes its
    own features with, so those are used instead and the view shows what the
    scoring sees.
    """
    section = params.get(channel, {}) or {}
    settings = {}
    if section.get("binsize"):
        settings["binsize"] = float(section["binsize"])
    if section.get("overlapratio") is not None:
        settings["overlapratio"] = float(section["overlapratio"])
    return settings


class NyxSpectrogramWorker(SpectrogramWorker):
    """Upstream's worker, without the debug print and the log of zero.

    Two things upstream does that are wrong in a GUI: it prints ``nperseg`` and
    ``noverlap`` to stdout on *every redraw*, and it takes ``log10`` of a
    spectrogram that can contain exact zeros -- a flat or disconnected stretch
    of signal -- which fills the console with divide-by-zero warnings and puts
    ``-inf`` in the image.

    Copied rather than extended because both are in the middle of the method.
    The lines that differ are marked.
    """

    def on_request_data(self, chan, t, t_start, t_stop, visible_channels, worker_params):
        if chan != self.chan or not visible_channels[chan]:
            return
        if self.viewer.t != t:
            return  # the viewer has moved on already

        binsize = worker_params["binsize"]
        overlapratio = worker_params["overlapratio"]
        scaling = worker_params["scaling"]
        detrend = worker_params["detrend"]
        mode = worker_params["mode"]

        i_start = self.source.time_to_index(t_start)
        i_stop = self.source.time_to_index(t_stop)
        i_start = min(max(0, i_start), self.source.get_length())
        i_stop = min(max(0, i_stop), self.source.get_length())

        sr = self.source.sample_rate
        nperseg = int(binsize * sr)
        noverlap = int(overlapratio * nperseg)
        if noverlap >= nperseg:
            noverlap = noverlap - 1
        # -- nyx: no debug print here. Upstream writes nperseg and noverlap to
        #    stdout on every redraw, which is once per scroll step.

        if nperseg == 0 or (i_stop - i_start) < nperseg:
            _emit(self.data_ready, chan, t, t_start, t_stop, t_start, t_stop, None)
            return

        sigs_chunk = self.source.get_chunk(i_start=i_start, i_stop=i_stop)
        sig = sigs_chunk[:, chan]

        _freqs, times, Sxx = scipy.signal.spectrogram(
            sig, fs=sr, nperseg=nperseg, noverlap=noverlap,
            detrend=detrend, scaling=scaling, mode=mode,
        )

        if worker_params["scale"] == "dB" and mode == "psd":
            # -- nyx: floor it. A flat or disconnected stretch gives exact
            #    zeros, and log10(0) is -inf plus a warning per redraw.
            Sxx = 10.0 * np.log10(np.maximum(Sxx, 1e-12))

        if len(times) >= 2:
            slide = times[1] - times[0]
            t1 = self.source.index_to_time(i_start) + times[0] - slide / 2.0
            t2 = self.source.index_to_time(i_start) + times[-1] + slide / 2.0
            _emit(self.data_ready, chan, t, t_start, t_stop, t1, t2, Sxx)
        else:
            _emit(self.data_ready, chan, t, t_start, t_stop, t_start, t_stop, None)


class NyxSpectrogramViewer(SpectrogramViewer):
    """The fast Fourier view, with settings that show something.

    Same relationship to upstream as :class:`NyxTimeFreqViewer`: the worker is
    swapped, which upstream has no hook for, so ``__init__`` is copied to do
    it. A two-line ``_WorkerClass`` attribute upstream would remove both.
    """

    def __init__(self, **kargs):
        # Copied from SpectrogramViewer.__init__ so the worker class can be
        # swapped; upstream constructs SpectrogramWorker by name. The
        # attribute names below are upstream's -- `timefreq_makers` for the
        # workers, even in the Fourier viewer -- because closeEvent stops the
        # threads by those names.
        from ephyviewer.base import BaseMultiChannelViewer

        BaseMultiChannelViewer.__init__(self, **kargs)

        self.make_params()

        self.by_channel_params.blockSignals(True)
        for c in range(self.source.nb_channel):
            self.by_channel_params["ch" + str(c), "visible"] = c == 0
        self.by_channel_params.blockSignals(False)

        self.make_param_controller()
        self.params_controller.some_clim_changed.connect(self.refresh)

        self.set_layout()
        self.change_color_scale()
        self.create_grid()

        self.last_Sxx = {}
        self.threads = []
        self.timefreq_makers = []
        for c in range(self.source.nb_channel):
            thread = QT.QThread(parent=self)
            self.threads.append(thread)
            worker = NyxSpectrogramWorker(self.source, self, c)  # <- the change
            self.timefreq_makers.append(worker)
            worker.moveToThread(thread)
            thread.start()
            self.last_Sxx[c] = None
            worker.data_ready.connect(self.on_data_ready)
            self.request_data.connect(worker.on_request_data)

        self.params.param("xsize").setLimits((0, np.inf))

    def apply_settings(self, settings: dict) -> None:
        """Set spectrogram parameters from :func:`spectrogram_params_from`."""
        # Upstream calls the group "scalogram" even in the Fourier viewer.
        group = self.params.param("scalogram")
        for name, value in settings.items():
            try:
                group.param(name).setValue(value)
            except Exception:  # noqa: BLE001 - an unknown key is not fatal
                continue
        self.refresh()


def make_timefreq_viewer(source, name: str, params: dict, channel: str,
                         scalogram: bool):
    """The time-frequency view for a channel, set up to match the params.

    ``scalogram`` picks the Morlet view -- easier to read, much slower, and
    what the scalogram feature backend computes -- over the Fourier one.
    """
    if scalogram:
        viewer = NyxTimeFreqViewer(source=source, name=name)
        viewer.apply_settings(timefreq_params_from(params, channel))
        return viewer

    viewer = NyxSpectrogramViewer(source=source, name=name)
    viewer.apply_settings(spectrogram_params_from(params, channel))
    return viewer
