"""Scroll the recording, and correct the hypnogram by hand.

An automatic scoring is a starting point. The part that decides whether it can
be used is looking at it: scrolling the traces, reading the spectrogram, and
fixing the epochs that are wrong. That is what this window is for, and it is
the one thing the notebooks cannot do.

What you edit is what gets saved. The corrected hypnogram goes straight back
into the session, so ``save_results`` writes it, agreement is recomputed
against it, and ``run.json`` records that a human changed it -- without which a
saved record would claim to replay a scoring it cannot.
"""

from __future__ import annotations

import numpy as np
from ephyviewer import (
    EpochViewer,
    MainViewer,
    SpectrogramViewer,
    TraceViewer,
)
from ephyviewer.datasource import (
    InMemoryEpochSource,
    SpikeInterfaceRecordingSource,
    WritableEpochSource,
)

from nyx.gui.hypnogram import from_epoch_dict, stage_palette, to_epoch_dict
from nyx.gui.viewers import NyxEpochEncoder, NyxTimeFreqViewer, timefreq_params_from

__all__ = ["NyxEpochSource", "build_review_window", "open_review_window"]


class NyxEpochSource(WritableEpochSource):
    """A hypnogram the epoch encoder can edit, backed by nyx's format.

    ``save()`` does not write a file. It hands the corrected hypnogram back to
    whoever asked for it -- the session -- which is what makes the edit part of
    the result rather than a separate artefact. Pass ``savepath`` to also write
    a CSV.

    Subclassing ``WritableEpochSource`` rather than reusing ``CsvEpochSource``
    is deliberate: nyx's ``savehypno`` writes its own header, ephyviewer's CSV
    convention has changed between releases, and round-tripping through a
    temporary file would lose the direct hand-back anyway.
    """

    def __init__(
        self,
        hypnogram: dict,
        *,
        name: str = "nyx",
        t_offset: float = 0.0,
        window_duration: float | None = None,
        possible_labels=None,
        color_labels=None,
        on_save=None,
        savepath: str | None = None,
    ):
        self._hypnogram = hypnogram
        self.t_offset = float(t_offset)
        self.window_duration = window_duration
        self.on_save = on_save
        self.savepath = savepath

        labels = list(hypnogram["label"])
        if possible_labels is None:
            possible_labels, palette = stage_palette(labels)
            color_labels = color_labels or palette

        super().__init__(
            epoch=to_epoch_dict(hypnogram, name, self.t_offset),
            possible_labels=list(possible_labels),
            color_labels=color_labels,
            channel_name=name,
        )

    def load(self) -> dict:
        return to_epoch_dict(self._hypnogram, self.channel_name, self.t_offset)

    def to_nyx(self) -> dict:
        """The current state, back in nyx's format and on nyx's clock."""
        return from_epoch_dict(
            {
                "time": np.asarray(self.ep_times, dtype="float64"),
                "duration": np.asarray(self.ep_durations, dtype="float64"),
                "label": np.asarray(self.ep_labels, dtype="U16"),
            },
            self.t_offset,
            duration=self.window_duration,
        )

    def save(self) -> None:
        hypnogram = self.to_nyx()
        if self.savepath:
            from nyx.scoring import savehypno

            savehypno(hypnogram, self.savepath)
        if self.on_save is not None:
            self.on_save(hypnogram)


class ReviewWindow(MainViewer):
    """A review window that shuts down deterministically.

    Every ephyviewer view owns worker ``QThread``s and stops them in its
    ``closeEvent``. ``MainViewer`` forwards the close on, so that works -- but
    only if something calls ``close()``. A window that is simply garbage
    collected leaves those threads running against deleted C++ objects, which
    takes the whole process down on the way out rather than raising anything.

    So closing has to be made to happen -- by the user, or by whoever opened
    it. It is not done in ``__del__``, which is the obvious place and the
    wrong one: see the note below.

    It also clears the encoder's unsaved-changes flag first. That prompt is
    ephyviewer asking whether to write a file, which is not how saving works
    here -- edits go back to the session, and the session is what writes.
    """

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        encoder = getattr(self, "encoder", None)
        if encoder is not None:
            encoder.changes_since_save = 0
        super().closeEvent(event)

    # Deliberately no __del__: it can run during interpreter shutdown, after
    # Qt has torn itself down, and touching a Qt object there is an access
    # violation rather than an exception. The window is closed explicitly --
    # by the user, or by the fixture that built it.


def build_review_window(
    result,
    *,
    reference=None,
    params: dict | None = None,
    rules=None,
    on_save=None,
    scalogram: bool | None = None,
) -> MainViewer:
    """Traces, time-frequency and the editable hypnogram, in one window.

    The rows are in the same order as :func:`nyx.plot_scoring_overview`, so the
    two are read the same way.

    Both a scalogram and a spectrogram view are offered. The scalogram is far
    easier to read and much slower -- a wavelet transform on every redraw -- so
    the one that starts visible follows the feature backend the params declare:
    what you scroll through is then the transform the scoring used.
    """
    params = params if params is not None else (result.params or {})
    if scalogram is None:
        scalogram = params.get("features", {}).get("method") == "scalogram"

    window = ReviewWindow(debug=False, show_auto_scale=True)
    recording = result.recording

    eeg_source = SpikeInterfaceRecordingSource(recording=recording.eeg)
    # Derived from the source, never from result.window. Recording.time_slice
    # keeps absolute times, but SpikeInterfaceRecordingSource restarts its
    # clock at zero, and time_slice returns the recording untouched when the
    # window is the whole thing -- three behaviours whose combination is not
    # worth predicting. Asking gets it right in every case; assuming
    # window[0] would shift a windowed recording by the length of the trim.
    t_offset = float(getattr(eeg_source, "t_start", 0.0) or 0.0)

    if recording.emg is not None:
        # A separate source because the EMG may be sampled at a different rate,
        # and one source cannot carry two.
        emg_source = SpikeInterfaceRecordingSource(recording=recording.emg)
        window.add_view(TraceViewer(source=emg_source, name="EMG"))
        window.add_view(
            _timefreq(emg_source, "EMG spectrum", params, "EMG", scalogram),
            tabify_with="EMG",
        )

    window.add_view(TraceViewer(source=eeg_source, name="EEG"))
    window.add_view(
        _timefreq(eeg_source, "EEG spectrum", params, "EEG", scalogram)
    )

    if reference is not None:
        from nyx.metrics import normalise_labels

        window.add_view(
            EpochViewer(
                source=InMemoryEpochSource(
                    all_epochs=[
                        to_epoch_dict(normalise_labels(reference), "reference", t_offset)
                    ]
                ),
                name="reference",
            )
        )

    # No window_duration: the scored hypnogram stops at the last whole epoch,
    # a little short of the window, and padding that out here would mean
    # opening the reviewer and saving without touching anything changed the
    # result. Holes *inside* are still filled; the tail is nyx's business, and
    # save_hypno_with_padding already does it.
    source = NyxEpochSource(
        result.hypnogram, name="nyx", t_offset=t_offset, on_save=on_save
    )
    encoder = NyxEpochEncoder(source=source, name="hypnogram", rules=rules)
    window.add_view(encoder)

    window.epoch_source = source  # so a caller can read the edit back
    window.encoder = encoder
    return window


def _timefreq(source, name: str, params: dict, channel: str, scalogram: bool):
    """The time-frequency view, matched to what the scoring computed."""
    if scalogram:
        viewer = NyxTimeFreqViewer(source=source, name=name)
        viewer.apply_settings(timefreq_params_from(params, channel))
        return viewer

    viewer = SpectrogramViewer(source=source, name=name)
    section = params.get(channel, {}) or {}
    for key, value in (
        ("f_start", section.get("min_freq")),
        ("f_stop", section.get("max_freq")),
    ):
        if value is None:
            continue
        try:
            viewer.params.param("spectrogram").param(key).setValue(float(value))
        except Exception:  # noqa: BLE001 - an older ephyviewer may name it differently
            pass
    return viewer


def open_review_window(session, parent=None):
    """Open the review window on a session, wiring edits back into it."""
    result = session.result()

    def apply(hypnogram):
        session.set_hypnogram(hypnogram, source="manual")

    window = build_review_window(
        result,
        reference=session.reference,
        params=session.params,
        rules=session.postprocess_rules(),
        on_save=apply,
    )
    window.setWindowTitle(f"nyx -- review {result.recording.name or 'recording'}")
    window.show()
    return window  # hold on to it, or Qt collects it
