"""Tab 2 -- mains interference, and where to start and stop.

The notebooks' second STOP, and the first place a scrollable view earns its
keep: deciding where a recording really starts is a matter of *looking* at the
first few minutes, not at a thumbnail of twelve hours. The traces and their
time-frequency views are ephyviewer's, locked to one clock, so scrolling one
scrolls them all.

Two decisions, both cheap now and expensive to discover later:

*The notch.* ``check_signals`` measures the power at 50 and 60 Hz and suggests
one. It matters most for the EMG, whose band is 30-100 Hz: mains hum lands
inside it and inflates the very power that separates wake from sleep. The EEG
band usually stops below it. That asymmetry is why the two channels have
separate checkboxes rather than one.

*The window.* Recordings often start before the animal is connected and end
after it is disconnected, and flat or saturated ends drag the scaling around
for everything in between.
"""

from __future__ import annotations

from ephyviewer import TraceViewer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

import nyx
from nyx.gui.panels import MplPanel, TextPanel
from nyx.gui.session import Stage
from nyx.gui.sources import autoscale, trace_sources
from nyx.gui.tabs.base import Tab
from nyx.gui.viewers import (
    make_timefreq_viewer,
    spectrogram_params_from,
    timefreq_params_from,
)
from nyx.gui.widgets import Section

__all__ = ["SignalTab"]


class SignalTab(Tab):
    title = "Signal check"
    subtitle = (
        "Scroll the traces before scoring them. Decide the notch and the "
        "analysis window here -- both are cheap to fix now and expensive to "
        "discover afterwards."
    )
    stage = Stage.PREPROCESS
    run_label = "Apply and preprocess"

    def build_controls(self) -> list:
        self._check = None
        self._seeded = False

        # -- checking
        check_box = Section(
            "Measure the signal",
            "Reports the sampling rate, how much of the recording is flat "
            "or saturated, and how much power sits at 50 and 60 Hz.\n\n"
            "The preview is how much signal to measure; 0 uses all of it, "
            "which on a long recording is slow.",
        )
        check_form = QFormLayout(check_box)
        self.preview = QDoubleSpinBox()
        self.preview.setRange(0.0, 1e7)
        self.preview.setSingleStep(1000.0)
        self.preview.setValue(10000.0)
        self.preview.setSuffix(" s")
        self.preview.setToolTip(
            "How much signal to measure. 0 uses all of it, which on a long "
            "recording is slow."
        )
        check_form.addRow("preview", self.preview)

        measure = QPushButton("Measure")
        measure.setToolTip(
            "Reports the sampling rate, flat or saturated stretches, and how "
            "much power sits at 50 and 60 Hz."
        )
        measure.clicked.connect(self._measure)
        check_form.addRow("", measure)

        # -- notch
        notch_box = Section(
            "Mains notch",
            "Mains hum at 50 or 60 Hz lands squarely inside the EMG band "
            "of 30-100 Hz, where it inflates the very power that "
            "separates wake from sleep. That is why the EMG is ticked and "
            "the EEG is not: the EEG band usually stops below the mains "
            "frequency, so notching it is often unnecessary.\n\n"
            "Measure first -- it suggests a frequency, or tells you there "
            "is no interference worth removing.",
        )
        notch_form = QFormLayout(notch_box)
        self.notch = QComboBox()
        self.notch.addItems(["none", "50", "60"])
        notch_form.addRow("frequency", self.notch)

        self.notch_eeg = QCheckBox("EEG")
        self.notch_emg = QCheckBox("EMG")
        self.notch_emg.setChecked(True)
        self.notch_emg.setToolTip(
            "The EMG band is 30-100 Hz, so mains hum lands inside it. This is "
            "the one that usually matters."
        )
        self.notch_eeg.setToolTip(
            "The EEG band usually stops below the mains frequency, so notching "
            "it is often unnecessary."
        )
        channels = QWidget()
        channels_layout = QHBoxLayout(channels)
        channels_layout.setContentsMargins(0, 0, 0, 0)
        channels_layout.addWidget(self.notch_eeg)
        channels_layout.addWidget(self.notch_emg)
        channels_layout.addStretch(1)
        notch_form.addRow("apply to", channels)

        self.harmonics = QDoubleSpinBox()
        self.harmonics.setRange(0, 20)
        self.harmonics.setDecimals(0)
        self.harmonics.setValue(3)
        notch_form.addRow("harmonics", self.harmonics)

        # -- what the spectrogram shows
        view_box = Section(
            "Time-frequency view",
            "The wavelet view is a Morlet scalogram in dB, the same "
            "transform nyx's scalogram backend computes. Far easier to "
            "read than the Fourier view, and much slower.\n\n"
            "Binsize is the window length of the transform and the epoch "
            "length nyx scores on. ephyviewer's own default is 0.01 s, "
            "which at these sampling rates is a single sample and shows "
            "nothing -- these start from your parameters instead.\n\n"
            "Changing them retunes the view. Apply to the params writes "
            "them into the parameters so the scoring uses them too.",
        )
        view_form = QFormLayout(view_box)

        self.scalogram = QCheckBox("wavelet (slower, easier to read)")
        self.scalogram.setToolTip(
            "A Morlet scalogram in dB, as nyx's scalogram feature backend "
            "computes it. Much slower than the Fourier view -- it recomputes "
            "on every scroll -- and much easier to read."
        )
        self.scalogram.toggled.connect(self._rebuild_viewers)
        view_form.addRow("", self.scalogram)

        self.binsize = QDoubleSpinBox()
        self.binsize.setRange(0.01, 600.0)
        self.binsize.setDecimals(2)
        self.binsize.setSingleStep(0.5)
        self.binsize.setSuffix(" s")
        self.binsize.setToolTip(
            "Window length of the transform, and the epoch length nyx scores "
            "on. ephyviewer's own default is 0.01 s, which at these sampling "
            "rates is a one-sample window and shows nothing."
        )
        view_form.addRow("binsize", self.binsize)

        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0.0, 0.95)
        self.overlap.setDecimals(2)
        self.overlap.setSingleStep(0.05)
        view_form.addRow("overlap", self.overlap)

        self.fmax = QDoubleSpinBox()
        self.fmax.setRange(1.0, 5000.0)
        self.fmax.setDecimals(1)
        self.fmax.setSuffix(" Hz")
        view_form.addRow("max frequency", self.fmax)

        for spin in (self.binsize, self.overlap, self.fmax):
            spin.valueChanged.connect(self._settings_changed)

        self.channel = QComboBox()
        self.channel.addItems(["EEG", "EMG"])
        self.channel.setToolTip(
            "Which section of the params these settings belong to. The two "
            "bands differ: EEG is roughly 0.5-40 Hz, EMG 30-100."
        )
        self.channel.currentTextChanged.connect(self._load_settings)
        view_form.addRow("settings for", self.channel)

        apply_settings = QPushButton("Apply to the params")
        apply_settings.setToolTip(
            "Write these into the parameters, so the scoring uses them too -- "
            "not just the view."
        )
        apply_settings.clicked.connect(self._apply_settings)
        view_form.addRow("", apply_settings)

        # -- window
        window_box = Section(
            "Analysis window",
            "Recordings often start before the animal is connected and "
            "end after it is disconnected. Flat or saturated ends drag "
            "the scaling around for everything in between, so trimming "
            "them is worth doing.\n\n"
            "Scroll the traces to the stretch you want and press Use the "
            "view as the window.",
        )
        window_form = QFormLayout(window_box)
        self.whole = QCheckBox("the whole recording")
        self.whole.setChecked(True)
        self.whole.toggled.connect(self._toggle_window)
        window_form.addRow("", self.whole)

        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(0.0, 1e9)
            spin.setDecimals(1)
            spin.setSingleStep(60.0)
            spin.setSuffix(" s")
            spin.setEnabled(False)
        window_form.addRow("start", self.start)
        window_form.addRow("end", self.end)

        here = QPushButton("Use the view as the window")
        here.setToolTip(
            "Take the start and end from what the traces are currently showing."
        )
        here.clicked.connect(self._window_from_view)
        window_form.addRow("", here)

        return [check_box, view_box, notch_box, window_box]

    def build_docks(self) -> None:
        self.summary = TextPanel(
            "measurements",
            placeholder="Press Measure to see the sampling rate, artefacts "
                        "and mains power.",
        )
        self.spectra = MplPanel(
            "spectra", placeholder="Press Measure to see the power spectra."
        )
        self._built_for = None

    # -- panels ------------------------------------------------------------

    def _build_viewers(self) -> None:
        """Put the recording's traces in the dock area, once per recording."""
        recording = self.session.recording
        if self._built_for is recording:
            return

        self.docks.clear()
        eeg, emg, _offset = trace_sources(recording)

        self.docks.add(autoscale(TraceViewer(source=eeg, name="EEG")))
        self.docks.add(
            self._timefreq(eeg, "EEG spectrum", "EEG"), tabify_with="EEG"
        )
        if emg is not None:
            self.docks.add(autoscale(TraceViewer(source=emg, name="EMG")))
            self.docks.add(
                self._timefreq(emg, "EMG spectrum", "EMG"), tabify_with="EMG"
            )

        self.docks.add(self.summary)   # bottom, by TextPanel default
        self.docks.add(self.spectra, tabify_with="measurements")
        self._built_for = recording

    def _timefreq(self, source, name: str, channel: str):
        """The time-frequency view for a channel, set from the params.

        The Fourier view by default: this tab is for scrolling a whole
        recording, and a wavelet transform per redraw is not what that wants.
        The checkbox switches it.
        """
        return make_timefreq_viewer(
            source, name, self._view_params(channel), channel,
            self.scalogram.isChecked(),
        )

    def _view_params(self, channel: str) -> dict:
        """The session's params with the panel's overrides on top.

        The settings can be tried on the view before being written into the
        params, so a bad binsize costs a redraw rather than a recompute.
        """
        params = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in self.session.params.items()}
        section = dict(params.get(channel, {}) or {})
        if self.channel.currentText() == channel:
            section["binsize"] = float(self.binsize.value())
            section["overlapratio"] = float(self.overlap.value())
            section["max_freq"] = float(self.fmax.value())
        params[channel] = section
        return params

    def _rebuild_viewers(self) -> None:
        self._built_for = None
        self.safe_refresh()

    def _settings_changed(self) -> None:
        """Retune the open viewers without rebuilding the dock layout."""
        channel = self.channel.currentText()
        params = self._view_params(channel)
        viewer = self.docks.panel(f"{channel} spectrum")
        if viewer is None or not hasattr(viewer, "apply_settings"):
            return
        if self.scalogram.isChecked():
            viewer.apply_settings(timefreq_params_from(params, channel))
        else:
            viewer.apply_settings(spectrogram_params_from(params, channel))

    def _load_settings(self) -> None:
        """Read the spin boxes back from the params for the chosen channel."""
        section = self.session.params.get(self.channel.currentText(), {}) or {}
        for spin, key, fallback in (
            (self.binsize, "binsize", 4.0),
            (self.overlap, "overlapratio", 0.5),
            (self.fmax, "max_freq", 40.0),
        ):
            with self.quiet(spin):
                spin.setValue(float(section.get(key, fallback) or fallback))

    def _apply_settings(self) -> None:
        """Write the view's settings into the params the scoring will use."""
        channel = self.channel.currentText()
        params = self._view_params(channel)
        self.session.set_params(params)
        self.status.emit(
            f"{channel}: binsize {self.binsize.value():g} s, overlap "
            f"{self.overlap.value():g}, up to {self.fmax.value():g} Hz. "
            f"The scoring will use these too."
        )

    # -- measuring ---------------------------------------------------------

    def _measure(self) -> None:
        """Bounded by ``preview``, so it runs here rather than on a worker."""
        try:
            recording = self.session.recording
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"Load a recording first: {exc}")
            return

        self.status.emit("Measuring...")
        self._check = nyx.check_signals(recording, preview=self.preview.value() or None)
        self.summary.set_text(self._check.summary())
        self.spectra.set_figure(self._check.plot())

        suggested = self._check.suggested_notch()
        with self.quiet(self.notch):
            self.notch.setCurrentText("none" if not suggested else str(int(suggested)))
        self.status.emit(
            f"Mains interference at {int(suggested)} Hz -- worth notching, the "
            f"EMG band especially." if suggested
            else "No obvious mains interference."
        )

    def _toggle_window(self, whole: bool) -> None:
        self.start.setEnabled(not whole)
        self.end.setEnabled(not whole)

    def _window_from_view(self) -> None:
        """Read the window off whatever the traces are showing."""
        toolbar = self.docks.navigation_toolbar
        centre = float(toolbar.t)
        width = float(getattr(toolbar, "xsize", 0.0) or 0.0)
        if width <= 0:
            self.status.emit("Scroll the traces to the stretch you want first.")
            return
        self.whole.setChecked(False)
        self.start.setValue(max(0.0, centre - width / 2))
        self.end.setValue(centre + width / 2)
        self.status.emit(
            f"Window set to {self.start.value():.0f}-{self.end.value():.0f} s."
        )

    # -- session -----------------------------------------------------------

    def apply(self) -> None:
        text = self.notch.currentText()
        frequency = None if text == "none" else float(text)
        harmonics = int(self.harmonics.value())

        params = dict(self.session.params)
        for section, wanted in (("EEG", self.notch_eeg.isChecked()),
                                ("EMG", self.notch_emg.isChecked())):
            settings = dict(params.get(section, {}))
            settings["notch"] = frequency if wanted else None
            if frequency is not None and wanted:
                settings["notch_harmonics"] = harmonics
            params[section] = settings
        self.session.set_params(params)

        if self.whole.isChecked():
            self.session.set_window(None)
        else:
            self.session.set_window((self.start.value(), self.end.value()))

    def refresh(self) -> None:
        recording = self.session.recording

        # Before _build_viewers: the viewers are built from these, and a spin
        # box still at its minimum means a 0.01 s window -- one sample, and a
        # blank panel.
        if self.binsize.value() == self.binsize.minimum():
            self._load_settings()

        # Seeded once, from the feature backend the params declare. Doing it
        # on every refresh would undo the checkbox the moment it was ticked,
        # since ticking it refreshes.
        if not self._seeded:
            self._seeded = True
            with self.quiet(self.scalogram):
                self.scalogram.setChecked(
                    self.session.params.get("features", {}).get("method")
                    == "scalogram"
                )

        self._build_viewers()

        self.end.setMaximum(float(recording.duration))
        if self.end.value() == 0.0:
            with self.quiet(self.end):
                self.end.setValue(float(recording.duration))

        params = self.session.params
        notch = params.get("EMG", {}).get("notch") or params.get("EEG", {}).get("notch")
        with self.quiet(self.notch):
            self.notch.setCurrentText("none" if not notch else str(int(notch)))
        with self.quiet(self.notch_eeg):
            self.notch_eeg.setChecked(bool(params.get("EEG", {}).get("notch")))
        with self.quiet(self.notch_emg):
            self.notch_emg.setChecked(bool(params.get("EMG", {}).get("notch")))
