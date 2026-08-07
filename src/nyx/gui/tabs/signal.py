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

from ephyviewer import SpectrogramViewer, TraceViewer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

import nyx
from nyx.gui.panels import MplPanel, TextPanel
from nyx.gui.session import Stage
from nyx.gui.sources import trace_sources
from nyx.gui.tabs.base import Tab
from nyx.gui.viewers import NyxTimeFreqViewer, timefreq_params_from

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

        # -- checking
        check_box = QGroupBox("Measure the signal")
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
        notch_box = QGroupBox("Mains notch")
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

        # -- window
        window_box = QGroupBox("Analysis window")
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

        return [check_box, notch_box, window_box]

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

        self.docks.add(TraceViewer(source=eeg, name="EEG"))
        self.docks.add(
            self._timefreq(eeg, "EEG spectrum", "EEG"), tabify_with="EEG"
        )
        if emg is not None:
            self.docks.add(TraceViewer(source=emg, name="EMG"))
            self.docks.add(
                self._timefreq(emg, "EMG spectrum", "EMG"), tabify_with="EMG"
            )

        self.docks.add(self.summary, location="right")
        self.docks.add(self.spectra, tabify_with="measurements")
        self._built_for = recording

    def _timefreq(self, source, name: str, channel: str):
        """The spectrogram, matched to what the scoring will compute.

        The fast Fourier view by default -- this tab is for scrolling a whole
        recording, and a wavelet transform per redraw is not what that wants.
        """
        params = self.session.params
        if params.get("features", {}).get("method") == "scalogram":
            viewer = NyxTimeFreqViewer(source=source, name=name)
            viewer.apply_settings(timefreq_params_from(params, channel))
            return viewer

        viewer = SpectrogramViewer(source=source, name=name)
        section = params.get(channel, {}) or {}
        for key, value in (("f_start", section.get("min_freq")),
                           ("f_stop", section.get("max_freq"))):
            if value is None:
                continue
            try:
                viewer.params.param("spectrogram").param(key).setValue(float(value))
            except Exception:  # noqa: BLE001 - naming varies between releases
                pass
        return viewer

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
