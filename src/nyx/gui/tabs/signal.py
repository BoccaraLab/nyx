"""Tab 2 -- mains interference, and where to start and stop.

The notebooks' second STOP. Two decisions, both cheap to get right here and
expensive to discover later:

*The notch.* ``check_signals`` measures how much power sits at 50 and 60 Hz and
suggests one. It matters most for the EMG, whose band is 30-100 Hz -- mains hum
lands squarely inside it and inflates the power that separates wake from sleep.
The EEG band usually stops below it. That asymmetry is why the two channels get
separate checkboxes rather than one.

*The window.* Recordings often start before the animal is connected and end
after it is disconnected, and flat or saturated ends drag the scaling around
for everything in between.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.canvas import FigureView
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab, controls_column
from nyx.gui.widgets import monospace

__all__ = ["SignalTab"]


class SignalTab(Tab):
    title = "Signal check"
    subtitle = (
        "Look at the raw signal before scoring it. Decide the notch and the "
        "analysis window here -- both are cheap to fix now and expensive to "
        "discover afterwards."
    )
    stage = Stage.PREPROCESS
    run_label = "Check the signal"

    def build(self) -> None:
        self._check = None

        layout = QHBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- checking
        check_box = QGroupBox("Check")
        check_form = QFormLayout(check_box)
        self.preview = QDoubleSpinBox()
        self.preview.setRange(0.0, 1e7)
        self.preview.setSingleStep(1000.0)
        self.preview.setValue(10000.0)
        self.preview.setSuffix(" s")
        self.preview.setToolTip(
            "How much signal to look at. 0 uses all of it, which on a long "
            "recording is slow."
        )
        check_form.addRow("preview", self.preview)

        run_check = QPushButton("Run the check")
        run_check.clicked.connect(lambda: self.run_requested.emit(Stage.LOAD))
        run_check.clicked.connect(self._run_check)
        check_form.addRow("", run_check)

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

        layout.addWidget(controls_column(check_box, notch_box, window_box))

        # -- figure and summary
        splitter = QSplitter()
        splitter.setOrientation(splitter.orientation().Vertical)

        self.figure = FigureView(
            placeholder="Run the check to see the traces and their spectra."
        )
        splitter.addWidget(self.figure)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(monospace())
        self.summary.setPlaceholderText(
            "The check reports the sampling rate, the amount of flat or "
            "saturated signal, and how much power sits at 50 and 60 Hz."
        )
        self.summary.setMaximumHeight(220)
        splitter.addWidget(self.summary)
        splitter.setSizes([600, 200])

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(splitter)
        layout.addWidget(right, 1)

    # -- checking ----------------------------------------------------------

    def _run_check(self) -> None:
        """Runs on this thread: the check is bounded by ``preview``.

        Deliberately not a job. It reads at most ``preview`` seconds, and
        keeping it here means the figure is built on the thread that owns it.
        """
        try:
            recording = self.session.recording
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"Load a recording first: {exc}")
            return

        preview = self.preview.value() or None
        self.status.emit("Checking the signal...")
        self._check = nyx.check_signals(recording, preview=preview)

        self.figure.set_figure(self._check.plot())
        self.summary.setPlainText(self._check.summary())

        suggested = self._check.suggested_notch()
        with self.quiet(self.notch):
            self.notch.setCurrentText("none" if not suggested else str(int(suggested)))
        if suggested:
            self.status.emit(
                f"Mains interference at {int(suggested)} Hz -- worth notching, "
                f"the EMG band especially."
            )
        else:
            self.status.emit("No obvious mains interference.")

        self.end.setMaximum(float(recording.duration))
        if self.end.value() == 0.0:
            with self.quiet(self.end):
                self.end.setValue(float(recording.duration))

    def _toggle_window(self, whole: bool) -> None:
        self.start.setEnabled(not whole)
        self.end.setEnabled(not whole)

    # -- session -----------------------------------------------------------

    def apply(self) -> None:
        text = self.notch.currentText()
        frequency = None if text == "none" else float(text)
        harmonics = int(self.harmonics.value())

        params = dict(self.session.params)
        for section, wanted in (
            ("EEG", self.notch_eeg.isChecked()),
            ("EMG", self.notch_emg.isChecked()),
        ):
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
