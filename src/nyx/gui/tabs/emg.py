"""Tab 3 -- the EMG threshold, dragged.

The notebooks' third STOP, and the one that most rewards being interactive.
The cut is a single number, but choosing it well means looking at two things at
once -- the distribution, where it should sit in a valley, and the time course,
where it should track what the animal is obviously doing. So both are on
screen, the same line is drawn on both, and dragging either moves both.

Re-scoring is instant: ``classify_wake_sleep`` with an explicit threshold is
milliseconds, and the EMG features it works from are cached. The automatic
threshold is *not* instant -- it fits four GaussianMixtures and picks by BIC --
so it is fitted once, when the features are computed, and reused.

Two things this tab has to survive:

*A recording with no EMG.* ``compute_emg_features`` returns ``None`` and every
epoch starts as SLEEP, leaving clustering to find wake in the EEG alone.

*The scalogram backend.* Its power vector is one column per *sample* rather
than per epoch, and the run-length encoding behind ``classify_wake_sleep`` is a
Python loop over it. Live re-scoring on every mouse-move would crawl, so past a
threshold of size the update waits for the mouse button to come up.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.canvas import FigureView, PanelCanvas
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab, controls_column
from nyx.gui.widgets import monospace
from nyx.interactive import ThresholdSelector

__all__ = ["EmgTab"]

#: Past this many samples, re-score on mouse-release rather than on every
#: motion event. The scalogram backend produces one column per sample.
_LIVE_LIMIT = 200_000


class EmgTab(Tab):
    title = "EMG threshold"
    subtitle = (
        "Drag the line. It should sit in the valley of the histogram and "
        "track the trace; the wake percentage updates as you let go."
    )
    stage = Stage.WAKE_SLEEP
    run_label = "Compute the EMG features"

    def build(self) -> None:
        self._selector = None
        self._nosignal_selector = None

        layout = QHBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)

        threshold_box = QGroupBox("Threshold")
        form = QFormLayout(threshold_box)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e6, 1e6)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.valueChanged.connect(self._threshold_typed)
        form.addRow("wake / sleep", self.threshold)

        self.nosignal = QDoubleSpinBox()
        self.nosignal.setRange(-1e6, 1e6)
        self.nosignal.setDecimals(3)
        self.nosignal.setSingleStep(0.01)
        self.nosignal.setToolTip(
            "Power at or below this is signal loss rather than deep sleep. "
            "0 disables it."
        )
        self.nosignal.valueChanged.connect(self._threshold_typed)
        form.addRow("no signal", self.nosignal)

        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0.0, 600.0)
        self.min_duration.setDecimals(1)
        self.min_duration.setSuffix(" s")
        self.min_duration.setToolTip(
            "Bouts shorter than this are absorbed into their neighbours."
        )
        self.min_duration.valueChanged.connect(self._threshold_typed)
        form.addRow("minimum bout", self.min_duration)

        self.automatic = QPushButton("Use the automatic threshold")
        self.automatic.clicked.connect(self._use_automatic)
        form.addRow("", self.automatic)

        self.automatic_note = QLabel()
        self.automatic_note.setWordWrap(True)
        self.automatic_note.setStyleSheet("color: palette(mid);")
        form.addRow("", self.automatic_note)

        # -- no EMG at all
        self.surrogate_box = QGroupBox("No EMG channel")
        surrogate_layout = QVBoxLayout(self.surrogate_box)
        note = QLabel(
            "This recording has no EMG, so every epoch starts as SLEEP and "
            "clustering has to find wake in the EEG alone -- expect noticeably "
            "worse agreement, REM especially.\n\n"
            "If the file has other wideband channels, a surrogate built from "
            "them is far better than nothing."
        )
        note.setWordWrap(True)
        surrogate_layout.addWidget(note)
        surrogate_layout.addWidget(QLabel("<i>nyx.emg_from_lfp</i>"))
        self.surrogate_box.hide()

        layout.addWidget(controls_column(threshold_box, self.surrogate_box))

        # -- figures
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = FigureView(
            placeholder=(
                "Compute the EMG features to see the distribution and the "
                "power over time."
            )
        )
        right_layout.addWidget(self.figure, 3)

        self.hypnogram = PanelCanvas(figsize=(8, 1.8))
        right_layout.addWidget(self.hypnogram, 1)

        self.readout = QPlainTextEdit()
        self.readout.setReadOnly(True)
        self.readout.setFont(monospace())
        self.readout.setMaximumHeight(120)
        right_layout.addWidget(self.readout)

        layout.addWidget(right, 1)

    # -- the threshold -----------------------------------------------------

    def _emg(self):
        """The cached EMG features, or ``None`` if there are none."""
        try:
            return self.session.emg()
        except Exception:  # noqa: BLE001 - not computed yet is normal
            return None

    def _live_updates(self, emg) -> bool:
        return emg is not None and len(emg.power) <= _LIVE_LIMIT

    def _threshold_typed(self) -> None:
        """A spin box changed: move the line, then re-score."""
        if self._selector is not None:
            self._selector.set_value(self.threshold.value(), notify=False)
        if self._nosignal_selector is not None:
            self._nosignal_selector.set_value(self.nosignal.value(), notify=False)
        self._rescore()

    def _dragged(self, value: float) -> None:
        """During a drag: the label only, never the hypnogram."""
        emg = self._emg()
        if emg is None:
            return
        above = float((emg.power > value).mean())
        self.status.emit(f"threshold {value:.3f} -> {100 * above:.1f}% of epochs wake")

    def _released(self, value: float) -> None:
        with self.quiet(self.threshold):
            self.threshold.setValue(value)
        self._rescore()

    def _nosignal_released(self, value: float) -> None:
        with self.quiet(self.nosignal):
            self.nosignal.setValue(value)
        self._rescore()

    def _rescore(self) -> None:
        """Re-classify and redraw. Instant, so it needs no worker."""
        emg = self._emg()
        if emg is None:
            return
        self.session.set_emg_threshold(self.threshold.value(), nosignal=self.nosignal.value())
        self.session.set_min_duration(self.min_duration.value())
        self.session.compute(Stage.WAKE_SLEEP)
        self._draw_hypnogram()

    def _use_automatic(self) -> None:
        try:
            automatic = self.session.auto_threshold()
        except Exception as exc:  # noqa: BLE001
            self.status.emit(str(exc))
            return
        with self.quiet(self.threshold):
            self.threshold.setValue(automatic)
        if self._selector is not None:
            self._selector.set_value(automatic, notify=False)
        self._rescore()

    # -- drawing -----------------------------------------------------------

    def _draw_hypnogram(self) -> None:
        try:
            wake_sleep = self.session.wake_sleep()
        except Exception:  # noqa: BLE001
            return
        self.hypnogram.draw_panel(nyx.plot_wake_sleep, wake_sleep)

        durations: dict[str, float] = {}
        for duration, label in zip(
            wake_sleep.hypnogram["duration"], wake_sleep.hypnogram["label"], strict=True
        ):
            durations[str(label)] = durations.get(str(label), 0.0) + float(duration)
        total = sum(durations.values()) or 1.0

        lines = [
            f"threshold {wake_sleep.threshold:.3f} ({wake_sleep.threshold_source})",
            "",
        ]
        for label, seconds in sorted(durations.items()):
            lines.append(
                f"  {label:<12} {seconds / 3600:6.2f} h  ({100 * seconds / total:5.1f}%)"
            )
        self.readout.setPlainText("\n".join(lines))

    def apply(self) -> None:
        self.session.set_min_duration(self.min_duration.value())

    def refresh(self) -> None:
        with self.quiet(self.min_duration):
            if self.min_duration.value() == 0.0:
                self.min_duration.setValue(self.session.min_duration())

        emg = self._emg()
        if emg is None:
            if self.session.has(Stage.EMG):
                # Computed, and there is genuinely no EMG.
                self.surrogate_box.show()
                self.figure.show_message(
                    "This recording has no EMG channel, so there is no "
                    "threshold to choose."
                )
                self._draw_hypnogram()
            return

        self.surrogate_box.hide()

        with self.quiet(self.nosignal):
            self.nosignal.setValue(self.session.nosignal_threshold())
        with self.quiet(self.threshold):
            current = self.session.emg_threshold()
            if current is not None:
                self.threshold.setValue(float(current))

        try:
            automatic = self.session.auto_threshold()
            self.automatic_note.setText(f"Automatic: {automatic:.3f}")
        except Exception:  # noqa: BLE001
            self.automatic_note.setText("")

        self._draw_threshold_figure(emg)
        self._draw_hypnogram()

    def _draw_threshold_figure(self, emg) -> None:
        """Draw plot_emg_check and hang the draggable lines off its two panels.

        Reusing the composite rather than laying the two panels out again keeps
        its 1:2 width ratio, its labels and its no-EMG placeholders in one
        place.
        """
        figure = nyx.plot_emg_check(
            emg,
            threshold=self.threshold.value(),
            nosignal=self.nosignal.value(),
        )
        self.figure.set_figure(figure)

        histogram, trace = figure.axes[0], figure.axes[1]
        live = self._live_updates(emg)

        self._selector = ThresholdSelector(
            self.threshold.value(),
            hist_ax=histogram,
            trace_ax=trace,
            on_change=self._dragged if live else None,
            on_release=self._released,
            label="wake/sleep",
        )
        self._nosignal_selector = ThresholdSelector(
            self.nosignal.value(),
            hist_ax=histogram,
            color="darkorange",
            linestyle=":",
            on_release=self._nosignal_released,
            label="no signal",
        )

        if not live:
            self.status.emit(
                "Long power trace: the scoring updates when you release the "
                "mouse rather than while dragging."
            )
