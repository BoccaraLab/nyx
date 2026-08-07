"""Tab 3 -- the EMG threshold, dragged.

The notebooks' third STOP, and the one that most rewards being interactive.
The cut is a single number, but choosing it well means watching three things:
the distribution, where it should sit in a valley; the power trace, where it
should track what the animal is obviously doing; and the wake/sleep bouts that
come out, which is what you are actually deciding.

So all three are here, locked to one clock, and the same line is drawn on the
distribution and the trace. Dragging either moves both, and the bouts redraw
when you let go -- ``classify_wake_sleep`` with an explicit threshold is
milliseconds, and the features it works from are cached.

The automatic threshold is *not* milliseconds -- it fits four GaussianMixtures
and picks by BIC -- so it is fitted once, when the features are computed.

Two things this tab has to survive:

*A recording with no EMG.* ``compute_emg_features`` returns ``None``, every
epoch starts as SLEEP, and clustering has to find wake in the EEG alone.

*The scalogram backend*, whose power vector is one value per **sample** rather
than per epoch. The run-length encoding behind ``classify_wake_sleep`` is a
Python loop over it, so past a certain size the bouts redraw on release only.
"""

from __future__ import annotations

from ephyviewer import TraceViewer
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from nyx.gui.histogram import HistogramViewer, ThresholdLines
from nyx.gui.session import Stage
from nyx.gui.sources import power_source, trace_sources
from nyx.gui.tabs.base import Tab
from nyx.gui.widgets import Section, help_label, monospace

__all__ = ["EmgTab"]

#: Past this many samples the bouts redraw on release rather than during a drag.
LIVE_LIMIT = 200_000


class EmgTab(Tab):
    title = "EMG threshold"
    subtitle = (
        "Drag the red line, on the distribution or on the trace -- they are "
        "the same number. The bouts below redraw when you let go."
    )
    run_at_top = True
    stage = Stage.WAKE_SLEEP
    run_label = "Compute the EMG features"

    def build_controls(self) -> list:
        self._lines: ThresholdLines | None = None
        self._built_for = None

        box = Section(
            "Threshold",
            "Where the wake/sleep cut goes. Drag the red line on the "
            "distribution or on the power trace -- they are the same number "
            "shown twice. It should sit in the valley of the distribution "
            "and track what the animal is obviously doing.\n\n"
            "The no-signal cut marks power at or below it as signal loss "
            "rather than deep sleep; 0 disables it.\n\n"
            "Bouts shorter than the minimum are absorbed into their "
            "neighbours. The bouts redraw when you let go of the line.",
        )
        form = QFormLayout(box)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e6, 1e6)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.valueChanged.connect(self._typed)
        form.addRow("wake / sleep", self.threshold)

        self.nosignal = QDoubleSpinBox()
        self.nosignal.setRange(-1e6, 1e6)
        self.nosignal.setDecimals(3)
        self.nosignal.setSingleStep(0.01)
        self.nosignal.setToolTip(
            "Power at or below this is signal loss rather than deep sleep. "
            "0 disables it."
        )
        self.nosignal.valueChanged.connect(self._typed)
        form.addRow("no signal", self.nosignal)

        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0.0, 600.0)
        self.min_duration.setDecimals(1)
        self.min_duration.setSuffix(" s")
        self.min_duration.setToolTip(
            "Bouts shorter than this are absorbed into their neighbours."
        )
        self.min_duration.valueChanged.connect(self._typed)
        form.addRow("minimum bout", self.min_duration)

        self.automatic = QPushButton("Use the automatic threshold")
        self.automatic.clicked.connect(self._use_automatic)
        form.addRow("", self.automatic)

        self.confirm = QPushButton("Confirm these thresholds")
        self.confirm.setMinimumHeight(30)
        self.confirm.setStyleSheet("font-weight: bold;")
        self.confirm.setToolTip(
            "Take the cut as it stands and score the wake/sleep bouts with it. "
            "Dragging already re-scores, so this is for saying you are done."
        )
        self.confirm.clicked.connect(self._confirm)
        form.addRow("", self.confirm)

        self.automatic_note = QLabel()
        self.automatic_note.setWordWrap(True)
        self.automatic_note.setStyleSheet("font-size: 11px;")
        form.addRow("", self.automatic_note)

        # In the controls rather than a panel of its own: it is three lines,
        # and a dock for three lines steals width from the traces.
        self.readout = QLabel()
        self.readout.setWordWrap(True)
        self.readout.setFont(monospace())
        self.readout.setStyleSheet("font-size: 11px;")
        form.addRow("", self.readout)

        self.no_emg_box = Section(
            "No EMG channel",
            "Wake has to be recovered from the EEG spectrum alone, where "
            "quiet wake and REM look much alike, so expect noticeably worse "
            "agreement -- REM especially.\n\n"
            "If the file has other wideband channels, nyx.emg_from_lfp builds "
            "a surrogate from them, which is far better than nothing.",
        )
        no_emg_layout = QVBoxLayout(self.no_emg_box)
        no_emg_layout.addWidget(help_label(
            "Every epoch starts as SLEEP; clustering has to find wake itself."
        ))
        self.no_emg_box.hide()

        return [box, self.no_emg_box]

    def build_docks(self) -> None:
        pass

    # -- panels ------------------------------------------------------------

    def _emg(self):
        try:
            return self.session.emg()
        except Exception:  # noqa: BLE001 - not computed yet is normal
            return None

    def _build_viewers(self) -> None:
        """Distribution, power trace, raw EMG and the resulting bouts."""
        emg = self._emg()
        key = (self.session.windowed(), id(emg))
        if self._built_for == key:
            return

        self.docks.clear()
        self._built_for = key

        if emg is None:
            return

        _eeg, emg_trace, offset = trace_sources(self.session.windowed())

        # Top to bottom, this is the decision being made, in order: the
        # signal, the power it is summarised into, the distribution of that
        # power with the cut on it, and the bouts that fall out of the cut.
        if emg_trace is not None:
            self.docks.add(TraceViewer(source=emg_trace, name="EMG"))

        self.power = TraceViewer(
            source=power_source(emg, offset), name="EMG power"
        )
        # The power is min-max scaled to [0, 1], and auto-scaling a trace whose
        # tails are near-flat leaves the interesting part in a sliver.
        _fix_range(self.power, -0.01, 1.01)
        self.docks.add(self.power)

        self.histogram = HistogramViewer(name="EMG distribution")
        self.histogram.set_values(emg.power, "EMG power (scaled)")
        self.docks.add(self.histogram)

        if self.session.has(Stage.WAKE_SLEEP):
            self.docks.add(self._encoder(offset))

        # One value, two views. Dragging either moves both, which is the whole
        # point of having both.
        self._lines = ThresholdLines(
            [self.nosignal.value(), self.threshold.value()],
            on_change=self._dragging,
            on_release=self._released,
            colours=["#ff8c00", "#dc143c"],
            labels=["no signal", "wake / sleep"],
        )
        self.histogram.attach(self._lines)
        self._lines.add(self.power.plot, "h")

        # The traces are read along, the distribution across; give the traces
        # the height and leave the distribution enough to see the valley in.
        names = ["EMG", "EMG power", "EMG distribution", "wake / sleep"]
        docks = [self.docks.viewers[n]["dock"]
                 for n in names if n in self.docks.viewers]
        sizes = [300, 260, 220, 160][: len(docks)]
        self.docks.resizeDocks(docks, sizes, Qt.Vertical)

    def _encoder(self, offset: float):
        """The wake/sleep bouts, editable, in nyx's stage colours.

        An encoder rather than a viewer: the automatic cut is a starting point,
        and a bout you can see is wrong should be fixable where you see it.
        """
        from nyx.gui.review import NyxEpochSource
        from nyx.gui.viewers import NyxEpochEncoder

        self.epoch_source = NyxEpochSource(
            self.session.wake_sleep().hypnogram,
            name="wake / sleep",
            t_offset=offset,
            on_save=self._apply_edit,
        )
        self.encoder = NyxEpochEncoder(
            source=self.epoch_source, name="wake / sleep",
            rules=self.session.postprocess_rules(),
        )
        return self.encoder

    def _apply_edit(self, hypnogram) -> None:
        """A hand-edited wake/sleep split replaces the thresholded one."""
        from dataclasses import replace as _replace

        self.session._wake_sleep = _replace(
            self.session.wake_sleep(), hypnogram=hypnogram,
            threshold_source="manual",
        )
        self.session.invalidate(Stage.STEPS)
        self.status.emit("Wake/sleep edit kept.")
        self._update_readout()

    def _confirm(self) -> None:
        self._rescore()
        self.status.emit(
            f"Threshold {self.threshold.value():.3f} confirmed. "
            f"Move on to the sleep stages."
        )

    # -- the threshold -----------------------------------------------------

    def _live(self) -> bool:
        emg = self._emg()
        return emg is not None and len(emg.power) <= LIVE_LIMIT

    def _dragging(self, values) -> None:
        """During a drag: a readout, never a re-scoring."""
        emg = self._emg()
        if emg is None or not self._live():
            return
        above = float((emg.power > values[1]).mean())
        self.status.emit(
            f"threshold {values[1]:.3f} -> {100 * above:.1f}% of epochs wake"
        )

    def _released(self, values) -> None:
        with self.quiet(self.nosignal):
            self.nosignal.setValue(values[0])
        with self.quiet(self.threshold):
            self.threshold.setValue(values[1])
        self._rescore()

    def _typed(self) -> None:
        if self._lines is not None:
            self._lines.set_all(
                [self.nosignal.value(), self.threshold.value()], notify=False
            )
        self._rescore()

    def _rescore(self) -> None:
        """Re-classify and redraw the bouts. Instant, so no worker."""
        if self._emg() is None:
            return
        self.session.set_emg_threshold(
            self.threshold.value(), nosignal=self.nosignal.value()
        )
        self.session.set_min_duration(self.min_duration.value())
        self.session.compute(Stage.WAKE_SLEEP)
        self._update_bouts()

    def _use_automatic(self) -> None:
        try:
            automatic = self.session.auto_threshold()
        except Exception as exc:  # noqa: BLE001
            self.status.emit(str(exc))
            return
        self.threshold.setValue(automatic)

    # -- readouts ----------------------------------------------------------

    def _update_bouts(self) -> None:
        """Swap the epoch source in place rather than rebuilding the panels.

        Rebuilding is what made the view jump on every release: the whole dock
        area was torn down and put back, losing the scroll position with it.
        """
        try:
            wake_sleep = self.session.wake_sleep()
        except Exception:  # noqa: BLE001
            return

        _eeg, _emg, offset = trace_sources(self.session.windowed())
        panel = self.docks.panel("wake / sleep")
        if panel is None:
            self.docks.add(self._encoder(offset), split_with="EMG power",
                           orientation="vertical")
        else:
            from nyx.gui.hypnogram import to_epoch_dict

            epoch = to_epoch_dict(wake_sleep.hypnogram, "wake / sleep", offset)
            source = panel.source
            source._clean_and_set(
                epoch["time"], epoch["duration"], epoch["label"],
                __import__("numpy").arange(len(epoch["time"])),
            )
            panel.refresh_flags()
            panel.refresh()
            panel.refresh_table()

        self._update_readout()

    def _update_readout(self) -> None:
        try:
            wake_sleep = self.session.wake_sleep()
        except Exception:  # noqa: BLE001
            return

        durations: dict[str, float] = {}
        for duration, label in zip(
            wake_sleep.hypnogram["duration"], wake_sleep.hypnogram["label"],
            strict=True,
        ):
            durations[str(label)] = durations.get(str(label), 0.0) + float(duration)
        total = sum(durations.values()) or 1.0

        lines = [
            f"threshold {wake_sleep.threshold:.3f} ({wake_sleep.threshold_source})",
            "",
        ]
        for label, seconds in sorted(durations.items()):
            lines.append(
                f"  {label:<12} {seconds / 3600:6.2f} h  "
                f"({100 * seconds / total:5.1f}%)"
            )
        self.readout.setText("\n".join(lines))

    # -- session -----------------------------------------------------------

    def apply(self) -> None:
        self.session.set_min_duration(self.min_duration.value())

    def refresh(self) -> None:
        with self.quiet(self.min_duration):
            if self.min_duration.value() == 0.0:
                self.min_duration.setValue(self.session.min_duration())

        emg = self._emg()
        if emg is None:
            if self.session.has(Stage.EMG):
                self.no_emg_box.show()
                self._build_viewers()
                self.readout.setText(
                    "No EMG channel, so there is no threshold to choose.\n"
                    "Every epoch starts as SLEEP."
                )
            return

        self.no_emg_box.hide()

        with self.quiet(self.nosignal):
            self.nosignal.setValue(self.session.nosignal_threshold())
        with self.quiet(self.threshold):
            current = self.session.emg_threshold()
            if current is not None:
                self.threshold.setValue(float(current))

        try:
            self.automatic_note.setText(
                f"Automatic: {self.session.auto_threshold():.3f}"
            )
        except Exception:  # noqa: BLE001
            self.automatic_note.setText("")

        self._build_viewers()
        if self._lines is not None:
            self._lines.set_all(
                [self.nosignal.value(), self.threshold.value()], notify=False
            )
        if self.session.has(Stage.WAKE_SLEEP):
            self._update_bouts()
        if not self._live():
            self.status.emit(
                "Long power trace: the bouts redraw when you release the "
                "mouse rather than while dragging."
            )


def _fix_range(viewer, low: float, high: float) -> None:
    """Pin a trace viewer's y range instead of letting it auto-scale.

    Auto-scaling a min-max scaled power trace puts the whole distribution in a
    sliver, because its tails run right to the edges.
    """
    try:
        viewer.params["ylim_min"] = low
        viewer.params["ylim_max"] = high
        viewer.params["auto_scale_factor"] = 1.0
        for i in range(viewer.source.nb_channel):
            viewer.by_channel_params[f"ch{i}", "gain"] = 1.0
            viewer.by_channel_params[f"ch{i}", "offset"] = 0.0
    except Exception:  # noqa: BLE001 - parameter names vary between releases
        pass
