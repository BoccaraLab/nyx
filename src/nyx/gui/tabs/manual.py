"""Scoring by hand -- the last tab of ``nyx-manual``.

The recording and the signal check come first, exactly as in ``nyx-gui``: pick
the EEG and the EMG, notch and trim. Then this tab shows the traces and their
time-frequency views, and a hypnogram you fill in one epoch at a time. The
epoch length is yours to choose; once chosen, nothing can be scored off it
(see :mod:`nyx.gui.manual`).

A reference scoring, if one was loaded on the Recording tab, stays hidden
until you ask to compare. Seeing it while you score would defeat the point of
scoring -- and of training on a recording someone has already scored.
"""

from __future__ import annotations

import os

import numpy as np
from ephyviewer import TraceViewer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nyx.gui.hypnogram import EpochGrid, onto_grid, stage_palette
from nyx.gui.manual import DEFAULT_LABELS, GridEpochEncoder, GridEpochSource
from nyx.gui.panels import TextPanel
from nyx.gui.session import Stage
from nyx.gui.sources import autoscale, trace_sources
from nyx.gui.tabs.base import Tab
from nyx.gui.viewers import make_timefreq_viewer
from nyx.gui.widgets import Section, help_label

__all__ = ["ManualTab"]

_EMPTY = {"time": np.array([], "float64"), "duration": np.array([], "float64"),
          "label": np.array([], "U16")}

class ManualTab(Tab):
    title = "Manual scoring"
    subtitle = (
        "Press a stage's number to score the epoch under the cursor and move to "
        "the next. Every epoch is exactly one epoch long: nothing can be "
        "scored off the grid. Alt+arrows jump between changes of stage."
    )
    stage = Stage.PREPROCESS
    run_label = "Open the recording for scoring"
    run_at_top = True

    def build_controls(self) -> list:
        self._built_for = None
        #: The scoring, on the *recording's* clock, whenever the viewers are
        #: rebuilt -- so changing the window or the view does not lose it.
        self._scored = dict(_EMPTY)
        self._saved_state = self._state_key(self._scored)
        self.source = None
        self.encoder = None

        epochs_box = Section(
            "Epochs",
            "The length of one scored epoch, and the stages the number keys "
            "give: 1 is the first stage listed, 2 the second, and so on.\n\n"
            "The grid starts at the start of the recording, not of the "
            "window, so trimming the window does not move it -- and a "
            "scoring saved here lines up with a reference scored on the whole "
            "recording.\n\n"
            "Both are fixed once anything has been scored: changing the "
            "epoch length would leave the epochs already scored off the grid.",
        )
        epochs_form = QFormLayout(epochs_box)
        self.epoch_length = QDoubleSpinBox()
        self.epoch_length.setRange(0.5, 600.0)
        self.epoch_length.setDecimals(1)
        self.epoch_length.setSingleStep(1.0)
        self.epoch_length.setSuffix(" s")
        self.epoch_length.setValue(4.0)
        self.epoch_length.valueChanged.connect(self._rebuild)
        epochs_form.addRow("epoch length", self.epoch_length)

        self.labels = QLineEdit(", ".join(DEFAULT_LABELS))
        self.labels.setToolTip("The stages, in the order the number keys pick them.")
        self.labels.editingFinished.connect(self._rebuild)
        epochs_form.addRow("stages", self.labels)
        self.keys_note = help_label("")
        epochs_form.addRow("", self.keys_note)

        view_box = Section(
            "View",
            "The wavelet view is a Morlet scalogram in dB, the transform nyx's "
            "scalogram backend computes -- easier to read than the Fourier "
            "view, and slower. The band shown is the params' for each "
            "channel; the Signal check tab changes it.",
        )
        view_form = QFormLayout(view_box)
        self.scalogram = QCheckBox("wavelet (slower, easier to read)")
        self.scalogram.setChecked(True)
        self.scalogram.toggled.connect(self._rebuild)
        view_form.addRow("", self.scalogram)
        self.palette = QComboBox()
        self.palette.addItems(["jet", "viridis", "magma", "inferno", "turbo", "gray"])
        self.palette.currentTextChanged.connect(self._set_palette)
        view_form.addRow("colours", self.palette)

        file_box = Section(
            "Hypnogram file",
            "A CSV of time, duration and label, in seconds from the start of "
            "the recording -- the format nyx reads a reference in "
            "(interval_csv). Only what has been scored is written.\n\n"
            "Continue opens a scoring saved earlier. One made at another "
            "epoch length is put on this grid: each epoch takes the stage "
            "that covers most of it.",
        )
        file_form = QFormLayout(file_box)
        self.path = QLineEdit()
        self.path.setPlaceholderText("<recording>_manual.csv")
        browse = QPushButton("...")
        browse.clicked.connect(self._browse_save)
        file_form.addRow("save to", _row(self.path, browse))
        buttons = QWidget()
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        save = QPushButton("Save")
        save.setToolTip("Same as Ctrl+S in the hypnogram panel.")
        save.clicked.connect(self.save)
        resume = QPushButton("Continue a scoring...")
        resume.clicked.connect(self._browse_resume)
        buttons_layout.addWidget(save)
        buttons_layout.addWidget(resume)
        file_form.addRow("", buttons)
        self.progress = help_label("Nothing scored yet.")
        file_form.addRow("", self.progress)

        compare_box = Section(
            "Compare",
            "Against the manual scoring loaded on the Recording tab, if you "
            "loaded one. It stays hidden until you press this, so it cannot "
            "steer your scoring.\n\n"
            "Only the epochs you have scored are compared. The reference "
            "appears tabbed with your hypnogram, read-only, with the "
            "agreement beside it.",
        )
        compare_layout = QVBoxLayout(compare_box)
        self.compare = QPushButton("Compare with the reference")
        self.compare.setCheckable(True)
        self.compare.toggled.connect(self._toggle_compare)
        compare_layout.addWidget(self.compare)
        self.compare_note = help_label("")
        compare_layout.addWidget(self.compare_note)

        return [epochs_box, view_box, file_box, compare_box]

    def build_docks(self) -> None:
        self.agreement = TextPanel(
            "agreement", placeholder="Press Compare to see the agreement."
        )

    # -- the scoring -------------------------------------------------------

    def _stages(self) -> list[str]:
        names = [s.strip() for s in self.labels.text().split(",") if s.strip()]
        return list(dict.fromkeys(names)) or list(DEFAULT_LABELS)

    def _grid(self) -> EpochGrid:
        start, end = self.session.window
        _eeg, _emg, offset = trace_sources(self.session.windowed())
        return EpochGrid.for_window(
            self.epoch_length.value(), start, end - start, t_offset=offset
        )

    def _viewer_to_recording(self) -> float:
        """Add this to a viewer time to get a recording time."""
        _eeg, _emg, offset = trace_sources(self.session.windowed())
        return float(self.session.window[0]) - offset

    def _capture(self) -> None:
        """Copy the scoring out of the editor before it is torn down."""
        if self.source is None:
            return
        shift = self._viewer_to_recording()
        scored = self.source.scored()
        self._scored = {**scored, "time": scored["time"] + shift}

    def scored(self) -> dict:
        """The scoring so far, on the recording's clock."""
        self._capture()
        return {k: np.asarray(v).copy() for k, v in self._scored.items()}

    def _lock(self, locked: bool) -> None:
        for widget in (self.epoch_length, self.labels):
            widget.setEnabled(not locked)
        if locked:
            self.epoch_length.setToolTip(
                "Fixed once something has been scored. Save, then start a new "
                "scoring to use another length."
            )

    def _state_key(self, hypnogram) -> tuple:
        return tuple(
            (round(float(t), 6), round(float(d), 6), str(label))
            for t, d, label in zip(
                hypnogram["time"], hypnogram["duration"], hypnogram["label"],
                strict=True,
            )
        )

    def unsaved(self) -> bool:
        return self._state_key(self.scored()) != self._saved_state

    # -- panels ------------------------------------------------------------

    def _rebuild(self) -> None:
        if len(self.scored()["time"]) and self.source is not None:
            # Locked in the UI already; this catches a programmatic change.
            self._lock(True)
        self._built_for = None
        self.safe_refresh()

    def _build_viewers(self) -> None:
        windowed = self.session.windowed()
        key = (id(windowed), self.epoch_length.value(), tuple(self._stages()),
               self.scalogram.isChecked())
        if self._built_for == key:
            return
        self._capture()
        self.docks.clear()
        self.source = self.encoder = None
        self._built_for = key

        eeg, emg, offset = trace_sources(windowed)
        params = self.session.params
        if emg is not None:
            self.docks.add(autoscale(TraceViewer(source=emg, name="EMG")))
        self.docks.add(autoscale(TraceViewer(source=eeg, name="EEG")))
        if emg is not None:
            self.docks.add(self._timefreq(emg, "EMG spectrum", "EMG", params))
        self.docks.add(self._timefreq(eeg, "EEG spectrum", "EEG", params))

        grid = self._grid()
        stages = self._stages()
        shift = self._viewer_to_recording()
        on_viewer = {**self._scored, "time": self._scored["time"] - shift}
        _names, colors = stage_palette(stages)
        colors = dict(zip(_names, colors, strict=True))
        self.source = GridEpochSource(
            onto_grid(on_viewer, grid) if len(on_viewer["time"]) else dict(_EMPTY),
            grid, labels=stages, colors=[colors[s] for s in stages],
            on_save=lambda _hypnogram: self.save(),
        )
        self.encoder = GridEpochEncoder(source=self.source, name="hypnogram")
        self.encoder.edited.connect(self._update_progress)
        self.docks.add(self.encoder)
        # A stage key only works with the hypnogram focused; start there.
        self.encoder.setFocus()

        self.keys_note.setText(
            "  ".join(f"{i + 1}={name}" for i, name in enumerate(stages[:10]))
        )
        if self.compare.isChecked():
            self._add_reference()
        self._update_progress()

    def _timefreq(self, source, name: str, channel: str, params: dict):
        viewer = make_timefreq_viewer(
            source, name, params, channel, self.scalogram.isChecked()
        )
        try:
            viewer.params["colormap"] = self.palette.currentText()
            # The dock already says which channel it is; the "0: EEG" title
            # over the image only takes height from it.
            viewer.params["display_labels"] = False
        except Exception:  # noqa: BLE001
            pass
        return viewer

    def _set_palette(self, name: str) -> None:
        for title in ("EEG spectrum", "EMG spectrum"):
            viewer = self.docks.panel(title)
            if viewer is not None:
                try:
                    viewer.params["colormap"] = name
                except Exception:  # noqa: BLE001
                    pass

    def _update_progress(self) -> None:
        if self.source is None:
            return
        done, total = self.source.n_scored(), self.source.grid.n_epochs()
        self._lock(done > 0)
        share = f" ({100 * done / total:.0f}%)" if total else ""
        self.progress.setText(f"{done} of {total} epochs scored{share}.")

    # -- compare -----------------------------------------------------------

    def _toggle_compare(self, on: bool) -> None:
        if not on:
            self.docks.remove("reference")
            self.docks.remove("agreement")
            self.compare_note.setText("")
            return
        if self.session.reference is None:
            self.compare_note.setText(
                "No manual scoring was loaded on the Recording tab."
            )
            self.compare.setChecked(False)
            return
        if self.source is None:
            self.compare_note.setText("Open the recording for scoring first.")
            self.compare.setChecked(False)
            return
        self._add_reference()

    def _add_reference(self) -> None:
        from nyx.gui.review import NyxEpochSource
        from nyx.gui.viewers import NyxEpochEncoder
        from nyx.metrics import normalise_labels, trim_manual_scores

        start, end = self.session.window
        _eeg, _emg, offset = trace_sources(self.session.windowed())
        reference = trim_manual_scores(
            normalise_labels(self.session.reference), start, end
        )
        if "reference" not in self.docks.viewers:
            encoder = NyxEpochEncoder(
                source=NyxEpochSource(reference, name="reference", t_offset=offset),
                name="reference", rules=[], epoch_length=self.epoch_length.value(),
            )
            encoder.make_read_only()
            self.docks.add(encoder, tabify_with="hypnogram")
            self.docks.add(self.agreement, tabify_with="reference")
        self._compare_text()

    def _compare_text(self) -> None:
        agreement = self.agreement_with_reference()
        if agreement is None:
            self.agreement.set_text(
                "Nothing to compare yet: score some epochs inside the window."
            )
            self.compare_note.setText("")
            return
        done = self.source.n_scored() if self.source is not None else 0
        self.agreement.set_text(
            f"Over the {done} epochs you have scored:\n\n{agreement.summary()}"
        )
        self.compare_note.setText(
            f"MF1 {agreement.mf1:.2f}, accuracy {agreement.accuracy:.2f}. "
            "The reference is tabbed with your hypnogram."
        )

    def agreement_with_reference(self):
        """Your scoring against the reference, over the epochs you scored."""
        import nyx

        if self.session.reference is None:
            return None
        scored = self.scored()
        if not len(scored["time"]):
            return None
        start, end = self.session.window
        relative = {**scored, "time": scored["time"] - start}
        return nyx.evaluate(
            relative, self.session.reference, window=(start, end), verbose=False
        )

    # -- files -------------------------------------------------------------

    def _default_path(self) -> str:
        recording = self.session.recording
        folder = os.path.dirname(recording.source_path or "") or os.getcwd()
        return os.path.join(folder, f"{recording.name or 'recording'}_manual.csv")

    def _browse_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the scoring as", self.path.text() or self._default_path(),
            "Hypnogram (*.csv)",
        )
        if path:
            self.path.setText(path)

    def save(self) -> str | None:
        """Write the scoring to the chosen CSV. Returns the path written."""
        from nyx.scoring import savehypno

        if self.source is None and not len(self._scored["time"]):
            self.status.emit("Nothing to save yet.")
            return None
        path = self.path.text().strip()
        if not path:
            self._browse_save()
            path = self.path.text().strip()
            if not path:
                return None
        scored = self.scored()
        savehypno(scored, path)
        self._saved_state = self._state_key(scored)
        if self.encoder is not None:
            self.encoder.changes_since_save = 0
            self.encoder.refresh_toolbar()
        self.saved.emit(os.path.dirname(os.path.abspath(path)))
        self.status.emit(f"Saved {len(scored['time'])} segments to {path}.")
        if self.compare.isChecked():
            self._compare_text()
        return path

    def _browse_resume(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Continue a scoring", self.path.text() or "",
            "Hypnogram (*.csv *.txt);;All files (*)",
        )
        if path:
            self.resume(path)

    def resume(self, path: str) -> None:
        """Open a saved scoring to carry on with it."""
        import nyx

        hypnogram = nyx.read_annotations(path, format="interval_csv")
        self._scored = {
            "time": np.asarray(hypnogram["time"], dtype="float64"),
            "duration": np.asarray(hypnogram["duration"], dtype="float64"),
            "label": np.asarray(hypnogram["label"]).astype("U16"),
        }
        stages = self._stages()
        extra = [s for s in dict.fromkeys(self._scored["label"]) if s not in stages]
        if extra:
            self.labels.setText(", ".join([*stages, *extra]))
        self.source = None       # so the rebuild takes _scored, not the editor
        self.path.setText(path)
        self._saved_state = self._state_key(self._scored)
        self._built_for = None
        self.safe_refresh()
        self.status.emit(f"Continuing the scoring in {path}.")

    def confirm_close(self) -> bool:
        """Ask before closing over unsaved scoring. ``False`` means stay."""
        if not self.unsaved():
            return True
        answer = QMessageBox.question(
            self, "Unsaved scoring",
            "The manual scoring has changes that are not saved. Save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save() is not None
        return True

    # -- tab contract ------------------------------------------------------

    def refresh(self) -> None:
        if not self.session.has(Stage.PREPROCESS):
            return
        self._build_viewers()
        if self.compare.isChecked():
            self._compare_text()


def _row(*widgets) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if index == 0 else 0)
    return holder
