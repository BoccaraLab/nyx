"""Tab 5 -- the finished scoring, and writing it out.

The overview knobs are exposed rather than fixed. ``plot_scoring_overview``
takes ``eeg_range``, ``emg_range``, ``eeg_fmax``, ``emg_fmax`` and ``palette``
precisely because no single colour scale suits every recording, and a
spectrogram you cannot read tells you nothing.

The figures are built here, on the main thread, from data a worker already
fetched. ``plot_scoring_overview`` reads the whole window's traces off disk,
which is the slow part -- but it also goes through pyplot, which is not thread
safe, so the read is what moves and the drawing stays.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.canvas import FigureView, PanelCanvas
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab, controls_column
from nyx.gui.widgets import monospace

__all__ = ["ResultTab"]


class ResultTab(Tab):
    title = "Result"
    subtitle = (
        "The scoring in context. Adjust the colour scales until the "
        "spectrograms are readable, then write everything out."
    )
    stage = Stage.RESULT
    needs_worker = False

    def build(self) -> None:
        layout = QHBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- display
        display_box = QGroupBox("Display")
        display_form = QFormLayout(display_box)

        self.eeg_low, self.eeg_high = _range_row(display_form, "EEG colour")
        self.emg_low, self.emg_high = _range_row(display_form, "EMG colour")

        self.eeg_fmax = QDoubleSpinBox()
        self.eeg_fmax.setRange(0.0, 1000.0)
        self.eeg_fmax.setSuffix(" Hz")
        self.eeg_fmax.setToolTip("0 uses the params' upper frequency.")
        display_form.addRow("EEG f max", self.eeg_fmax)

        self.emg_fmax = QDoubleSpinBox()
        self.emg_fmax.setRange(0.0, 1000.0)
        self.emg_fmax.setSuffix(" Hz")
        display_form.addRow("EMG f max", self.emg_fmax)

        self.palette = QComboBox()
        self.palette.addItems(
            ["jet", "viridis", "magma", "inferno", "turbo", "RdYlBu_r"]
        )
        display_form.addRow("palette", self.palette)

        redraw = QPushButton("Redraw")
        redraw.clicked.connect(self._draw_overview)
        display_form.addRow("", redraw)

        # -- review
        review_box = QGroupBox("Check it by hand")
        review_layout = QVBoxLayout(review_box)
        note = QLabel(
            "Scroll the traces and the spectrogram, and correct the "
            "hypnogram epoch by epoch. Anything you change here is what gets "
            "saved."
        )
        note.setWordWrap(True)
        review_layout.addWidget(note)
        self.review_button = QPushButton("Review and edit...")
        self.review_button.clicked.connect(self._open_review)
        review_layout.addWidget(self.review_button)
        self.edited_note = QLabel()
        self.edited_note.setWordWrap(True)
        self.edited_note.setStyleSheet("color: #e37400;")
        review_layout.addWidget(self.edited_note)

        # -- saving
        save_box = QGroupBox("Save")
        save_form = QFormLayout(save_box)
        self.output = QLineEdit()
        self.output.setPlaceholderText("results/<recording>")
        browse = QPushButton("Folder...")
        browse.clicked.connect(self._browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.output, 1)
        row_layout.addWidget(browse)
        save_form.addRow("into", row)

        self.granularities = QLineEdit()
        self.granularities.setPlaceholderText("5, 4, 3")
        self.granularities.setToolTip(
            "Extra hypnograms at coarser stage counts. Leave blank for none."
        )
        save_form.addRow("granularities", self.granularities)

        self.write_plots = QCheckBox("write the figures too")
        self.write_plots.setToolTip(
            "Re-renders every panel to PNG. Slow, and redundant when they are "
            "already on screen."
        )
        save_form.addRow("", self.write_plots)

        self.save_emg = QCheckBox("save the EMG power trace")
        self.save_emg.setChecked(True)
        save_form.addRow("", self.save_emg)

        save = QPushButton("Save")
        save.clicked.connect(self._save)
        save_form.addRow("", save)

        layout.addWidget(controls_column(display_box, review_box, save_box))

        # -- figures
        self.figures = QTabWidget()
        self.overview = FigureView(placeholder="Score a recording to see it here.")
        self.summary_view = FigureView(placeholder="The nine-panel summary.")
        self.confusion = PanelCanvas(figsize=(5, 4.5))
        self.agreement = QPlainTextEdit()
        self.agreement.setReadOnly(True)
        self.agreement.setFont(monospace())
        self.agreement.setPlaceholderText(
            "No manual scoring was loaded, so there is nothing to compare "
            "against. That is a normal way to run nyx."
        )

        self.figures.addTab(self.overview, "Overview")
        self.figures.addTab(self.summary_view, "Summary")
        self.figures.addTab(self.confusion, "Confusion")
        self.figures.addTab(self.agreement, "Agreement")
        layout.addWidget(self.figures, 1)

    # -- actions -----------------------------------------------------------

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Where to save", self.output.text() or ""
        )
        if path:
            self.output.setText(path)

    def _open_review(self) -> None:
        try:
            from nyx.gui.review import open_review_window
        except ImportError as exc:
            self.status.emit(
                f"The review window needs ephyviewer: pip install "
                f'"nyx-sleep[gui]"  ({exc})'
            )
            return
        try:
            self._review = open_review_window(self.session, parent=self)
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"Could not open the review window: {exc}")

    def _save(self) -> None:
        output = self.output.text().strip()
        if not output:
            self.status.emit("Choose somewhere to save first.")
            return

        granularities = [
            int(part) for part in self.granularities.text().replace(",", " ").split()
        ] or None

        try:
            written = self.session.save(
                output,
                plots=self.write_plots.isChecked(),
                granularities=granularities,
                save_emg_power=self.save_emg.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001
            self.status.emit(f"Could not save: {exc}")
            return

        self.status.emit(f"Saved to {written}")
        for name in sorted(os.listdir(written)):
            self.status.emit(f"  {name}")

    # -- drawing -----------------------------------------------------------

    def _overview_kwargs(self) -> dict:
        def span(low, high):
            return None if low.value() == high.value() else (low.value(), high.value())

        return {
            "eeg_range": span(self.eeg_low, self.eeg_high),
            "emg_range": span(self.emg_low, self.emg_high),
            "eeg_fmax": self.eeg_fmax.value() or None,
            "emg_fmax": self.emg_fmax.value() or None,
            "palette": self.palette.currentText(),
        }

    def _draw_overview(self) -> None:
        if not self.session.has(Stage.STEPS):
            return
        result = self.session.result()
        self.status.emit("Drawing the overview...")
        self.overview.set_figure(
            nyx.plot_scoring_overview(result, **self._overview_kwargs())
        )
        self.status.emit("Drawn.")

    def refresh(self) -> None:
        if not self.session.has(Stage.STEPS):
            return

        if not self.output.text().strip():
            name = self.session.recording.name or "recording"
            self.output.setText(os.path.join("results", name))

        self.edited_note.setText(
            "This hypnogram has been edited by hand. The edit is what will be "
            "saved, and it is recorded in run.json."
            if self.session.was_edited() else ""
        )

        result = self.session.result()
        self.overview.set_figure(
            nyx.plot_scoring_overview(result, **self._overview_kwargs())
        )
        self.summary_view.set_figure(nyx.plot_summary(result))

        if result.agreement is not None:
            self.confusion.draw_panel(nyx.report.plot_confusion, result)
            self.agreement.setPlainText(result.agreement.summary())
        else:
            self.confusion.clear()
            self.agreement.setPlainText("")


def _range_row(form: QFormLayout, label: str):
    """A low/high pair on one row. Equal values mean "let the plot decide"."""
    first, second = QDoubleSpinBox(), QDoubleSpinBox()
    for spin in (first, second):
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(0.0)
        spin.setToolTip(
            "Leave both at the same value to let the plot choose its own scale."
        )
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(first)
    layout.addWidget(QLabel("to"))
    layout.addWidget(second)
    form.addRow(label, row)
    return first, second
