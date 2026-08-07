"""Tab 5 -- the finished scoring, checked and written out.

This is where a GUI earns its existence. A notebook can print an agreement
table; what it cannot do is let you scroll to the twelve minutes where nyx and
the reference disagree and see *why*. So the scoring, the reference, the traces
and the spectrogram are all here, on one clock, and the hypnogram is editable
in place -- ``alt``+arrows for the next change of stage, ``ctrl``+arrows for
the next epoch a postprocessing rule objects to.

What you edit is what gets saved: the corrected hypnogram goes back into the
session, agreement is recomputed against it, and ``run.json`` records that a
human changed it.
"""

from __future__ import annotations

import os

from ephyviewer import EpochViewer, TraceViewer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.panels import TextPanel
from nyx.gui.session import Stage
from nyx.gui.sources import component_source, epoch_source, trace_sources
from nyx.gui.tabs.base import Tab
from nyx.gui.viewers import make_timefreq_viewer

__all__ = ["ResultTab"]


class ResultTab(Tab):
    title = "Result"
    subtitle = (
        "Scroll to where nyx and the reference disagree, and fix it. "
        "Alt+arrows jump to the next change of stage, Ctrl+arrows to the next "
        "epoch a rule objects to."
    )
    stage = Stage.RESULT
    needs_worker = False

    def build_controls(self) -> list:
        self._built_for = None

        view_box = QGroupBox("Spectrogram")
        view_form = QFormLayout(view_box)
        self.scalogram = QCheckBox("wavelet (slower, easier to read)")
        self.scalogram.setToolTip(
            "A Morlet scalogram in dB, as the scalogram feature backend "
            "computes it. Prettier and much slower than the Fourier view."
        )
        self.scalogram.toggled.connect(self._rebuild)
        view_form.addRow("", self.scalogram)

        self.palette = QComboBox()
        self.palette.addItems(
            ["jet", "viridis", "magma", "inferno", "turbo", "gray"]
        )
        self.palette.currentTextChanged.connect(self._set_palette)
        view_form.addRow("colours", self.palette)

        compare_box = QGroupBox("Compare")
        compare_layout = QVBoxLayout(compare_box)
        self.show_reference = QPushButton("Show the reference")
        self.show_reference.setCheckable(True)
        self.show_reference.setToolTip(
            "Add the manual scoring as a second, read-only hypnogram, "
            "tabbed with yours so you can flip between them in place."
        )
        self.show_reference.toggled.connect(self._toggle_reference)
        compare_layout.addWidget(self.show_reference)
        self.compare_note = QLabel()
        self.compare_note.setWordWrap(True)
        self.compare_note.setStyleSheet("color: palette(mid);")
        compare_layout.addWidget(self.compare_note)

        edit_box = QGroupBox("Editing")
        edit_layout = QFormLayout(edit_box)
        self.edited_note = QLabel("Not edited.")
        self.edited_note.setWordWrap(True)
        edit_layout.addRow("", self.edited_note)
        keep = QPushButton("Keep my edits")
        keep.setToolTip(
            "Take the hypnogram as it is now. Also what Ctrl+S in the "
            "hypnogram panel does."
        )
        keep.clicked.connect(self._keep_edits)
        edit_layout.addRow("", keep)

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
            "Extra hypnograms at coarser stage counts. Blank for none."
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
        save.setMinimumHeight(30)
        save.clicked.connect(self._save)
        save_form.addRow("", save)

        return [view_box, compare_box, edit_box, save_box]

    def build_docks(self) -> None:
        # No confusion matrix or summary here: they are what you look at
        # once, at the end, and they open as their own windows on save.
        self.agreement = TextPanel(
            "agreement",
            placeholder="No manual scoring was loaded, so there is nothing to "
                        "compare against. That is a normal way to run nyx.",
        )
        self._windows: list = []

    # -- panels ------------------------------------------------------------

    def _rebuild(self) -> None:
        self._built_for = None
        self.safe_refresh()

    def _build_viewers(self) -> None:
        if not self.session.has(Stage.STEPS):
            return
        key = (id(self.session.clusters()), self.scalogram.isChecked())
        if self._built_for == key:
            return

        from nyx.gui.review import NyxEpochSource
        from nyx.gui.viewers import NyxEpochEncoder

        self.docks.clear()
        self._built_for = key

        result = self.session.result()
        eeg, emg, offset = trace_sources(result.recording)
        params = self.session.params

        if emg is not None:
            self.docks.add(TraceViewer(source=emg, name="EMG"))
        self.docks.add(TraceViewer(source=eeg, name="EEG"))
        self.docks.add(self._timefreq(eeg, "EEG spectrum", "EEG", params))

        components = component_source(self.session.pca(), n=4, t_offset=offset)
        if components is not None:
            self.docks.add(_coloured(
                TraceViewer(source=components, name="components")
            ))

        self.epoch_source = NyxEpochSource(
            result.hypnogram, name="nyx", t_offset=offset,
            on_save=self._apply_edit,
        )
        self.encoder = NyxEpochEncoder(
            source=self.epoch_source, name="hypnogram",
            rules=self.session.postprocess_rules(),
        )
        self.docks.add(self.encoder)
        self.docks.add(self.agreement)   # bottom, by TextPanel default

        if self.show_reference.isChecked():
            self._add_reference(offset)

    def _timefreq(self, source, name: str, channel: str, params: dict):
        viewer = make_timefreq_viewer(
            source, name, params, channel, self.scalogram.isChecked()
        )
        try:
            viewer.params["colormap"] = self.palette.currentText()
        except Exception:  # noqa: BLE001
            pass
        return viewer

    def _add_reference(self, offset: float) -> None:
        """The manual scoring, read-only, tabbed with yours.

        A viewer rather than an encoder: it is the thing being compared
        against, and editing it would make the comparison meaningless.
        """
        from nyx.metrics import normalise_labels

        reference = self.session.result().reference
        if reference is None or "reference" in self.docks.viewers:
            return
        self.docks.add(
            EpochViewer(
                source=epoch_source(
                    normalise_labels(reference), "reference", offset
                ),
                name="reference",
            ),
            tabify_with="hypnogram",
        )

    def _toggle_reference(self, on: bool) -> None:
        if not self.session.has(Stage.STEPS):
            return
        if not on:
            self.docks.remove("reference")
            self.compare_note.setText("")
            return

        if self.session.result().reference is None:
            self.compare_note.setText("No manual scoring was loaded.")
            self.show_reference.setChecked(False)
            return

        _eeg, _emg, offset = trace_sources(self.session.result().recording)
        self._add_reference(offset)
        self.compare_note.setText(
            "Tabbed with yours -- click the tab to flip between them."
        )

    def _set_palette(self, name: str) -> None:
        for entry in self.docks.viewers.values():
            widget = entry["widget"]
            if hasattr(widget, "params") and hasattr(widget, "change_color_scale"):
                try:
                    widget.params["colormap"] = name
                except Exception:  # noqa: BLE001
                    continue

    # -- editing -----------------------------------------------------------

    def _apply_edit(self, hypnogram) -> None:
        self.session.set_hypnogram(hypnogram, source="manual")
        self.status.emit("Edit kept. It is what will be saved.")
        self._update_readouts()

    def _keep_edits(self) -> None:
        source = getattr(self, "epoch_source", None)
        if source is None:
            self.status.emit("Nothing to keep yet.")
            return
        source.save()

    # -- saving ------------------------------------------------------------

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Where to save", self.output.text() or ""
        )
        if path:
            self.output.setText(path)

    def _save(self) -> None:
        output = self.output.text().strip()
        if not output:
            self.status.emit("Choose somewhere to save first.")
            return

        granularities = [
            int(p) for p in self.granularities.text().replace(",", " ").split()
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

        self._show_final_figures()

    def _show_final_figures(self) -> None:
        """The look-at-it-once figures, as windows of their own.

        Not docked: they are the end of the run, wanted large and often
        side by side with something else, and a dock would only take width
        from the scoring you are still looking at.
        """
        result = self.session.result()
        name = self.session.recording.name or "recording"

        self._windows = [
            _figure_window(nyx.plot_summary(result), f"{name} -- summary", self)
        ]
        if result.agreement is not None:
            self._windows.append(
                _figure_window(
                    _confusion_figure(result), f"{name} -- confusion", self
                )
            )

    # -- readouts ----------------------------------------------------------

    def _update_readouts(self) -> None:
        result = self.session.result()

        self.edited_note.setText(
            "Edited by hand. That is what will be saved, and run.json records it."
            if self.session.was_edited() else "Not edited."
        )
        self.edited_note.setStyleSheet(
            "color: #e37400;" if self.session.was_edited() else "color: palette(mid);"
        )

        self.agreement.set_text(
            result.agreement.summary() if result.agreement is not None else ""
        )

    def refresh(self) -> None:
        if not self.session.has(Stage.STEPS):
            return

        if not self.output.text().strip():
            name = self.session.recording.name or "recording"
            self.output.setText(os.path.join("results", name))

        self._build_viewers()
        self._update_readouts()


def _confusion_figure(result):
    """``plot_confusion`` lays out its own colorbar, so give it its own figure."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    nyx.report.plot_confusion(result, ax)
    return figure


#: One colour per component, so PC1 and PC3 are told apart at a glance.
PC_COLOURS = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#a1887f"]


def _coloured(viewer):
    """Give each component trace its own colour."""
    try:
        for i in range(viewer.source.nb_channel):
            viewer.by_channel_params[f"ch{i}", "color"] = PC_COLOURS[
                i % len(PC_COLOURS)
            ]
    except Exception:  # noqa: BLE001 - parameter names vary between releases
        pass
    return viewer


def _figure_window(figure, title: str, parent=None):
    """Show a figure in a window of its own, with a save-image toolbar."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QVBoxLayout as _V
    from PySide6.QtWidgets import QWidget as _W

    from nyx.gui.canvas import FigureView

    window = _W(parent, Qt.Window)
    window.setWindowTitle(title)
    window.resize(1100, 800)
    layout = _V(window)
    layout.setContentsMargins(0, 0, 0, 0)
    view = FigureView()
    view.set_figure(figure)
    layout.addWidget(view)
    window.show()
    return window
