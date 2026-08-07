"""Tab 1 -- the recording, the parameters and the optional reference.

The notebooks' first STOP: *check the channels*. Getting EEG and EMG the wrong
way round produces a scoring that looks plausible and is wrong, and the only
defence is looking at what was actually picked -- which is why
``Recording.describe()`` is on screen rather than in a log.

The channel lists come from :func:`nyx.io.list_channels`, so a channel is
chosen from what the file contains rather than typed as an index and hoped for.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

import nyx
from nyx.gui.filters import (
    annotation_filters,
    annotation_formats,
    can_list_channels,
    recording_filters,
    recording_formats,
)
from nyx.gui.panels import TextPanel
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab

__all__ = ["LoadTab"]

_NO_EMG = "(no EMG)"


class LoadTab(Tab):
    title = "Recording"
    subtitle = (
        "Pick the file, the two channels and a parameter preset. Check the "
        "summary panel before moving on: EEG and EMG the wrong way "
        "round gives a scoring that looks plausible and is wrong."
    )
    stage = Stage.LOAD
    run_label = "Load"

    def build_controls(self) -> list:
        # -- recording
        recording_box = QGroupBox("Recording")
        form = QFormLayout(recording_box)

        self.path = QLineEdit()
        self.path.setPlaceholderText("Choose a file, or a folder for spikeinterface")
        browse_file = QPushButton("File...")
        browse_file.clicked.connect(self._browse_file)
        browse_folder = QPushButton("Folder...")
        browse_folder.clicked.connect(self._browse_folder)
        browse_folder.setToolTip(
            "spikeinterface recordings are a folder, not a file."
        )
        form.addRow("path", _row(self.path, browse_file, browse_folder))

        self.format = QComboBox()
        self.format.addItems(["auto", *recording_formats()])
        form.addRow("format", self.format)

        self.eeg_channel = QComboBox()
        self.eeg_channel.setEditable(True)
        self.emg_channel = QComboBox()
        self.emg_channel.setEditable(True)
        form.addRow("EEG channel", self.eeg_channel)
        form.addRow("EMG channel", self.emg_channel)

        self.channel_note = QLabel()
        self.channel_note.setWordWrap(True)
        self.channel_note.setStyleSheet("color: palette(mid);")
        form.addRow("", self.channel_note)

        # -- params
        params_box = QGroupBox("Parameters")
        params_form = QFormLayout(params_box)
        self.preset = QComboBox()
        self.preset.addItems(nyx.available_params())
        self.preset.setCurrentText("mouse")
        self.preset.currentTextChanged.connect(self._load_params)
        params_browse = QPushButton("File...")
        params_browse.clicked.connect(self._browse_params)
        params_form.addRow("preset", _row(self.preset, params_browse))

        self.params_note = QLabel()
        self.params_note.setWordWrap(True)
        self.params_note.setStyleSheet("color: palette(mid);")
        params_form.addRow("", self.params_note)

        # -- reference
        reference_box = QGroupBox("Manual scoring (optional)")
        reference_form = QFormLayout(reference_box)
        self.reference_path = QLineEdit()
        self.reference_path.setPlaceholderText("Leave empty to score without one")
        reference_browse = QPushButton("File...")
        reference_browse.clicked.connect(self._browse_reference)
        reference_form.addRow("path", _row(self.reference_path, reference_browse))

        self.reference_format = QComboBox()
        self.reference_format.addItems(["auto", *annotation_formats()])
        reference_form.addRow("format", self.reference_format)

        self.epoch_length = QLineEdit()
        self.epoch_length.setPlaceholderText("e.g. 4 or 30, for epoch formats")
        reference_form.addRow("epoch length", self.epoch_length)

        self.label_map = QLineEdit()
        self.label_map.setPlaceholderText("0=WAKE, 1=NREM1, 2=NREM2")
        reference_form.addRow("label map", self.label_map)

        # -- demo
        demo = QPushButton("Try the demo recording")
        demo.setToolTip(
            "A synthetic recording that needs no data. Built to contain "
            "exactly the structure nyx looks for, so it scores near-perfectly "
            "-- good for learning the mechanics, useless as a measure of "
            "how well nyx works."
        )
        demo.clicked.connect(self._load_demo)

        self.path.editingFinished.connect(self._offer_channels)
        self.format.currentTextChanged.connect(self._offer_channels)

        # Only seed the session when it has nothing. Loading a preset here
        # unconditionally would invalidate a session that arrived already
        # configured -- a resumed run.json, or one the tests built.
        if self.session.params:
            self._describe_params()
        else:
            self._load_params(self.preset.currentText())

        return [recording_box, params_box, reference_box, demo]

    def build_docks(self) -> None:
        self.summary = TextPanel(
            "what was loaded", placeholder="Nothing loaded yet."
        )
        self.docks.add(self.summary)

    # -- browsing ----------------------------------------------------------

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a recording", self.path.text() or "", recording_filters()
        )
        if path:
            self.path.setText(path)
            self._offer_channels()

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Open a recording folder", self.path.text() or ""
        )
        if path:
            self.path.setText(path)
            self.format.setCurrentText("spikeinterface")
            self._offer_channels()

    def _browse_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a parameter file", "", "Parameters (*.json);;All files (*)"
        )
        if path:
            self._load_params(path)

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a manual scoring", "", annotation_filters()
        )
        if path:
            self.reference_path.setText(path)

    # -- channels ----------------------------------------------------------

    def _offer_channels(self) -> None:
        """Fill the channel combos from the file, without loading it."""
        path = self.path.text().strip()
        if not path or not os.path.exists(path):
            return

        format = self.format.currentText()
        try:
            resolved = format if format != "auto" else _infer(path)
            if not can_list_channels(resolved):
                self.channel_note.setText(
                    f"Cannot list channels for {resolved!r} -- type a name or "
                    f"an index (0 is the first channel)."
                )
                return
            names = nyx.io.list_channels(path, format=format)
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            self.channel_note.setText(f"Could not read the channels: {exc}")
            return

        for combo, default, extra in (
            (self.eeg_channel, 0, []),
            (self.emg_channel, 1, [_NO_EMG]),
        ):
            with self.quiet(combo):
                current = combo.currentText()
                combo.clear()
                combo.addItems([*names, *extra])
                if current in names:
                    combo.setCurrentText(current)
                elif len(names) > default:
                    combo.setCurrentIndex(default)

        self.channel_note.setText(
            f"{len(names)} channel(s) in the file."
            + ("" if len(names) > 1 else "  Only one -- there is no EMG to pick.")
        )

    # -- loading -----------------------------------------------------------

    def _load_params(self, name: str) -> None:
        try:
            params = nyx.load_params(name)
        except Exception as exc:  # noqa: BLE001
            self.params_note.setText(f"Could not read {name!r}: {exc}")
            return

        self.session.set_params(params)
        self._describe_params()

    def _describe_params(self) -> None:
        steps = self.session.steps
        self.params_note.setText(
            f"{len(steps)} clustering step(s): "
            + ", ".join(f"{s.name} ({s.method} within {s.within})" for s in steps)
        )

    def _load_demo(self) -> None:
        recording, truth = nyx.demo_recording()
        self.session.set_params(nyx.demo_params())
        self.session.set_recording(recording, reference=truth)
        self.path.setText("(demo)")
        self.status.emit(
            "Loaded the synthetic demo recording and its true hypnogram."
        )
        self.safe_refresh()

    def apply(self) -> None:
        path = self.path.text().strip()
        if not path or path == "(demo)":
            return

        emg = self.emg_channel.currentText()
        recording = nyx.read_recording(
            path,
            format=self.format.currentText(),
            eeg_channel=_channel(self.eeg_channel.currentText()),
            emg_channel=None if emg in ("", _NO_EMG) else _channel(emg),
        )

        reference = None
        reference_path = self.reference_path.text().strip()
        if reference_path:
            kwargs = {}
            if self.epoch_length.text().strip():
                kwargs["epoch_length"] = float(self.epoch_length.text())
            mapping = _parse_label_map(self.label_map.text())
            if mapping:
                kwargs["label_map"] = mapping
            reference = nyx.read_annotations(
                reference_path, format=self.reference_format.currentText(), **kwargs
            )

        self.session.set_recording(recording, reference=reference)

    def refresh(self) -> None:
        try:
            recording = self.session.recording
        except Exception:  # noqa: BLE001 - nothing loaded yet is normal here
            self.summary.set_text("")
            return

        lines = [recording.describe(), ""]
        if self.session.reference is not None:
            labels = self.session.reference["label"]
            lines.append(f"reference: {len(labels)} segments")
            counts: dict[str, int] = {}
            for label in labels:
                counts[str(label)] = counts.get(str(label), 0) + 1
            lines.append(
                "  " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
            )
        else:
            lines.append("reference: none -- scoring will not be scored against anything")
        self.summary.set_text("\n".join(lines))


def _row(*widgets) -> QWidget:
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if index == 0 else 0)
    return holder


def _infer(path: str) -> str:
    from nyx.io.recordings import _infer_format

    return _infer_format(path)


def _channel(text: str):
    """A channel is an index or a name; digits mean an index."""
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _parse_label_map(text: str) -> dict:
    """``"0=WAKE, 1=NREM"`` -> ``{0: "WAKE", 1: "NREM"}``."""
    mapping: dict = {}
    for part in str(text).split(","):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        mapping[int(key) if key.lstrip("-").isdigit() else key] = value
    return mapping
