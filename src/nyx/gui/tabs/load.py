"""Tab 1 -- the recording, the parameters and the optional reference.

The notebooks' first STOP: *check the channels*. Getting EEG and EMG the wrong
way round produces a scoring that looks plausible and is wrong, and the only
defence is looking at what was actually picked -- which is why
``Recording.describe()`` is on screen rather than in a log.

The channel lists come from the file, so a channel is chosen from what it
contains rather than typed as an index and hoped for. For EDF, spikeinterface
folders and anything Neo reads (``format="neo"``: Open Ephys, Intan, SpikeGLX,
Spike2...), the tab goes further before anything is loaded: every channel of
every stream is drawn, scrollable, beside a table of what the file says about
each one -- name, stream, rate, unit, gain -- so the EEG and the EMG can be
told apart by looking rather than by guessing from a raw file's numbering.
"""

from __future__ import annotations

import os

from ephyviewer import TraceViewer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
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
from nyx.gui.panels import ChannelTablePanel, TextPanel
from nyx.gui.session import Stage
from nyx.gui.sources import autoscale, colour_channels, stream_source
from nyx.gui.tabs.base import Tab
from nyx.gui.widgets import Section

__all__ = ["LoadTab"]

_NO_EMG = "(no EMG)"
_AUTO = "auto"

#: Formats whose every channel can be opened for a look before choosing.
_BROWSABLE = ("neo", "edf", "spikeinterface")


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
        recording_box = Section(
            "Recording",
            "The file, and which two channels in it are the EEG and the "
            "EMG. The channel lists are read out of the file itself, so "
            "you pick from what is there rather than typing an index and "
            "hoping.\n\n"
            "Check the summary panel before moving on. EEG and EMG the "
            "wrong way round gives a scoring that looks entirely "
            "plausible and is wrong.\n\n"
            "Raw files from an acquisition system -- Open Ephys, Intan, "
            "SpikeGLX, Spike2, Blackrock, Plexon, Neuralynx, TDT and more "
            "-- are read through Neo: format 'neo', or 'auto' and let Neo "
            "recognise it. Every channel is then drawn on the right with a "
            "table of what the file says about it; select a row there and "
            "use it as the EEG or the EMG.\n\n"
            "Several formats (spikeinterface, Open Ephys, SpikeGLX...) are "
            "a folder rather than a file, which is what the second browse "
            "button is for.",
        )
        self._form = form = QFormLayout(recording_box)

        self.path = QLineEdit()
        self.path.setPlaceholderText("Choose a file, or a folder")
        browse_file = QPushButton("File...")
        browse_file.clicked.connect(self._browse_file)
        browse_folder = QPushButton("Folder...")
        browse_folder.clicked.connect(self._browse_folder)
        browse_folder.setToolTip(
            "For recordings stored as a folder: spikeinterface, Open Ephys, "
            "SpikeGLX, Neuralynx, TDT..."
        )
        form.addRow("path", _row(self.path, browse_file, browse_folder))

        self.format = QComboBox()
        self.format.addItems([_AUTO, *recording_formats()])
        form.addRow("format", self.format)

        self.neo_format = QComboBox()
        self.neo_format.addItems([_AUTO, *nyx.io.neo_formats()])
        self.neo_format.setToolTip(
            "Which Neo reader to use. 'auto' lets Neo recognise the file "
            "from its extension or the folder's contents."
        )
        form.addRow("Neo format", self.neo_format)
        form.setRowVisible(self.neo_format, False)

        self.eeg_channel = QComboBox()
        self.eeg_channel.setEditable(True)
        self.emg_channel = QComboBox()
        self.emg_channel.setEditable(True)
        form.addRow("EEG channel", self.eeg_channel)
        form.addRow("EMG channel", self.emg_channel)

        self.channel_note = QLabel()
        self.channel_note.setWordWrap(True)
        self.channel_note.setStyleSheet("font-size: 11px;")
        form.addRow("", self.channel_note)

        # -- params
        params_box = Section(
            "Parameters",
            "Spectral settings, normalisation, and the structure of the "
            "scoring steps. The presets ship with nyx -- mouse, rat, "
            "human and the variants -- or point at your own JSON.\n\n"
            "Which species you are scoring is a matter of how many "
            "clustering steps run and what the clusters are called, not "
            "of different code.",
        )
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
        self.params_note.setStyleSheet("font-size: 11px;")
        params_form.addRow("", self.params_note)

        # -- reference
        reference_box = Section(
            "Manual scoring (optional)",
            "A scoring to compare against. Entirely optional -- nyx needs "
            "no training data, and scoring without a reference is the "
            "normal case.\n\n"
            "Epoch formats need the epoch length, and usually a label map "
            "saying which number means which stage: 0=WAKE, 1=NREM1 and "
            "so on.",
        )
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
        self.neo_format.currentTextChanged.connect(self._offer_channels)
        self.eeg_channel.currentIndexChanged.connect(self._mark_chosen)
        self.emg_channel.currentIndexChanged.connect(self._mark_chosen)

        #: What the file says about each channel, when it can be browsed.
        self._channels: list = []
        self._previewed: tuple | None = None

        # Opening a file to list its channels runs here rather than on the
        # window's runner: it is not a pipeline stage, and must not wait
        # behind one or mark the window busy.
        from nyx.gui.jobs import JobRunner

        self._reader = JobRunner(self)
        self._reader.finished.connect(self._on_read)
        self._reader.failed.connect(self._on_read_failed)
        #: Every file asked for, by job generation; the last is the one wanted.
        self._requests: list[tuple] = []
        self._reading_key: tuple | None = None

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
        self.channel_table = ChannelTablePanel("channels in the file")
        self.channel_table.use_requested.connect(self._use_channel)
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
            # auto tells a folder spikeinterface saved from one Neo reads.
            if self.format.currentText() not in ("spikeinterface", "neo"):
                with self.quiet(self.format):
                    self.format.setCurrentText(_AUTO)
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
            self._form.setRowVisible(self.neo_format, self.format.currentText() == "neo")
            return

        # Whatever was being read is no longer what is wanted.
        self._reading_key = None
        format = self.format.currentText()
        try:
            resolved = format if format != _AUTO else _infer(path)
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            self._form.setRowVisible(self.neo_format, False)
            self._clear_preview()
            self.channel_note.setText(
                f"{exc}\n\nIf this is a raw acquisition file, set the format "
                f"to 'neo' and pick the Neo format."
            )
            return
        self._form.setRowVisible(self.neo_format, resolved == "neo")

        if resolved in _BROWSABLE:
            self._offer_every_channel(path, resolved)
            return
        self._offer_channel_names(path, resolved)

    def _offer_channel_names(self, path: str, resolved: str) -> None:
        """The plain list of channel names, for a file that cannot be browsed."""
        self._clear_preview()
        try:
            if not can_list_channels(resolved):
                self.channel_note.setText(
                    f"Cannot list channels for {resolved!r} -- type a name or "
                    f"an index (0 is the first channel)."
                )
                return
            names = nyx.io.list_channels(path, format=resolved, **self._neo_kwargs())
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

    def _neo_kwargs(self) -> dict:
        chosen = self.neo_format.currentText()
        return {} if chosen == _AUTO else {"neo_format": chosen}

    def _offer_every_channel(self, path: str, format: str) -> None:
        """Open every stream, draw every channel, list them with their metadata.

        Opening is done on a worker: a raw file on a network share can take a
        while, and a window that stops responding meanwhile looks crashed.
        The latest request wins -- choosing another file while one is being
        read drops the first one's result when it arrives.

        When the file cannot be opened this way, the plain channel list is
        tried instead -- an EDF that only MNE will read, say.
        """
        kwargs = self._neo_kwargs() if format == "neo" else {}
        key = (path, format, tuple(sorted(kwargs.items())))
        if key == self._previewed:
            return
        self._reading_key = key
        generation = len(self._requests)
        self._requests.append(key)

        if _edf_backed(path, format, kwargs):
            # EDF goes through pyedflib, whose C library keeps its open files
            # in global tables with no locking: opening one on a worker while
            # the last preview's reader is freed here is an access violation.
            # An EDF header is small, so it is read here instead.
            try:
                value = _open_every_stream(path, format, kwargs)
            except Exception as exc:  # noqa: BLE001 - shown, not raised
                self._on_read_failed(f"{type(exc).__name__}: {exc}", "", generation)
                return
            self._on_read(value, generation)
            return

        name = os.path.basename(path.rstrip("/\\"))
        self.channel_note.setText(f"Reading the channels of {name}...")
        self._reader.submit(
            _open_every_stream, path, format, kwargs,
            description="Reading the channels", generation=generation,
        )

    def _on_read(self, value, generation: int) -> None:
        key = self._requests[generation]
        if key != self._reading_key:
            return   # another file was chosen while this one was read
        self._reading_key = None
        path, format, kwargs = key[0], key[1], dict(key[2])
        streams, channels = value
        if not channels:
            self._offer_channel_names(path, format)
            return
        self._show_every_channel(path, format, kwargs, key, streams, channels)

    def _on_read_failed(self, message: str, _tb: str, generation: int) -> None:
        key = self._requests[generation]
        if key != self._reading_key:
            return
        self._reading_key = None
        self._offer_channel_names(key[0], key[1])
        self.channel_note.setText(
            f"Could not open every channel ({message}). "
            + self.channel_note.text()
        )

    def reading(self) -> bool:
        """Whether a file's channels are still being read."""
        return self._reading_key is not None

    def _show_every_channel(self, path, format, kwargs, key, streams, channels):
        self._channels = channels
        self._previewed = key
        self._build_preview(streams, channels)

        # One channel is an EEG with no EMG, whatever it is called.
        emg_default = (
            _guess(channels, "emg", avoid=None) if len(channels) > 1 else None
        )
        eeg_default = _guess(channels, "eeg", avoid=emg_default)
        for combo, default, extra in (
            (self.eeg_channel, eeg_default, []),
            (self.emg_channel, emg_default, [_NO_EMG]),
        ):
            with self.quiet(combo):
                combo.clear()
                # The row in self._channels, not the ChannelInfo itself: Qt
                # cannot find a Python object again by value.
                for row, channel in enumerate(channels):
                    combo.addItem(_label(channel, channels), row)
                for text in extra:
                    combo.addItem(text, None)
                combo.setCurrentIndex(
                    channels.index(default) if default is not None else len(channels)
                )
        self._mark_chosen()

        n_streams = len({c.stream_id for c in channels})
        guessed = ""
        if format == "neo" and not kwargs:
            guessed = f" Read as {nyx.io.guess_neo_format(path)}."
        self.channel_note.setText(
            f"{len(channels)} channel(s)"
            + (f" in {n_streams} streams" if n_streams > 1 else "")
            + f".{guessed} Every one is drawn on the right: scroll through "
            "them, then pick the EEG and the EMG here or from the table."
        )
        # Said every time the file is opened, not once: it is about this
        # file, and whoever opens it should know what was not checked.
        note = nyx.io.intan_unchecked_note(path) if format == "neo" else None
        self.show_warnings([note] if note else [])

    def _build_preview(self, streams, channels) -> None:
        self.docks.clear()
        first = None
        for stream_id, stream_name, rec in streams:
            names = [c.name for c in channels if c.stream_id == stream_id]
            title = "all channels" if len(streams) == 1 else f"{stream_name}"
            viewer = TraceViewer(source=stream_source(rec, names), name=title)
            # Stacked one above the other, each labelled, and in its own
            # colour -- left alone they are all drawn on the same baseline,
            # in the same green, and look like one trace.
            viewer.params["scale_mode"] = "same_for_all"
            viewer.params["display_labels"] = True
            colour_channels(viewer)
            autoscale(viewer)
            if first is None:
                self.docks.add(viewer)
                first = title
            else:
                self.docks.add(viewer, tabify_with=first)
        self.channel_table.set_channels(channels)
        self.docks.add(self.channel_table)
        self.docks.add(self.summary, tabify_with=self.channel_table.name)
        # Tabified last, so it would be the one in front.
        self.docks.viewers[self.channel_table.name]["dock"].raise_()

    def _clear_preview(self) -> None:
        if self._previewed is None and not self._channels:
            return
        self._channels = []
        self._previewed = None
        self.docks.clear()
        self.docks.add(self.summary)

    def _use_channel(self, role: str, channel) -> None:
        combo = self.eeg_channel if role == "EEG" else self.emg_channel
        if channel in self._channels:
            combo.setCurrentIndex(self._channels.index(channel))

    def _chosen(self, combo):
        """The ``ChannelInfo`` a combo is on; ``None`` if typed or '(no EMG)'.

        The combos stay editable, so what is shown may have been typed over an
        item; only an item still showing its own label counts as picked.
        """
        index = combo.currentIndex()
        if index < 0 or combo.itemText(index) != combo.currentText():
            return None
        row = combo.itemData(index)
        if row is None or not 0 <= int(row) < len(self._channels):
            return None
        return self._channels[int(row)]

    def _mark_chosen(self, *_args) -> None:
        if not self._channels:
            return
        self.channel_table.mark("EEG", self._chosen(self.eeg_channel))
        self.channel_table.mark("EMG", self._chosen(self.emg_channel))

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
        format = self.format.currentText()
        eeg_info = self._chosen(self.eeg_channel)
        emg_info = self._chosen(self.emg_channel)

        if eeg_info is not None:
            # Picked from the file's own channels: select by what identifies
            # the channel in its stream, and say which stream.
            if emg_info == eeg_info:
                raise ValueError(
                    f"The EEG and the EMG are both {eeg_info.name!r}. Pick "
                    f"two different channels -- or '(no EMG)'."
                )
            if emg_info is not None:
                emg_spec = _selector(emg_info, self._channels)
            elif emg in ("", _NO_EMG):
                emg_spec = None
            else:
                emg_spec = _channel(emg)

            format = _infer(path) if format == _AUTO else format
            kwargs = self._neo_kwargs() if format == "neo" else {}
            if len({c.stream_id for c in self._channels}) > 1:
                kwargs["stream_id"] = eeg_info.stream_id
                if emg_info is not None and emg_info.stream_id != eeg_info.stream_id:
                    kwargs["emg_stream_id"] = emg_info.stream_id
            recording = nyx.read_recording(
                path,
                format=format,
                eeg_channel=_selector(eeg_info, self._channels),
                emg_channel=emg_spec,
                **kwargs,
            )
        else:
            recording = nyx.read_recording(
                path,
                format=format,
                eeg_channel=_channel(self.eeg_channel.currentText()),
                emg_channel=None if emg in ("", _NO_EMG) else _channel(emg),
                **(self._neo_kwargs() if format == "neo" else {}),
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

    def shutdown(self) -> None:
        # A read still running would report into a tab being torn down.
        self._reading_key = None
        self._reader.wait()
        super().shutdown()

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


def _edf_backed(path: str, format: str, kwargs: dict) -> bool:
    """Whether opening this means pyedflib, which must stay on one thread."""
    if format == "edf":
        return True
    if format != "neo":
        return False
    return (kwargs.get("neo_format") or nyx.io.guess_neo_format(path)) == "EDF"


def _open_every_stream(path: str, format: str, kwargs: dict):
    """``(streams, channels)`` -- on a worker thread, so no Qt here."""
    streams = nyx.io.recording_streams(path, format=format, **kwargs)
    return streams, nyx.io.describe_channels(streams)


def _label(channel, channels) -> str:
    """How a channel reads in a combo: its name, and whatever tells it apart."""
    parts = [channel.name]
    if channel.channel_id != channel.name:
        parts.append(f"id {channel.channel_id}")
    parts.append(f"{channel.fs:g} Hz")
    if len({c.stream_id for c in channels}) > 1:
        parts.append(channel.stream_name)
    return f"{parts[0]}  ({', '.join(parts[1:])})"


def _selector(channel, channels) -> str:
    """What to pass as ``eeg_channel`` / ``emg_channel`` for this channel.

    Its name when that is unambiguous within the stream, since a name is what
    a saved run.json should say -- ``"EMG"``, not ``"2"``. Otherwise the id:
    raw files repeat names, and a name that is also another channel's id
    would select that channel instead (ids are matched first).
    """
    same_stream = [c for c in channels if c.stream_id == channel.stream_id]
    names = [c.name for c in same_stream]
    other_ids = {c.channel_id for c in same_stream if c is not channel}
    if names.count(channel.name) == 1 and channel.name not in other_ids:
        return channel.name
    return channel.channel_id


def _guess(channels, role: str, avoid):
    """A starting choice: the first channel named like ``role``, else by position.

    Only a default -- the table and the traces are there to check it.
    """
    for channel in channels:
        if role in channel.name.lower() and channel is not avoid:
            return channel
    if role == "emg":
        return channels[1] if len(channels) > 1 else channels[0]
    for channel in channels:
        if channel is not avoid:
            return channel
    return channels[0]


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
