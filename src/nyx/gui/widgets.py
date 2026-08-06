"""Reusable inputs and readouts.

Two of these are built by *reading nyx* rather than by hand:
:class:`ClusteringForm` walks ``DEFAULT_CLUSTERING`` and :class:`PostprocessForm`
walks ``RULES``. Adding a clustering knob or a postprocessing rule to the
library therefore gives it a control here for free, which is the same bargain
the reader registries make with the file dialog. Hand-written forms drift; these
cannot.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nyx.gui.session import Stage

__all__ = [
    "State",
    "StageBadge",
    "WarningBanner",
    "LogPane",
    "DataFrameTable",
    "ClusteringForm",
    "PostprocessForm",
    "StageTable",
    "monospace",
]

#: Which clustering settings each method actually reads. Keys not listed apply
#: to every method. Taken from ``nyx.clustering.run_clustering_step``'s
#: docstring, which is where they are documented.
METHOD_SETTINGS = {
    "kmeans": {"n_clusters"},
    "gmm": {"n_clusters"},
    "hdbscan": {"hdbscan_min_cluster_size", "hdbscan_min_samples"},
    "elliptic": {"elliptic_contamination", "elliptic_support_fraction"},
}
_METHOD_ONLY = set().union(*METHOD_SETTINGS.values())


class State:
    """What a tab's stage looks like right now.

    Free navigation means "not computed yet" cannot be implied by position the
    way a wizard implies it, so it has to be drawn.
    """

    BLOCKED = "blocked"    # an upstream stage has no input yet
    READY = "ready"        # inputs exist, not run
    CURRENT = "current"    # computed and up to date
    STALE = "stale"        # was computed, then something upstream changed
    RUNNING = "running"

    SYMBOLS = {
        BLOCKED: "·",   # ·
        READY: "●",     # ●
        CURRENT: "✓",   # ✓
        STALE: "⟳",     # ⟳
        RUNNING: "…",   # …
    }
    COLOURS = {
        BLOCKED: "#9aa0a6",
        READY: "#1a73e8",
        CURRENT: "#188038",
        STALE: "#e37400",
        RUNNING: "#1a73e8",
    }
    TOOLTIPS = {
        BLOCKED: "Not available yet -- an earlier step has to run first.",
        READY: "Ready to run.",
        CURRENT: "Up to date.",
        STALE: "Out of date: something earlier changed. Re-run to update.",
        RUNNING: "Running...",
    }


class StageBadge(QLabel):
    """A one-character state marker for a tab in the side rail."""

    def __init__(self, state: str = State.BLOCKED, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(18)
        self.set_state(state)

    def set_state(self, state: str) -> None:
        self._state = state
        self.setText(State.SYMBOLS.get(state, "?"))
        self.setStyleSheet(
            f"color: {State.COLOURS.get(state, '#9aa0a6')}; font-weight: bold;"
        )
        self.setToolTip(State.TOOLTIPS.get(state, ""))

    def state(self) -> str:
        return self._state


class WarningBanner(QFrame):
    """A dismissible strip for warnings raised during a computation.

    nyx warns where a user must notice -- clusters it could not name, scoring
    without an EMG, clustering settings it did not recognise. On the command
    line those land on stderr. Here they would be lost.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #fff4e5; border: 1px solid #e37400;"
            " border-radius: 4px; }"
            "QLabel { color: #7a4100; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._label, 1)

        from PySide6.QtWidgets import QToolButton

        close = QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        layout.addWidget(close, 0, Qt.AlignTop)

        self.hide()

    def show_messages(self, messages: Sequence[str]) -> None:
        messages = [m for m in messages if m]
        if not messages:
            self.hide()
            return
        if len(messages) == 1:
            self._label.setText(messages[0])
        else:
            self._label.setText(
                "\n\n".join(f"• {m}" for m in messages)
            )
        self.show()


class LogPane(QPlainTextEdit):
    """Append-only text, for warnings, tracebacks and what was written."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(monospace())

    def append_line(self, text: str) -> None:
        for line in str(text).rstrip().splitlines() or [""]:
            self.appendPlainText(line)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


def monospace():
    from PySide6.QtGui import QFontDatabase

    return QFontDatabase.systemFont(QFontDatabase.FixedFont)


class DataFrameTable(QTableWidget):
    """A read-only view of a pandas DataFrame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)

    def set_frame(self, frame, float_format: str = "{:.3f}") -> None:
        columns = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
        self.clear()
        self.setColumnCount(len(columns))
        self.setRowCount(len(frame))
        self.setHorizontalHeaderLabels(columns)

        for row, (index, values) in enumerate(frame.iterrows()):
            cells = [index, *values.tolist()]
            for column, value in enumerate(cells):
                if isinstance(value, float):
                    text = float_format.format(value)
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.setItem(row, column, item)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class ClusteringForm(QGroupBox):
    """Every clustering setting, one widget each, built from the defaults.

    ``DEFAULT_CLUSTERING`` is walked rather than transcribed: a new key added
    to it in ``nyx.pipeline`` gets a control here without anyone editing this
    file. Widget type follows the default's type.
    """

    changed = Signal()

    def __init__(self, parent=None, title: str = "Clustering"):
        super().__init__(title, parent)
        from nyx.pipeline import DEFAULT_CLUSTERING

        self._widgets: dict[str, QWidget] = {}
        self._defaults = dict(DEFAULT_CLUSTERING)

        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)

        for key, default in self._defaults.items():
            widget = self._build(key, default)
            self._widgets[key] = widget
            layout.addRow(key.replace("_", " "), widget)

        self._method().currentTextChanged.connect(self._update_enabled)
        self._update_enabled()

    # -- construction ------------------------------------------------------

    def _build(self, key: str, default) -> QWidget:
        if key == "method":
            widget = QComboBox()
            widget.addItems(sorted(METHOD_SETTINGS))
            widget.setCurrentText(str(default))
            widget.currentTextChanged.connect(self.changed)
            return widget

        if isinstance(default, bool):
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.toggled.connect(self.changed)
            return widget

        if isinstance(default, (list, tuple)):
            widget = QLineEdit(", ".join(str(v) for v in default))
            widget.setToolTip("Comma-separated component indices, e.g. 0, 1, 2")
            widget.editingFinished.connect(self.changed)
            return widget

        if isinstance(default, int) and not isinstance(default, bool):
            widget = QSpinBox()
            widget.setRange(0, 1_000_000)
            widget.setValue(int(default))
            widget.valueChanged.connect(self.changed)
            return widget

        if isinstance(default, float):
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setRange(0.0, 1_000_000.0)
            widget.setSingleStep(0.05)
            widget.setValue(float(default))
            widget.valueChanged.connect(self.changed)
            return widget

        # None, typically: an optional numeric with a "none" checkbox beside it.
        widget = QLineEdit("" if default is None else str(default))
        widget.setPlaceholderText("none")
        widget.editingFinished.connect(self.changed)
        return widget

    def _method(self) -> QComboBox:
        return self._widgets["method"]

    def _update_enabled(self) -> None:
        method = self._method().currentText()
        wanted = METHOD_SETTINGS.get(method, set())
        for key, widget in self._widgets.items():
            if key in _METHOD_ONLY:
                widget.setEnabled(key in wanted)

    # -- values ------------------------------------------------------------

    def values(self) -> dict:
        """Current settings, in the shape ``cluster_sleep`` expects."""
        out: dict = {}
        method = self._method().currentText()
        for key, widget in self._widgets.items():
            if key in _METHOD_ONLY and key not in METHOD_SETTINGS.get(method, set()):
                continue  # not read by this method; leave it at the default
            out[key] = self._value_of(key, widget)
        return out

    def _value_of(self, key: str, widget: QWidget):
        default = self._defaults[key]
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        text = widget.text().strip()
        if isinstance(default, (list, tuple)):
            return [int(p) for p in text.replace(",", " ").split()]
        if not text or text.lower() == "none":
            return None
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return text

    def set_values(self, settings: dict) -> None:
        """Seed from ``resolve_clustering_params(params)``."""
        for key, value in settings.items():
            widget = self._widgets.get(key)
            if widget is None:
                continue
            blocked = widget.blockSignals(True)
            try:
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(value, (list, tuple)):
                    widget.setText(", ".join(str(v) for v in value))
                else:
                    widget.setText("" if value is None else str(value))
            finally:
                widget.blockSignals(blocked)
        self._update_enabled()


class PostprocessForm(QGroupBox):
    """A checkbox per postprocessing rule, built from ``RULES``.

    Rules are off unless the params turn them on, and each one trades agreement
    for plausibility -- so this shows what is enabled rather than encouraging
    more of it.
    """

    changed = Signal()

    def __init__(self, parent=None, title: str = "Postprocessing rules"):
        super().__init__(title, parent)
        from nyx.postprocess import RULES

        self._rules = dict(RULES)
        self._boxes: dict[str, QCheckBox] = {}
        self._kwargs: dict[str, QLineEdit] = {}

        layout = QFormLayout(self)
        for name in self._rules:
            box = QCheckBox()
            box.toggled.connect(self.changed)
            arguments = QLineEdit()
            arguments.setPlaceholderText("seconds=4")
            arguments.setToolTip(
                "Keyword arguments for this rule, e.g. seconds=4 or "
                "min_wake_duration=20. Leave blank for its defaults."
            )
            arguments.editingFinished.connect(self.changed)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(box)
            row_layout.addWidget(arguments, 1)

            self._boxes[name] = box
            self._kwargs[name] = arguments
            layout.addRow(name.replace("_", " "), row)

    def rules(self) -> list:
        """The enabled rules, in ``apply_rules``' format."""
        out: list = []
        for name, box in self._boxes.items():
            if not box.isChecked():
                continue
            kwargs = _parse_kwargs(self._kwargs[name].text())
            out.append({name: kwargs} if kwargs else name)
        return out

    def set_rules(self, rules: Sequence) -> None:
        from nyx.postprocess import _as_call

        enabled = {}
        for rule in rules or []:
            name, kwargs = _as_call(rule)
            enabled[name] = kwargs

        for name, box in self._boxes.items():
            blocked = box.blockSignals(True)
            box.setChecked(name in enabled)
            box.blockSignals(blocked)

            field = self._kwargs[name]
            blocked = field.blockSignals(True)
            field.setText(
                ", ".join(f"{k}={v}" for k, v in (enabled.get(name) or {}).items())
            )
            field.blockSignals(blocked)


def _parse_kwargs(text: str) -> dict:
    """``"seconds=4, order=REM"`` -> ``{"seconds": 4.0, "order": "REM"}``."""
    out: dict = {}
    for part in str(text).split(","):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        value = value.strip()
        try:
            out[key.strip()] = float(value) if "." in value else int(value)
        except ValueError:
            out[key.strip()] = value
    return out


class StageTable(QWidget):
    """The clusters, in order, each with an editable stage name.

    The order column matters: two clusters can sit almost on top of each other,
    and knowing *which is which* is the whole decision this table exists for.
    Editing a stage names that cluster by id, which wins over naming by
    position -- and costs nothing, because renaming never re-clusters.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self._combos: dict[int, QComboBox] = {}
        self._stages: list[str] = []

    def set_table(self, frame, stages: Sequence[str]) -> None:
        """``frame`` is :func:`nyx.cluster_ordering`'s DataFrame."""
        self._stages = list(stages)
        self._combos = {}

        columns = [str(frame.index.name or "order")] + [
            str(c) for c in frame.columns if c != "stage"
        ] + ["stage"]
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(frame))
        self.table.setHorizontalHeaderLabels(columns)

        for row, (order, values) in enumerate(frame.iterrows()):
            cells = [order] + [
                values[c] for c in frame.columns if c != "stage"
            ]
            for column, value in enumerate(cells):
                text = f"{value:.3f}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.table.setItem(row, column, item)

            cluster = int(values["cluster"])
            combo = QComboBox()
            combo.addItems(self._stages)
            current = str(values.get("stage", "")) if "stage" in frame.columns else ""
            if current and current in self._stages:
                combo.setCurrentText(current)
            combo.currentTextChanged.connect(self.changed)
            self._combos[cluster] = combo
            self.table.setCellWidget(row, len(columns) - 1, combo)

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

    def overrides(self) -> dict[int, str]:
        """Cluster id -> stage, as chosen in the table."""
        return {
            cluster: combo.currentText()
            for cluster, combo in self._combos.items()
            if combo.currentText()
        }


def stage_for(session, stage: Stage, *, running: bool = False) -> str:
    """The :class:`State` a tab should show for ``stage``."""
    if running:
        return State.RUNNING
    if session.has(stage):
        return State.CURRENT
    previous = Stage(max(int(stage) - 1, 0))
    if stage is Stage.LOAD or session.has(previous):
        return State.READY
    return State.BLOCKED
