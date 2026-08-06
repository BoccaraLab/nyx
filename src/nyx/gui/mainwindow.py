"""One window, tabs down the side.

Not a wizard. Every tab is reachable as soon as its inputs exist, so you can go
back and retune a threshold after seeing the clusters without clicking Next
four times to get forward again.

The cost of free navigation is that staleness has to be *drawn*. A wizard
encodes "not computed yet" in position; here each tab carries a badge --
blocked, ready, current, stale -- read straight off the session's invalidation
graph. Changing the EMG threshold leaves the EMG tab current and marks
everything after it stale, and you can see that without clicking anything.

The rail is built from a list even though the list is short. That is what makes
the multi-step human pipeline a data change rather than a rewrite: one entry
per clustering step instead of one entry, and the badges keep working.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QWidget,
)

from nyx.gui.jobs import JobRunner
from nyx.gui.session import ScoringSession, Stage
from nyx.gui.widgets import LogPane, StageBadge, State

__all__ = ["MainWindow", "tab_classes"]


def tab_classes() -> list:
    """The tabs, in order. Imported here so a broken tab fails loudly."""
    from nyx.gui.tabs.emg import EmgTab
    from nyx.gui.tabs.load import LoadTab
    from nyx.gui.tabs.result import ResultTab
    from nyx.gui.tabs.signal import SignalTab
    from nyx.gui.tabs.sleep import SleepTab

    return [LoadTab, SignalTab, EmgTab, SleepTab, ResultTab]


class _RailRow(QWidget):
    """One entry in the side rail: a badge, a title and a subtitle."""

    def __init__(self, index: int, title: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.badge = StageBadge()
        layout.addWidget(self.badge, 0, Qt.AlignTop)

        self.label = QLabel(f"<b>{index}. {title}</b>")
        self.label.setTextFormat(Qt.RichText)
        layout.addWidget(self.label, 1)


class MainWindow(QMainWindow):
    """The scoring window."""

    session_changed = Signal(int)

    def __init__(self, session: ScoringSession | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("nyx")
        self.resize(1400, 900)

        self.session = session or ScoringSession()
        self.jobs = JobRunner(self)
        self._running_stage: Stage | None = None
        # Stages that have been computed at some point for this recording. The
        # difference between "not run yet" and "was run, then you changed
        # something upstream" is the whole reason the badges exist, and the
        # session cannot tell them apart -- it only knows what it currently
        # holds.
        self._computed_once: set[Stage] = set()

        self._build()
        self._connect()
        self.refresh_all()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = QListWidget()
        self.rail.setFixedWidth(220)
        self.rail.setFrameShape(QListWidget.NoFrame)
        self.rail.setSpacing(1)
        layout.addWidget(self.rail)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.setCentralWidget(central)

        self.tabs = []
        self._rows = []
        for number, factory in enumerate(tab_classes(), start=1):
            tab = factory(self.session)
            tab.run_requested.connect(self._run_stage)
            tab.status.connect(self.status)
            self.tabs.append(tab)
            self.stack.addWidget(tab)

            row = _RailRow(number, tab.title)
            item = QListWidgetItem(self.rail)
            item.setSizeHint(row.sizeHint())
            self.rail.addItem(item)
            self.rail.setItemWidget(item, row)
            self._rows.append(row)

        self.rail.setCurrentRow(0)

        # -- status bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate: there is nothing to count
        self.progress.setFixedWidth(160)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        # -- log. Before the first status(), which writes to it.
        self.log = LogPane()
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.hide()
        self._log_dock = dock

        self._build_menu()
        self.status("Open a recording to begin, or try the demo.")

    def _build_menu(self) -> None:
        view = self.menuBar().addMenu("&View")
        action = self._log_dock.toggleViewAction()
        action.setText("Show &log")
        view.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About nyx", self._about)

    def _about(self) -> None:
        import nyx

        QMessageBox.about(
            self,
            "nyx",
            f"<b>nyx {nyx.__version__}</b><br><br>"
            "Unsupervised sleep scoring from EEG and EMG.<br>"
            "Every number here comes from the same functions the example "
            "notebooks call.",
        )

    def _connect(self) -> None:
        self.rail.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.rail.currentRowChanged.connect(lambda _: self.refresh_all())

        # The session calls its listeners on whichever thread changed it, which
        # for a worker job is not this one. Re-emitting as a Qt signal makes
        # the hop: a signal from a non-GUI thread to a slot on a GUI-thread
        # object is queued automatically.
        self.session.add_listener(lambda stage: self.session_changed.emit(int(stage)))
        self.session_changed.connect(self._on_session_changed, Qt.QueuedConnection)

        self.jobs.started.connect(self.status)
        self.jobs.finished.connect(self._on_finished)
        self.jobs.failed.connect(self._on_failed)
        self.jobs.warned.connect(self._on_warned)
        self.jobs.busy_changed.connect(self._on_busy)

    # -- running -----------------------------------------------------------

    def _run_stage(self, stage) -> None:
        stage = Stage(int(stage))
        if self.jobs.busy():
            self.status("Already working; wait for it to finish.")
            return

        current = self.current_tab()
        if current is not None:
            current.apply()

        self._running_stage = stage
        self.refresh_rail()
        self.jobs.submit(
            self.session.compute_through,
            stage,
            description=f"Computing {stage.name.lower().replace('_', ' ')}...",
            generation=self.session.generation(),
        )

    def _on_finished(self, _value, generation: int) -> None:
        if generation != self.session.generation():
            # Something changed while this ran, so it describes a session that
            # no longer exists. Dropping it is what "Abandon" means.
            self.log.append_line("(discarded a result that was overtaken)")
            return
        self.status("Done.")

    def _on_failed(self, message: str, tb: str, _generation: int) -> None:
        self.log.append_line(tb)
        self.status(message)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("That did not work")
        box.setText(message)
        box.setDetailedText(tb)
        box.exec()

    def _on_warned(self, messages: list, _generation: int) -> None:
        for message in messages:
            self.log.append_line(f"warning: {message}")
        tab = self.current_tab()
        if tab is not None:
            tab.show_warnings(messages)

    def _on_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        if not busy:
            self._running_stage = None
        for tab in self.tabs:
            tab.run_button.setEnabled(not busy)
        self.refresh_rail()

    # -- refreshing --------------------------------------------------------

    def _on_session_changed(self, _stage: int) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        self.refresh_rail()
        tab = self.current_tab()
        if tab is not None:
            tab.safe_refresh()

    def refresh_rail(self) -> None:
        # A different recording is a fresh start, not a stale one. The session
        # reports set_recording, set_params and set_window as the same event,
        # so the recording itself is what has to be watched.
        recording = getattr(self.session, "_recording", None)
        if recording is not getattr(self, "_last_recording", None):
            self._last_recording = recording
            self._computed_once.clear()

        for tab in self.tabs:
            if self.session.has(tab.stage):
                self._computed_once.add(tab.stage)
        for tab, row in zip(self.tabs, self._rows, strict=True):
            row.badge.set_state(self._state_of(tab))

    def _state_of(self, tab) -> str:
        if self._running_stage is not None and tab.stage is self._running_stage:
            return State.RUNNING
        if self.session.has(tab.stage):
            return State.CURRENT
        if tab.stage in self._computed_once:
            # It had a value and lost it, which means something upstream
            # changed. Worth saying, because what is on screen is now a
            # picture of a scoring that no longer exists.
            return State.STALE
        if self._can_run(tab.stage):
            return State.READY
        return State.BLOCKED

    def _can_run(self, stage: Stage) -> bool:
        """Whether everything this stage needs is already computed."""
        if stage is Stage.LOAD:
            return True
        return self.session.has(Stage(int(stage) - 1))

    def current_tab(self):
        index = self.rail.currentRow()
        if 0 <= index < len(self.tabs):
            return self.tabs[index]
        return None

    def go_to(self, stage: Stage) -> None:
        """Show the tab that owns ``stage``."""
        for index, tab in enumerate(self.tabs):
            if tab.stage is stage:
                self.rail.setCurrentRow(index)
                return

    def status(self, text: str) -> None:
        self.statusBar().showMessage(str(text), 8000)
        self.log.append_line(text)

    # -- teardown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt's spelling
        if self.jobs.busy():
            answer = QMessageBox.question(
                self,
                "Still working",
                "Something is still running. Close anyway?\n\n"
                "It cannot be stopped, but its result will be discarded.",
            )
            if answer is not QMessageBox.Yes:
                event.ignore()
                return
        QApplication.processEvents()
        super().closeEvent(event)
