"""The window, its tabs and its badges."""

from __future__ import annotations

import pytest

import nyx
from nyx.gui.session import Stage
from nyx.gui.widgets import State

pytestmark = pytest.mark.gui


@pytest.fixture
def windows(qtbot):
    """Build main windows and close them deterministically.

    Each tab's dock area holds ephyviewer views with worker threads that only
    stop on closeEvent, and a dock area is a child widget, so it never gets
    one unless the window is actually closed.
    """
    built = []

    def make(session):
        from nyx.gui.mainwindow import MainWindow

        w = MainWindow(session)
        built.append(w)
        qtbot.addWidget(w)
        return w

    yield make

    for w in built:
        w.close()


@pytest.fixture
def window(windows, session):
    return windows(session)


@pytest.fixture
def scored_window(windows, scored):
    return windows(scored)


def badges(window):
    window.refresh_rail()
    return [row.badge.state() for row in window._rows]


# ---------------------------------------------------------------------------
# The tab contract
# ---------------------------------------------------------------------------


def test_every_tab_is_built(window):
    assert [tab.title for tab in window.tabs] == [
        "Recording", "Signal check", "EMG threshold", "Sleep stages", "Result"
    ]


@pytest.mark.parametrize("index", range(5))
def test_a_tab_refreshes_on_a_finished_session(scored_window, index):
    """The GUI's analogue of test_panels_accept_ax_positionally."""
    tab = scored_window.tabs[index]
    scored_window.rail.setCurrentRow(index)

    tab.safe_refresh()  # must not raise


@pytest.mark.parametrize("index", range(5))
def test_a_tab_refreshes_on_an_empty_session(windows, index):
    """Every tab is reachable before its inputs exist, so this is normal."""
    from nyx.gui.session import ScoringSession

    w = windows(ScoringSession(nyx.demo_params()))

    w.tabs[index].safe_refresh()  # must not raise


def test_each_tab_declares_a_stage(window):
    stages = [tab.stage for tab in window.tabs]

    assert stages == [
        Stage.LOAD, Stage.PREPROCESS, Stage.WAKE_SLEEP, Stage.STEPS, Stage.RESULT
    ]


# ---------------------------------------------------------------------------
# Badges: the thing free navigation depends on
# ---------------------------------------------------------------------------


def test_badges_start_blocked_below_the_recording(window):
    assert badges(window)[0] == State.CURRENT
    assert badges(window)[-1] == State.BLOCKED


def test_badges_light_up_in_order(window):
    window.session.compute_through(Stage.EMG)
    assert badges(window) == [
        State.CURRENT, State.CURRENT, State.READY, State.BLOCKED, State.BLOCKED
    ]

    window.session.compute_through(Stage.WAKE_SLEEP)
    assert badges(window)[2] == State.CURRENT


def test_everything_is_current_once_it_is_scored(scored_window):
    assert badges(scored_window) == [State.CURRENT] * 5


def test_moving_the_threshold_marks_downstream_stale_not_blocked(scored_window):
    scored_window.session.set_emg_threshold(0.4)

    # "You changed something" has to read differently from "not run yet", or
    # free navigation gives no clue what is out of date.
    assert badges(scored_window) == [
        State.CURRENT, State.CURRENT, State.STALE, State.STALE, State.STALE
    ]


def test_reclustering_leaves_the_earlier_tabs_alone(scored_window):
    scored_window.session.set_clustering({"method": "kmeans"})

    assert badges(scored_window)[:3] == [State.CURRENT] * 3
    assert badges(scored_window)[3:] == [State.STALE, State.STALE]


def test_a_new_recording_is_a_fresh_start_not_a_stale_one(scored_window):
    recording, truth = nyx.demo_recording(seed=1)
    scored_window.session.set_recording(recording, reference=truth)

    assert State.STALE not in badges(scored_window)


# ---------------------------------------------------------------------------
# Running work
# ---------------------------------------------------------------------------


def test_running_a_stage_computes_it(window, qtbot):
    with qtbot.waitSignal(window.jobs.done, timeout=120_000):
        window._run_stage(Stage.WAKE_SLEEP)

    assert window.session.has(Stage.WAKE_SLEEP)


def test_a_stale_result_is_discarded_rather_than_applied(scored_window, qtbot):
    logged = []
    scored_window.log.append_line = logged.append

    # Arrives describing a session that no longer exists.
    scored_window._on_finished(None, scored_window.session.generation() - 1)

    assert any("overtaken" in line for line in logged)


def test_a_failure_is_logged_with_its_traceback(scored_window, monkeypatch):
    logged = []
    scored_window.log.append_line = logged.append
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.exec", lambda self: None
    )

    scored_window._on_failed("ValueError: no", "Traceback...\n  ValueError: no", 0)

    assert any("Traceback" in line for line in logged)


def test_warnings_reach_the_current_tab(scored_window):
    scored_window.rail.setCurrentRow(3)

    scored_window._on_warned(["two clusters could not be named"], 0)

    # isHidden rather than isVisible: nothing is "visible" while the window
    # itself is not shown, which offscreen it never is.
    assert not scored_window.tabs[3].warnings.isHidden()


# ---------------------------------------------------------------------------
# The panels: interactive, and rearrangeable
# ---------------------------------------------------------------------------


EXPECTED_PANELS = {
    "Signal check": {"EEG", "EEG spectrum"},
    "EMG threshold": {"EMG distribution", "EMG power", "wake / sleep"},
    "Sleep stages": {"clusters and spectra", "components", "hypnogram"},
    "Result": {"EEG", "EEG spectrum", "hypnogram"},
}


@pytest.mark.parametrize("title, expected", EXPECTED_PANELS.items())
def test_each_tab_docks_the_panels_it_should(scored_window, title, expected):
    tab = next(t for t in scored_window.tabs if t.title == title)
    scored_window.rail.setCurrentRow(scored_window.tabs.index(tab))
    tab.safe_refresh()

    assert expected <= set(tab.docks.viewers)


def test_the_panels_can_be_dragged_side_by_side(scored_window):
    # Nesting is what lets two panels sit beside each other rather than only
    # above and below. Without it the layout is not rearrangeable in the way
    # ephyviewer's own window is.
    for tab in scored_window.tabs:
        assert tab.docks.isDockNestingEnabled()


def test_every_tab_has_a_navigation_toolbar(scored_window):
    for tab in scored_window.tabs:
        assert tab.docks.navigation_toolbar is not None


def test_the_views_share_one_clock(scored_window):
    """Scrolling one panel has to scroll the rest, or they cannot be compared."""
    tab = next(t for t in scored_window.tabs if t.title == "Result")
    scored_window.rail.setCurrentRow(scored_window.tabs.index(tab))
    tab.safe_refresh()

    tab.docks.on_time_changed(600.0)

    assert all(
        entry["widget"].t == pytest.approx(600.0)
        for entry in tab.docks.viewers.values()
    )


def test_the_run_button_is_visible_where_the_controls_are(scored_window):
    # It used to sit in a footer under the panels, where it was off-screen.
    for tab in scored_window.tabs:
        if tab.needs_worker:
            assert not tab.run_button.isHidden()


def test_dragging_the_threshold_does_not_rebuild_the_layout(scored_window):
    """The bug this replaced: every release tore the panels down and back up,
    which lost the scroll position and looked like the window reloading."""
    tab = next(t for t in scored_window.tabs if t.title == "EMG threshold")
    scored_window.rail.setCurrentRow(scored_window.tabs.index(tab))
    tab.safe_refresh()
    before = list(tab.docks.viewers)

    tab._lines.set(1, 0.31)

    assert list(tab.docks.viewers) == before
    assert scored_window.session.wake_sleep().threshold == pytest.approx(0.31)


def test_the_threshold_shows_on_both_the_distribution_and_the_trace(scored_window):
    tab = next(t for t in scored_window.tabs if t.title == "EMG threshold")
    scored_window.rail.setCurrentRow(scored_window.tabs.index(tab))
    tab.safe_refresh()

    # One value, two lines -- that is the whole reason for showing both.
    drawn = tab._lines._lines[1]
    assert {orientation for _line, orientation in drawn} == {"v", "h"}

    tab._lines.set(1, 0.27)
    assert all(
        float(line.value()) == pytest.approx(0.27) for line, _o in drawn
    )


# ---------------------------------------------------------------------------
# Canvases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel", nyx.report.AX_PANELS, ids=lambda p: p.__name__)
def test_every_ax_panel_draws_on_a_canvas(qtbot, scored, panel):
    from nyx.gui.canvas import PanelCanvas

    canvas = PanelCanvas()
    qtbot.addWidget(canvas)

    ax = canvas.draw_panel(panel, scored.result())

    assert ax is canvas.figure.axes[0]


def test_a_panel_that_raises_is_drawn_as_a_message_not_propagated(qtbot):
    from nyx.gui.canvas import PanelCanvas

    canvas = PanelCanvas()
    qtbot.addWidget(canvas)

    def broken(piece, ax):
        raise RuntimeError("nope")

    canvas.draw_panel(broken, None)

    assert any("nope" in t.get_text() for t in canvas.figure.axes[0].texts)


def test_a_figure_view_releases_the_figure_it_replaces(qtbot, scored):
    import matplotlib.pyplot as plt

    from nyx.gui.canvas import FigureView

    view = FigureView()
    qtbot.addWidget(view)

    first = nyx.plot_emg_check(scored.emg(), threshold=0.5)
    view.set_figure(first)
    view.set_figure(nyx.plot_emg_check(scored.emg(), threshold=0.6))

    # The composites build their figures through pyplot, so an unreleased one
    # stays in its registry for ever.
    assert first not in [plt.figure(n) for n in plt.get_fignums()]
