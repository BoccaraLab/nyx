"""The window, its tabs and its badges."""

from __future__ import annotations

import numpy as np
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
    # The clustering only: the traces and the hypnogram belong to the tabs
    # either side of it.
    "Sleep stages": {"clusters and spectra", "other dimensions"},
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
# What the panels are actually set to
# ---------------------------------------------------------------------------


def tab_named(window, title):
    tab = next(t for t in window.tabs if t.title == title)
    window.rail.setCurrentRow(window.tabs.index(tab))
    tab.safe_refresh()
    return tab


def test_the_spectrogram_uses_the_epoch_length_from_the_params(scored_window):
    """ephyviewer's own default is 0.01 s -- one sample, and a blank panel."""
    tab = tab_named(scored_window, "Signal check")

    viewer = tab.docks.panel("EEG spectrum")
    binsize = viewer.params["scalogram", "binsize"]

    assert binsize == pytest.approx(
        scored_window.session.params["EEG"]["binsize"]
    )


def test_the_fourier_view_shows_the_band_the_scoring_uses(qtbot, scored_window):
    """It ran 0 Hz to Nyquist whatever the params said, and ignored the
    max frequency box; only the wavelet view honoured it."""
    tab = tab_named(scored_window, "Signal check")
    eeg = scored_window.session.params["EEG"]

    viewer = tab.docks.panel("EEG spectrum")
    assert viewer.params["scalogram", "f_start"] == pytest.approx(eeg["min_freq"])
    assert viewer.params["scalogram", "f_stop"] == pytest.approx(eeg["max_freq"])

    viewer.refresh()
    qtbot.waitUntil(lambda: viewer.last_Sxx.get(0) is not None, timeout=15_000)
    low, high = viewer.plots[0].getViewBox().viewRange()[1]
    df = 1.0 / viewer.params["scalogram", "binsize"]
    assert low == pytest.approx(eeg["min_freq"], abs=df)
    assert high == pytest.approx(eeg["max_freq"], abs=df)


def test_narrowing_the_band_zooms_the_fourier_view(qtbot, scored_window):
    tab = tab_named(scored_window, "Signal check")
    tab.channel.setCurrentText("EEG")
    tab.fmin.setValue(5.0)
    tab.fmax.setValue(12.0)

    viewer = tab.docks.panel("EEG spectrum")
    assert viewer.params["scalogram", "f_start"] == pytest.approx(5.0)
    assert viewer.params["scalogram", "f_stop"] == pytest.approx(12.0)

    viewer.last_Sxx[0] = None
    viewer.refresh()
    qtbot.waitUntil(lambda: viewer.last_Sxx.get(0) is not None, timeout=15_000)
    df = 1.0 / viewer.params["scalogram", "binsize"]
    # Only the rows inside the band are computed and drawn.
    assert viewer.last_Sxx[0].shape[0] == pytest.approx(7.0 / df + 1, abs=1)


def test_a_spectrogram_for_the_old_band_is_not_drawn(scored_window):
    """One requested before the band changed can arrive after it.

    It failed in CI, where the runner is slow enough for that to happen:
    the stale image was stored and stretched over the new band's axis.
    """
    import numpy as np

    tab = tab_named(scored_window, "Signal check")
    tab.channel.setCurrentText("EEG")
    tab.fmin.setValue(5.0)
    tab.fmax.setValue(12.0)
    viewer = tab.docks.panel("EEG spectrum")
    viewer.last_Sxx[0] = None

    stale = np.zeros((80, 10))   # the rows of a wider band
    viewer.on_data_ready(0, viewer.t, 0.0, 10.0, 0.0, 10.0, stale)

    assert viewer.last_Sxx[0] is None


@pytest.mark.parametrize("wavelet", [False, True])
def test_a_result_that_lands_after_close_is_ignored(scored_window, wavelet):
    """A transform still running at close posts its result anyway.

    Drawn into a viewer being torn down, that segfaulted the CI run -- a
    slow runner is what lets the transform outlast the close.
    """
    import numpy as np

    tab = tab_named(scored_window, "Signal check")
    tab.scalogram.setChecked(wavelet)
    viewer = tab.docks.panel("EEG spectrum")
    stored = viewer.last_wt_maps if wavelet else viewer.last_Sxx
    stored[0] = None

    viewer.close()
    viewer.on_data_ready(0, viewer.t, 0.0, 10.0, 0.0, 10.0, np.zeros((5, 10)))

    assert stored[0] is None


def test_the_fourier_band_is_in_the_double_click_settings(scored_window):
    tab = tab_named(scored_window, "Signal check")
    viewer = tab.docks.panel("EEG spectrum")

    names = [p.name() for p in viewer.params.param("scalogram").children()]
    assert {"f_start", "f_stop"} <= set(names)
    nyquist = viewer.source.sample_rate / 2
    assert viewer.params.param("scalogram", "f_stop").opts["limits"][1] == nyquist


def test_the_wavelet_view_can_be_switched_on(scored_window):
    from nyx.gui.viewers import NyxSpectrogramViewer, NyxTimeFreqViewer

    tab = tab_named(scored_window, "Signal check")
    assert isinstance(tab.docks.panel("EEG spectrum"), NyxSpectrogramViewer)

    tab.scalogram.setChecked(True)

    assert isinstance(tab.docks.panel("EEG spectrum"), NyxTimeFreqViewer)


def test_the_wavelet_frequency_axis_covers_the_band(qtbot, scored_window):
    """It read 0 to 1 whatever the transform was computed over.

    The image is placed in data coordinates, so the plot's own range has to be
    set alongside it -- which the override of ``on_data_ready`` had dropped.
    """
    tab = tab_named(scored_window, "Signal check")
    tab.scalogram.setChecked(True)

    viewer = tab.docks.panel("EEG spectrum")
    viewer.refresh()
    qtbot.waitUntil(lambda: bool(viewer.last_wt_maps), timeout=15_000)

    low, high = viewer.plots[0].getViewBox().viewRange()[1]
    assert low == pytest.approx(viewer.params["timefreq", "f_start"], abs=0.5)
    assert high == pytest.approx(viewer.params["timefreq", "f_stop"], abs=0.5)


def test_the_text_panels_are_along_the_bottom(scored_window):
    from PySide6.QtCore import Qt

    for title, panel in (
        ("Recording", "what was loaded"),
        ("Signal check", "measurements"),
    ):
        tab = tab_named(scored_window, title)
        dock = tab.docks.viewers[panel]["dock"]
        assert tab.docks.dockWidgetArea(dock) == Qt.BottomDockWidgetArea


def test_a_button_that_has_to_be_pressed_first_is_at_the_top(scored_window):
    placement = {t.title: t.run_at_top for t in scored_window.tabs if t.needs_worker}

    # Compute the EMG features and Recompute the PCA gate everything else on
    # their tab; the others apply what you have set and move on.
    assert placement["EMG threshold"] is True
    assert placement["Sleep stages"] is True
    assert placement["Signal check"] is False


def test_the_emg_trace_is_the_top_panel(scored_window):
    tab = tab_named(scored_window, "EMG threshold")

    # Top to bottom: the signal, the power it is summarised into, the bouts.
    assert list(tab.docks.viewers)[0] == "EMG"


def test_the_histogram_can_be_resized(scored_window):
    from PySide6.QtWidgets import QSizePolicy

    tab = tab_named(scored_window, "EMG threshold")
    histogram = tab.docks.panel("EMG distribution")

    assert histogram.sizePolicy().verticalPolicy() == QSizePolicy.Expanding


def recluster(window, tab, qtbot, n):
    """Press Recluster the way a person does, and wait for the worker."""
    tab.clustering.set_values({"method": "kmeans", "n_clusters": n})
    with qtbot.waitSignal(window.jobs.done, timeout=120_000):
        tab._recluster()
    qtbot.waitUntil(lambda: not window.jobs.busy(), timeout=120_000)
    qtbot.wait(50)


@pytest.mark.parametrize("n", [3, 4, 2])
def test_changing_the_cluster_count_keeps_the_panels_drawn(
    scored_window, qtbot, n
):
    """The one that went blank and could not be recovered from.

    Rebuilding the dock area closed the panels, which released their canvases
    *and hid them*, and re-adding the same objects did not bring them back.
    """
    tab = tab_named(scored_window, "Sleep stages")

    recluster(scored_window, tab, qtbot, n)

    panel = tab.clusters_panel
    assert not panel.isHidden()
    assert panel.view.figure is not None and panel.view.figure.axes
    assert panel.view.canvas is not None
    assert tab.table.table.rowCount() == n


def test_every_cluster_ends_up_named_even_when_there_are_more_than_names(
    scored_window, qtbot
):
    """Extras become UNCLASSIFIED rather than staying provisional.

    Left as C2, C3 they are unscored, and nyx warns about it on every single
    recompute. The table has a row for each, so those names are pushed back --
    the ones you have not decided on say so.
    """
    tab = tab_named(scored_window, "Sleep stages")

    recluster(scored_window, tab, qtbot, 4)

    mapping = scored_window.session.cluster_to_stage()
    assert len(mapping) == 4
    assert not any(
        str(v).startswith("C") and str(v)[1:].isdigit() for v in mapping.values()
    )
    assert "UNCLASSIFIED" in mapping.values()


def test_the_cluster_count_can_be_taken_back_down(scored_window, qtbot):
    tab = tab_named(scored_window, "Sleep stages")

    recluster(scored_window, tab, qtbot, 4)
    recluster(scored_window, tab, qtbot, 2)

    assert tab.table.table.rowCount() == 2
    assert set(scored_window.session.cluster_to_stage().values()) == {"REM", "NREM"}


def test_the_table_is_the_only_thing_that_names_clusters(scored_window):
    tab = tab_named(scored_window, "Sleep stages")

    # A typed list of names could get out of step with the clustering; one row
    # per cluster cannot.
    assert not hasattr(tab, "stage_order")
    assert tab.table.table.rowCount() == len(
        scored_window.session.clusters().unique_labels
    )


def test_the_view_settings_can_be_written_into_the_params(scored_window):
    tab = tab_named(scored_window, "Signal check")

    tab.binsize.setValue(8.0)
    tab._apply_settings()

    assert scored_window.session.params["EEG"]["binsize"] == pytest.approx(8.0)


def test_the_wake_sleep_panel_is_editable_and_in_nyxs_colours(scored_window):
    from nyx.gui.viewers import NyxEpochEncoder
    from nyx.stages import COLORS

    tab = tab_named(scored_window, "EMG threshold")
    panel = tab.docks.panel("wake / sleep")

    assert isinstance(panel, NyxEpochEncoder)
    colours = dict(zip(panel.source.possible_labels, panel.source.color_labels,
                       strict=True))
    for label in ("WAKE", "SLEEP"):
        if label in colours:
            assert colours[label] == COLORS[label]


def test_the_emg_power_trace_is_not_left_to_auto_scale(scored_window):
    tab = tab_named(scored_window, "EMG threshold")
    power = tab.docks.panel("EMG power")

    # The power is min-max scaled to [0, 1] and its tails run to the edges,
    # so auto-scaling leaves the interesting part in a sliver.
    assert power.params["ylim_min"] == pytest.approx(-0.01)
    assert power.params["ylim_max"] == pytest.approx(1.01)


def test_the_emg_tab_has_a_confirm_button(scored_window):
    tab = tab_named(scored_window, "EMG threshold")

    assert "onfirm" in tab.confirm.text()


def test_the_sleep_tab_shows_only_the_clustering(scored_window):
    tab = tab_named(scored_window, "Sleep stages")

    # The traces and the hypnogram belong to the tabs either side; here they
    # only compete for width with the scatter being read.
    assert not {"EEG", "hypnogram", "components"} & set(tab.docks.viewers)


def test_the_matplotlib_panels_match_the_ephyviewer_ones(scored_window):
    tab = tab_named(scored_window, "Sleep stages")

    figure = tab.clusters_panel.view.figure
    assert figure is not None
    assert figure.patch.get_facecolor()[:3] != (1.0, 1.0, 1.0)


def test_epochs_can_be_assigned_by_hand(scored, scored_window):
    import numpy as np

    clusters = scored.clusters()
    mask = np.zeros(clusters.features_scaled.shape[0], dtype=bool)
    mask[:40] = True

    assigned = scored.assign_epochs(mask, "REM")
    scored.compute(Stage.STEPS)

    assert assigned == 40
    assert "REM" in set(scored.staging().hypnogram["label"])
    # A scoring a human touched must not claim to be automatic.
    assert "manual_epochs" in scored.to_run_config().decisions


def test_clearing_hand_assignments_puts_the_clustering_back(scored):
    import numpy as np

    before = list(scored.staging().hypnogram["label"])
    mask = np.zeros(scored.clusters().features_scaled.shape[0], dtype=bool)
    mask[:40] = True
    scored.assign_epochs(mask, "WAKE")
    scored.compute(Stage.STEPS)

    scored.clear_manual_epochs()
    scored.compute(Stage.STEPS)

    assert list(scored.staging().hypnogram["label"]) == before


def test_the_result_tab_hides_the_reference_until_asked(scored_window):
    tab = tab_named(scored_window, "Result")

    assert "reference" not in tab.docks.viewers

    tab.show_reference.setChecked(True)

    assert "reference" in tab.docks.viewers


def test_the_reference_is_drawn_the_same_way_as_the_scoring(scored_window):
    tab = tab_named(scored_window, "Result")
    tab.show_reference.setChecked(True)

    reference = tab.docks.panel("reference")
    mine = tab.docks.panel("hypnogram")

    # The same widget, so flipping between the tabs compares like with like...
    assert type(reference) is type(mine)
    # ...but read-only, since editing what you compare against would make the
    # comparison meaningless.
    assert reference.read_only is True


def test_the_agreement_table_comes_and_goes_with_the_reference(scored_window):
    tab = tab_named(scored_window, "Result")

    assert "agreement" not in tab.docks.viewers

    tab.show_reference.setChecked(True)
    assert "agreement" in tab.docks.viewers

    tab.show_reference.setChecked(False)
    assert "agreement" not in tab.docks.viewers
    assert "reference" not in tab.docks.viewers


def test_rebuilding_panels_opens_no_stray_windows(scored_window, qtbot):
    """setParent(None) on a visible widget makes it a window of its own.

    Redrawing a tab did that several times over, so windows flashed up and
    vanished on every change.
    """
    from PySide6.QtWidgets import QApplication

    tab = tab_named(scored_window, "Signal check")
    before = {id(w) for w in QApplication.topLevelWidgets()}

    tab.scalogram.setChecked(True)
    tab.scalogram.setChecked(False)
    qtbot.wait(20)

    appeared = [
        w for w in QApplication.topLevelWidgets()
        if id(w) not in before and w.isVisible()
    ]
    assert appeared == []


def test_the_warning_banner_can_be_dismissed(scored_window):
    from PySide6.QtWidgets import QPushButton

    tab = tab_named(scored_window, "Sleep stages")
    tab.show_warnings(["something worth noticing"])
    assert not tab.warnings.isHidden()

    dismiss = tab.warnings.findChildren(QPushButton)[0]
    assert dismiss.text() == "Dismiss"
    dismiss.click()

    assert tab.warnings.isHidden()


def test_every_section_explains_itself(scored_window):
    """The explanations moved behind a ``?``; none of them got lost."""
    from nyx.gui.widgets import Section

    for tab in scored_window.tabs:
        sections = tab.findChildren(Section)
        assert sections, f"{tab.title} has no sections"
        assert all(s.help_text for s in sections), f"{tab.title} has a bare section"


def test_the_polygon_can_be_closed_from_the_keyboard(scored_window, qtbot):
    """matplotlib only sends key events to a canvas that has focus, and a
    canvas in a dock never takes it by itself -- so enter did nothing."""
    from matplotlib.backend_bases import KeyEvent

    tab = tab_named(scored_window, "Sleep stages")
    tab.lasso.setChecked(True)

    assert tab.lasso.text() == "Finish"
    canvas = tab.clusters_panel.view.figure.canvas
    assert canvas.focusPolicy() != Qt_NoFocus()

    canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", canvas, "enter")
    )

    assert not tab.lasso.isChecked()
    assert tab.lasso.text() == "Draw"


def Qt_NoFocus():
    from PySide6.QtCore import Qt

    return Qt.NoFocus


def test_the_summary_figures_open_as_windows_when_you_save(scored_window, tmp_path):
    tab = tab_named(scored_window, "Result")

    assert not {"confusion", "summary"} & set(tab.docks.viewers)

    tab.output.setText(str(tmp_path))
    tab._save()

    # Windows of their own, not docks: they are the end of the run, wanted
    # large, and a dock would take width from the scoring still on screen.
    titles = [w.windowTitle() for w in tab._windows]
    assert any("summary" in t for t in titles)
    assert all(w.isWindow() for w in tab._windows)
    assert not {"confusion", "summary"} & set(tab.docks.viewers)


def test_the_components_are_shown_and_told_apart(scored_window):
    tab = tab_named(scored_window, "Result")
    components = tab.docks.panel("components")

    assert components is not None
    colours = [
        components.by_channel_params[f"ch{i}", "color"].name()
        for i in range(components.source.nb_channel)
    ]
    assert len(set(colours)) == len(colours)


# ---------------------------------------------------------------------------
# Defaults you should not have to set yourself
# ---------------------------------------------------------------------------


def test_raw_traces_are_scaled_to_their_own_amplitude(scored_window):
    """A recording in volts opens as a flat line on ephyviewer's fixed range."""
    tab = tab_named(scored_window, "Signal check")
    eeg = tab.docks.panel("EEG")

    low, high = eeg.params["ylim_min"], eeg.params["ylim_max"]
    assert low < 0 < high
    assert (high - low) != pytest.approx(2.0)   # not still the default


def test_the_power_and_the_components_are_pinned_instead(scored_window):
    emg = tab_named(scored_window, "EMG threshold")
    power = emg.docks.panel("EMG power")
    assert (power.params["ylim_min"], power.params["ylim_max"]) == (
        pytest.approx(-0.01), pytest.approx(1.01)
    )

    result = tab_named(scored_window, "Result")
    components = result.docks.panel("components")
    signal = scored_window.session.pca().signal[:, :4]
    finite = signal[np.isfinite(signal)]
    # Wide enough to hold the scores, and not much wider.
    assert components.params["ylim_min"] <= float(finite.min())
    assert components.params["ylim_max"] >= float(finite.max())


def test_the_hypnogram_opens_showing_the_hypnogram(scored_window):
    tab = tab_named(scored_window, "EMG threshold")
    encoder = tab.docks.panel("wake / sleep")

    # The controls take a third of the panel and are wanted only while editing.
    assert encoder.controls.isHidden()


def test_the_range_selector_starts_one_epoch_wide(scored_window):
    tab = tab_named(scored_window, "EMG threshold")
    encoder = tab.docks.panel("wake / sleep")

    # Upstream starts it at one second, which is not a length anything in a
    # hypnogram has.
    width = encoder.spin_limit2.value() - encoder.spin_limit1.value()
    assert width == pytest.approx(scored_window.session.min_duration())


@pytest.mark.parametrize("wavelet", [False, True])
def test_the_time_frequency_views_start_on_jet(scored_window, wavelet):
    tab = tab_named(scored_window, "Signal check")
    tab.scalogram.setChecked(wavelet)

    assert tab.docks.panel("EEG spectrum").params["colormap"] == "jet"


def icon_lightness(icon, size=32):
    """Mean lightness of an icon's opaque pixels, 0 (black) to 255 (white)."""
    from PySide6.QtGui import QImage

    pixmap = icon.pixmap(size, size)
    image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    buffer = buffer.reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )[:, : image.width()]
    opaque = buffer[..., 3] > 40
    return None if not opaque.any() else float(buffer[..., :3][opaque].mean())


@pytest.fixture
def dark(qtbot, monkeypatch):
    """Pretend the theme is dark, which is when the icons are a problem."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    before = app.palette()
    palette = QPalette(before)
    palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    app.setPalette(palette)
    yield
    app.setPalette(before)


def test_the_icons_are_lightened_on_a_dark_theme(dark, windows, scored):
    """ephyviewer's icons are dark line art drawn for a light theme.

    On a dark one the play and stop buttons, the encoder's save and undo, and
    the split and reorder buttons in the epoch table are black on near-black.
    """
    from PySide6.QtWidgets import QAbstractButton

    window = windows(scored)
    tab = tab_named(window, "EMG threshold")
    encoder = tab.docks.panel("wake / sleep")

    # play and stop -- plain buttons on a QWidget, not a toolbar
    for button in tab.docks.navigation_toolbar.findChildren(QAbstractButton):
        if not button.icon().isNull():
            assert icon_lightness(button.icon()) > 150

    # save, undo, redo -- actions on the encoder's toolbar
    for action in encoder.toolbar.actions():
        if not action.icon().isNull():
            assert icon_lightness(action.icon()) > 150

    # seek, split, duplicate, delete -- on QTableWidgetItems, which
    # findChildren cannot reach, so they are lightened at their source
    for icon in encoder.table_widget_icons.values():
        assert icon_lightness(icon) > 150


def test_the_panels_themselves_stay_dark(scored_window):
    """The icons are recoloured, not the bars they sit in.

    Lightening the bars works, and cuts a pale stripe across an otherwise dark
    window.
    """
    for tab in scored_window.tabs:
        assert tab.docks.styleSheet() == ""
        assert tab.docks.navigation_toolbar.styleSheet() == ""


def test_lightening_twice_does_not_put_them_back(dark, windows, scored):
    from PySide6.QtWidgets import QAbstractButton

    from nyx.gui.panels import brighten_icons

    window = windows(scored)
    tab = tab_named(window, "EMG threshold")
    button = next(
        b for b in tab.docks.navigation_toolbar.findChildren(QAbstractButton)
        if not b.icon().isNull()
    )
    before = icon_lightness(button.icon())

    brighten_icons(tab.docks)

    # Inverting an inverted icon gives the dark original back.
    assert icon_lightness(button.icon()) == pytest.approx(before)


def test_clearing_hand_assignments_takes_the_drawing_away(scored_window):
    tab = tab_named(scored_window, "Sleep stages")
    tab.lasso.setChecked(True)
    assert tab._selector is not None

    tab._clear_manual()

    # Leaving the shape on the scatter after its assignment is undone says
    # something that is no longer true.
    assert tab._selector is None
    assert not tab.lasso.isChecked()
    assert tab.lasso.text() == "Draw"


def test_the_window_says_whether_the_result_has_been_saved(scored_window, tmp_path):
    assert "not saved" in scored_window.saved_label.text()

    tab = tab_named(scored_window, "Result")
    tab.output.setText(str(tmp_path))
    tab._save()

    # A status message lasts eight seconds; whether the work is written out is
    # worth being able to check at any point.
    assert str(tmp_path) in scored_window.saved_label.text()


def test_a_new_recording_is_not_saved(scored_window, tmp_path):
    tab = tab_named(scored_window, "Result")
    tab.output.setText(str(tmp_path))
    tab._save()

    recording, truth = nyx.demo_recording(seed=3)
    scored_window.session.set_recording(recording, reference=truth)
    scored_window.refresh_rail()

    assert "not saved" in scored_window.saved_label.text()


def test_the_logo_ships_and_loads(scored_window):
    from nyx.gui import branding

    assert branding.logo_path() is not None
    assert branding.logo_path(branding.WINDOWS_ICON) is not None
    assert not branding.icon().isNull()
    assert not scored_window.windowIcon().isNull()


def test_the_splash_is_built_from_the_logo(qtbot):
    from nyx.gui import branding

    splash = branding.splash()
    assert splash is not None
    qtbot.addWidget(splash)
    assert not splash.pixmap().isNull()


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
