"""nyx-manual: scoring by hand, where nothing can be written off the grid."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.gui.session import ScoringSession, Stage

pytestmark = pytest.mark.gui


@pytest.fixture
def manual_window(qtbot, synthetic_recording, synthetic):
    from nyx.gui.mainwindow import MainWindow, manual_tab_classes

    _eeg, _emg, _fs, truth = synthetic
    session = ScoringSession(nyx.demo_params())
    session.set_recording(synthetic_recording, reference=truth)
    session.set_window((10.0, 1210.0))
    session.compute_through(Stage.PREPROCESS)

    # Not qtbot.addWidget: pytest-qt closes what it tracks before fixtures are
    # torn down, and closing over unsaved scoring asks a question -- a modal
    # dialog nobody can answer offscreen. This fixture closes it instead,
    # once the scoring is marked saved.
    window = MainWindow(session, tabs=manual_tab_classes(), title="nyx -- manual")
    tab = window.tabs[-1]
    window.rail.setCurrentRow(len(window.tabs) - 1)
    tab.safe_refresh()
    yield window
    tab._saved_state = tab._state_key(tab.scored())   # no save prompt on close
    window.close()


@pytest.fixture
def tab(manual_window):
    return manual_window.tabs[-1]


def epochs(tab):
    s = tab.source
    return [(float(t), float(d), str(label))
            for t, d, label in zip(s.ep_times, s.ep_durations, s.ep_labels)]


def assert_on_grid(tab):
    grid = tab.source.grid
    for start, length, _label in epochs(tab):
        assert grid.nearest(start) == pytest.approx(start)
        assert grid.nearest(start + length) == pytest.approx(start + length)
        assert length > 0


def test_the_manual_window_is_recording_signal_check_then_scoring(manual_window):
    assert [t.title for t in manual_window.tabs] == [
        "Recording", "Signal check", "Manual scoring"
    ]


def test_the_wavelet_view_is_the_default(tab):
    from nyx.gui.viewers import NyxTimeFreqViewer

    assert isinstance(tab.docks.panel("EEG spectrum"), NyxTimeFreqViewer)
    assert isinstance(tab.docks.panel("EMG spectrum"), NyxTimeFreqViewer)


@pytest.mark.parametrize("wavelet", [True, False])
def test_the_spectrograms_carry_no_labels(tab, wavelet):
    tab.scalogram.setChecked(wavelet)
    for name in ("EEG spectrum", "EMG spectrum"):
        assert tab.docks.panel(name).params["display_labels"] is False


def test_a_key_scores_the_epoch_the_cursor_is_in_and_moves_on(tab):
    tab.encoder.t = 5.3                      # inside the epoch 2-6 on this grid
    tab.encoder.on_label_shortcut("WAKE", False)

    assert epochs(tab) == [(2.0, 4.0, "WAKE")]
    assert tab.encoder.t == pytest.approx(6.0)


def test_the_same_stage_in_a_row_merges_as_you_score(tab):
    tab.encoder.t = 2.0
    for label in ("WAKE", "WAKE", "WAKE", "NREM"):
        tab.encoder.on_label_shortcut(label, False)

    assert epochs(tab) == [(2.0, 12.0, "WAKE"), (14.0, 4.0, "NREM")]


def test_relabelling_in_the_table_merges_too(tab):
    tab.encoder.t = 2.0
    for label in ("WAKE", "REM", "WAKE"):
        tab.encoder.on_label_shortcut(label, False)

    tab.encoder.on_change_label(1, "WAKE")

    assert epochs(tab) == [(2.0, 12.0, "WAKE")]


def test_scoring_over_the_middle_of_a_bout_splits_it(tab):
    tab.encoder.t = 2.0
    for label in ("WAKE", "WAKE", "WAKE"):
        tab.encoder.on_label_shortcut(label, False)
    tab.encoder.t = 6.0
    tab.encoder.on_label_shortcut("REM", False)

    assert epochs(tab) == [(2.0, 4.0, "WAKE"), (6.0, 4.0, "REM"), (10.0, 4.0, "WAKE")]


def test_the_overlap_modifier_does_not_stack_stages(tab):
    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("NREM", True)   # shift held

    assert epochs(tab) == [(2.0, 4.0, "NREM")]


def test_a_range_is_widened_to_whole_epochs(tab):
    enc = tab.encoder
    enc.range_group_box.setChecked(True)
    enc.region.setRegion((3.0, 11.5))
    enc.combo_labels.setCurrentText("REM")
    enc.apply_region()

    assert epochs(tab) == [(2.0, 12.0, "REM")]


def test_typed_boundaries_are_not_taken(tab):
    from ephyviewer.epochencoder import START_COL

    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    table = tab.encoder.table_widget
    table.item(0, START_COL).setText("3.3")

    assert epochs(tab) == [(2.0, 4.0, "WAKE")]
    # ...and the table says so, rather than showing a start that is not there.
    assert float(table.item(0, START_COL).text()) == pytest.approx(2.0)


def test_an_epoch_cannot_be_split_off_the_grid(tab):
    tab.encoder.range_group_box.setChecked(True)
    tab.encoder.region.setRegion((2.0, 14.0))
    tab.encoder.apply_region()                   # 2-14, three epochs

    tab.encoder.t = 7.1
    tab.encoder.split_selected_epoch(0)
    assert [e[:2] for e in epochs(tab)] == [(2.0, 4.0), (6.0, 8.0)]
    assert_on_grid(tab)


def test_duplicating_is_refused(tab):
    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    tab.encoder.duplicate_selected_epoch(0)
    assert len(epochs(tab)) == 1


def test_whatever_reaches_the_source_ends_up_on_the_grid(tab):
    """The backstop: even an edit the encoder never makes is snapped."""
    tab.source.add_epoch(3.3, 5.1, "REM")
    tab.source.fill_blank(method="from_nearest")
    tab.encoder.on_undo()
    tab.encoder.on_redo()
    assert_on_grid(tab)


def test_the_epoch_length_is_fixed_once_something_is_scored(tab):
    assert tab.epoch_length.isEnabled()
    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    assert not tab.epoch_length.isEnabled()
    assert "1 of 299 epochs" in tab.progress.text()


def test_another_epoch_length_is_another_grid(tab):
    tab.epoch_length.setValue(10.0)
    tab.encoder.t = 3.0
    tab.encoder.on_label_shortcut("NREM", False)
    # Recording lines every 10 s: 10 s in the recording is 0 on the viewer.
    assert epochs(tab) == [(0.0, 10.0, "NREM")]


def test_saved_times_are_the_recordings_and_can_be_continued(tab, tmp_path):
    tab.encoder.t = 2.0
    for label in ("WAKE", "WAKE", "NREM"):
        tab.encoder.on_label_shortcut(label, False)
    path = str(tmp_path / "scored.csv")
    tab.path.setText(path)
    tab.save()

    saved = nyx.read_annotations(path, format="interval_csv")
    assert list(saved["time"]) == [12.0, 20.0]       # window starts at 10
    assert list(saved["label"]) == ["WAKE", "NREM"]
    assert not tab.unsaved()

    tab.resume(path)
    assert epochs(tab) == [(2.0, 8.0, "WAKE"), (10.0, 4.0, "NREM")]


def test_a_scoring_off_the_grid_is_put_on_it_when_continued(tab, tmp_path):
    from nyx.scoring import savehypno

    path = str(tmp_path / "other.csv")
    savehypno({"time": np.array([11.0]), "duration": np.array([6.5]),
               "label": np.array(["REM"])}, path)
    tab.resume(path)
    assert_on_grid(tab)


def test_the_reference_stays_hidden_until_you_compare(tab):
    assert "reference" not in tab.docks.viewers

    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    tab.compare.setChecked(True)

    assert "reference" in tab.docks.viewers
    assert tab.agreement_with_reference() is not None
    assert "epochs you have scored" in tab.agreement.text.toPlainText()


def test_closing_over_unsaved_scoring_asks(tab, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    tab.encoder.t = 2.0
    tab.encoder.on_label_shortcut("WAKE", False)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)

    assert tab.unsaved()
    assert tab.confirm_close() is False


def test_the_automatic_editor_is_left_free_form(scored):
    """The grid is nyx-manual's alone: correcting a scoring needs free edges."""
    from nyx.gui.review import NyxEpochSource

    source = NyxEpochSource(scored.staging().hypnogram)
    source.add_epoch(3.3, 5.1, "REM")
    assert 3.3 in np.round(source.ep_times, 6)
