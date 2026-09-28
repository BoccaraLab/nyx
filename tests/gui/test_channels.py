"""The Recording tab: every channel on screen before two are picked."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.gui.session import ScoringSession

pytestmark = pytest.mark.gui


@pytest.fixture
def load_tab(qtbot):
    from nyx.gui.mainwindow import MainWindow

    window = MainWindow(ScoringSession(nyx.demo_params()))
    qtbot.addWidget(window)
    yield window.tabs[0]
    window.close()


def open_file(tab, path, format="auto"):
    tab.format.setCurrentText(format)
    tab.path.setText(path)
    tab._offer_channels()


def names(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_every_stream_is_drawn_and_every_channel_listed(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)

    docks = load_tab.docks.viewers
    traces = [name for name in docks if name.startswith("stream")]
    assert len(traces) == 2                      # one per stream
    assert load_tab.channel_table.table.rowCount() == 3
    assert "channels in the file" in docks and "what was loaded" in docks


def test_the_traces_are_labelled_by_name_not_id(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)

    viewer = load_tab.docks.panel(next(n for n in load_tab.docks.viewers if "256" in n))
    assert [viewer.source.get_channel_name(i) for i in range(2)] == ["EEG1", "EEG2"]


def test_the_emg_is_guessed_from_its_name(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)

    assert load_tab._chosen(load_tab.eeg_channel).name == "EEG1"
    assert load_tab._chosen(load_tab.emg_channel).name == "EMG"
    assert names(load_tab.emg_channel)[-1] == "(no EMG)"


def test_the_table_marks_what_is_chosen(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)
    table = load_tab.channel_table.table

    assert [table.item(r, 0).text() for r in range(3)] == ["EEG", "", "EMG"]


def test_a_row_can_be_made_the_eeg(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)
    panel = load_tab.channel_table

    panel.table.selectRow(1)
    panel._use("EEG")

    assert load_tab._chosen(load_tab.eeg_channel).name == "EEG2"
    assert panel.table.item(1, 0).text() == "EEG"


def test_loading_reads_the_emg_from_its_own_stream(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf, format="neo")
    load_tab.apply()

    recording = load_tab.session.recording
    assert recording.eeg_channel_name == "EEG1"
    assert recording.emg_channel_name == "EMG"
    assert recording.emg_fs == 128.0
    assert recording.source_options == {"stream_id": "0", "emg_stream_id": "1"}


def test_the_saved_run_can_read_it_again(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf, format="neo")
    load_tab.apply()

    spec = load_tab.session.to_run_config().recording
    assert spec.format == "neo"
    again = nyx.read_recording(
        spec.path, format=spec.format, eeg_channel=spec.eeg_channel,
        emg_channel=spec.emg_channel, **spec.options,
    )
    assert again.emg_channel_name == "EMG" and again.emg_fs == 128.0


def test_the_same_channel_twice_is_refused(load_tab, two_stream_edf):
    open_file(load_tab, two_stream_edf)
    load_tab.emg_channel.setCurrentIndex(load_tab.eeg_channel.currentIndex())

    with pytest.raises(ValueError, match="two different channels"):
        load_tab.apply()


def test_the_neo_format_picker_shows_only_for_neo(load_tab, two_stream_edf):
    form, row = load_tab._form, load_tab.neo_format

    open_file(load_tab, two_stream_edf)          # auto -> edf
    assert not form.isRowVisible(row)
    open_file(load_tab, two_stream_edf, format="neo")
    assert form.isRowVisible(row)


def test_an_npz_has_nothing_to_preview(load_tab, two_stream_edf, tmp_path):
    open_file(load_tab, two_stream_edf)
    path = tmp_path / "r.npz"
    np.savez(path, eeg=np.zeros(100), emg=np.zeros(100), fs=100.0)

    open_file(load_tab, str(path))

    assert list(load_tab.docks.viewers) == ["what was loaded"]
    assert names(load_tab.eeg_channel) == ["eeg", "emg"]
