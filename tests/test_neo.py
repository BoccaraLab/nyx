"""Reading through Neo, and looking at every channel before choosing two.

The file is the ``two_stream_edf`` fixture: an EDF read as ``format="neo"``,
which goes the same way an Open Ephys folder or an Intan file would.
"""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.io import (
    describe_channels,
    guess_neo_format,
    neo_formats,
    neo_streams,
    recording_streams,
)


def test_the_formats_come_from_spikeinterface_not_a_list():
    formats = neo_formats()
    assert {"OpenEphysBinary", "Intan", "SpikeGLX", "Spike2", "EDF"} <= set(formats)
    assert formats["OpenEphysBinary"].is_folder
    assert not formats["Intan"].is_folder


def test_neo_recognises_the_file(two_stream_edf):
    assert guess_neo_format(two_stream_edf) == "EDF"


def test_a_file_neo_does_not_know_is_not_guessed(tmp_path):
    path = tmp_path / "notes.nothing"
    path.write_text("not a recording")
    assert guess_neo_format(str(path)) is None


def test_every_stream_is_found(two_stream_edf):
    assert [stream_id for stream_id, _name in neo_streams(two_stream_edf)] == ["0", "1"]


def test_every_channel_is_described_with_what_the_file_says(two_stream_edf):
    channels = describe_channels(recording_streams(two_stream_edf, format="neo"))

    assert [(c.name, c.stream_id, c.fs) for c in channels] == [
        ("EEG1", "0", 256.0), ("EEG2", "0", 256.0), ("EMG", "1", 128.0),
    ]
    assert all(c.unit == "uV" and c.gain for c in channels)
    assert channels[0].duration == pytest.approx(10.0)


def test_the_streams_are_lazy_multichannel_recordings(two_stream_edf):
    streams = recording_streams(two_stream_edf, format="edf")
    assert [rec.get_num_channels() for _id, _name, rec in streams] == [2, 1]


def test_npz_has_no_streams_to_browse(tmp_path):
    path = tmp_path / "r.npz"
    np.savez(path, eeg=np.zeros(10), emg=np.zeros(10), fs=10.0)
    with pytest.raises(LookupError):
        recording_streams(str(path))


def test_reading_through_neo(two_stream_edf):
    recording = nyx.read_recording(
        two_stream_edf, format="neo", eeg_channel="EEG2", emg_channel="EEG1"
    )
    assert recording.eeg_channel_name == "EEG2"
    assert recording.emg_channel_name == "EEG1"
    assert recording.fs == 256.0


def test_the_emg_can_come_from_another_stream_at_its_own_rate(two_stream_edf):
    recording = nyx.read_recording(
        two_stream_edf, format="neo", neo_format="EDF",
        eeg_channel="EEG1", emg_channel="EMG", emg_stream_id="1",
    )
    assert recording.emg_channel_name == "EMG"
    assert recording.fs == 256.0
    assert recording.emg_fs == 128.0


def test_the_recording_remembers_how_it_was_read(two_stream_edf):
    recording = nyx.read_recording(
        two_stream_edf, format="neo", eeg_channel="EEG1", emg_channel="EMG",
        emg_stream_id="1",
    )
    assert recording.source_format == "neo"
    assert recording.source_options == {"emg_stream_id": "1"}

    # ...and keeps remembering it through a window.
    sliced = recording.time_slice(1.0, 5.0)
    assert sliced.source_format == "neo"
    assert sliced.source_options == {"emg_stream_id": "1"}


def test_an_unknown_extension_is_offered_to_neo(tmp_path, two_stream_edf):
    from nyx.io.recordings import _infer_format

    assert _infer_format(two_stream_edf) == "edf"   # nyx's own reader first
    with pytest.raises(ValueError, match="neo_format"):
        _infer_format(str(tmp_path / "mystery.xyz"))


def test_a_folder_spikeinterface_saved_is_still_spikeinterface(tmp_path):
    from spikeinterface.core import NumpyRecording

    from nyx.io.recordings import _infer_format

    folder = tmp_path / "saved"
    NumpyRecording([np.zeros((50, 2), "float32")], 100.0).save(
        folder=str(folder), verbose=False
    )
    assert _infer_format(str(folder)) == "spikeinterface"


# -- Intan layouts: no Intan files ship with the tests, so these are about
#    telling the layouts apart, which is what decides whether the check runs.


def test_a_single_intan_file_is_header_attached(tmp_path):
    from nyx.io.neo import intan_layout, intan_unchecked_note

    path = tmp_path / "recording_1.rhs"
    path.write_bytes(b"")
    assert intan_layout(str(path)) == "header-attached"
    assert "not checked" in intan_unchecked_note(str(path))


def test_a_headerless_intan_folder_is_found_from_any_of_its_files(tmp_path):
    from nyx.io.neo import _intan_info_file, intan_layout, intan_unchecked_note

    for name in ("info.rhs", "amplifier.dat", "time.dat"):
        (tmp_path / name).write_bytes(b"")
    info = str(tmp_path / "info.rhs")

    for pointed_at in (tmp_path, tmp_path / "info.rhs", tmp_path / "amplifier.dat"):
        assert _intan_info_file(str(pointed_at)) == info
        assert intan_layout(str(pointed_at)) == "one-file-per-signal"
    assert intan_unchecked_note(info) is None       # its check still runs
    assert guess_neo_format(str(tmp_path)) == "Intan"


def test_other_files_are_not_intan(two_stream_edf):
    from nyx.io.neo import intan_layout

    assert intan_layout(two_stream_edf) is None


def test_the_check_is_only_ever_switched_off_for_the_open(tmp_path):
    from neo.rawio.intanrawio import IntanRawIO

    from nyx.io.neo import _intan_checks

    path = tmp_path / "recording_1.rhs"
    path.write_bytes(b"")
    original = IntanRawIO._assert_timestamp_continuity
    with pytest.warns(UserWarning, match="not checked"):
        with _intan_checks(str(path), "Intan"):
            assert IntanRawIO._assert_timestamp_continuity is not original
    assert IntanRawIO._assert_timestamp_continuity is original
