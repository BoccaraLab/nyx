"""Reading recordings, and looking at one before reading it."""

from __future__ import annotations

import numpy as np
import pytest

from nyx.io import (
    CHANNEL_LISTERS,
    RECORDING_EXTENSIONS,
    RECORDING_READERS,
    list_channels,
    register_channel_lister,
    register_recording_reader,
)


@pytest.fixture
def npz_file(tmp_path):
    path = tmp_path / "rec.npz"
    np.savez(path, eeg=np.zeros(100), emg=np.zeros(100), fs=100.0)
    return str(path)


def test_listing_the_channels_of_an_npz(npz_file):
    assert list_channels(npz_file) == ["eeg", "emg"]


def test_an_npz_without_an_emg_lists_only_the_eeg(tmp_path):
    path = tmp_path / "eeg_only.npz"
    np.savez(path, eeg=np.zeros(100), fs=100.0)

    assert list_channels(str(path)) == ["eeg"]


def test_a_missing_file_says_so_before_anything_else(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_channels(str(tmp_path / "nope.npz"))


def test_a_format_with_no_lister_says_which_formats_have_one(npz_file):
    @register_recording_reader("listerless")
    def _reader(path, eeg_channel, emg_channel, **kwargs):  # pragma: no cover
        raise AssertionError("never called")

    try:
        with pytest.raises(LookupError, match="npz"):
            list_channels(npz_file, format="listerless")
    finally:
        RECORDING_READERS.pop("listerless", None)


def test_a_registered_lister_is_used(npz_file):
    @register_channel_lister("test_only_format")
    def _lister(path, **kwargs):
        return ["one", "two"]

    try:
        assert list_channels(npz_file, format="test_only_format") == ["one", "two"]
    finally:
        CHANNEL_LISTERS.pop("test_only_format", None)


def test_every_shipped_reader_that_can_be_browsed_has_a_lister():
    # Not every format must be browsable, but the ones nyx ships are, and a
    # file dialog built from RECORDING_READERS would otherwise offer a format
    # whose channels it cannot show.
    assert set(CHANNEL_LISTERS) == set(RECORDING_READERS)


def test_the_extension_map_only_names_formats_that_exist():
    assert set(RECORDING_EXTENSIONS.values()) <= set(RECORDING_READERS)
