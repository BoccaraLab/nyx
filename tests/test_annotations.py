"""The generic annotation readers must reproduce the old per-dataset loaders.

Before the refactor there was one bespoke loader per dataset, three of which
were the same run-length-merge with a different label map. These tests pin the
behaviour of the generic readers against the formats those loaders handled.
"""

from __future__ import annotations

import numpy as np
import pytest

from nyx.io.annotations import (
    intervals_from_labels,
    merge_consecutive,
    read_annotations,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_intervals_from_labels_merges_runs():
    result = intervals_from_labels(["WAKE", "WAKE", "NREM", "WAKE"], epoch_length=10.0)

    assert list(result["label"]) == ["WAKE", "NREM", "WAKE"]
    assert list(result["duration"]) == [20.0, 10.0, 10.0]
    assert list(result["time"]) == [0.0, 20.0, 30.0]


def test_intervals_from_labels_handles_empty_and_single():
    assert intervals_from_labels([], epoch_length=4.0)["time"].size == 0

    single = intervals_from_labels(["REM"], epoch_length=4.0)
    assert list(single["time"]) == [0.0]
    assert list(single["duration"]) == [4.0]


def test_merge_consecutive_preserves_start_times():
    merged = merge_consecutive(
        {
            "time": np.array([0.0, 4.0, 8.0, 12.0]),
            "duration": np.array([4.0, 4.0, 4.0, 4.0]),
            "label": np.array(["NREM", "NREM", "REM", "NREM"]),
        }
    )
    assert list(merged["label"]) == ["NREM", "REM", "NREM"]
    assert list(merged["duration"]) == [8.0, 4.0, 4.0]
    assert list(merged["time"]) == [0.0, 8.0, 12.0]


# ---------------------------------------------------------------------------
# interval_csv -- nyx's own output format
# ---------------------------------------------------------------------------


def test_interval_csv_roundtrip(tmp_path):
    path = _write(
        tmp_path,
        "hypno.csv",
        "time,duration,label\n0.0,12.0,WAKE\n12.0,8.0,NREM\n",
    )
    result = read_annotations(path)  # auto-detected from the column names

    assert list(result["label"]) == ["WAKE", "NREM"]
    assert list(result["time"]) == [0.0, 12.0]
    assert list(result["duration"]) == [12.0, 8.0]


def test_interval_csv_reports_missing_columns(tmp_path):
    path = _write(tmp_path, "bad.csv", "t,d,l\n0,4,WAKE\n")
    with pytest.raises(KeyError, match="missing the column"):
        read_annotations(path, format="interval_csv")


# ---------------------------------------------------------------------------
# epoch_csv -- covers the SIESTA, Gulledge and Sippel layouts
# ---------------------------------------------------------------------------


def test_epoch_csv_siesta_layout(tmp_path):
    """One stage code per row, fixed 10 s epochs (was _load_siesta_dli_hypnogram)."""
    path = _write(tmp_path, "scores.csv", "stage\n1\n1\n2\n3\n255\n")
    result = read_annotations(
        path,
        format="epoch_csv",
        epoch_length=10.0,
        label_map={1: "WAKE", 2: "NREM", 3: "REM", 255: "UNDEFINED"},
    )

    assert list(result["label"]) == ["WAKE", "NREM", "REM", "UNDEFINED"]
    assert list(result["duration"]) == [20.0, 10.0, 10.0, 10.0]


def test_epoch_csv_gulledge_layout(tmp_path):
    """Clock time column, epoch length inferred (was _load_gulledge_hypnogram)."""
    path = _write(
        tmp_path,
        "staging.csv",
        "Time Stamp,Rodent Sleep\n00:00,5\n00:10,5\n00:20,3\n00:30,4\n",
    )
    result = read_annotations(
        path,
        format="epoch_csv",
        stage_column="Rodent Sleep",
        time_column="Time Stamp",
        time_format="clock",
        label_map={3: "NREM", 4: "REM", 5: "WAKE"},
    )

    assert list(result["label"]) == ["WAKE", "NREM", "REM"]
    assert list(result["duration"]) == [20.0, 10.0, 10.0]
    assert list(result["time"]) == [0.0, 20.0, 30.0]


def test_epoch_csv_infers_epoch_length_from_clock_column(tmp_path):
    path = _write(tmp_path, "s.csv", "t,stage\n00:00,5\n00:04,5\n00:08,3\n")
    result = read_annotations(
        path, format="epoch_csv", stage_column="stage", time_column="t",
        time_format="clock", label_map={3: "NREM", 5: "WAKE"},
    )
    assert list(result["duration"]) == [8.0, 4.0]


def test_epoch_csv_clock_times_unwrap_across_midnight(tmp_path):
    """Overnight recordings restart the clock at 00:00:00 partway through.

    Without unwrapping, every epoch after midnight lands ~24 h too early and
    stops overlapping the recording, which quietly halves the comparison.
    """
    path = _write(
        tmp_path,
        "staging.csv",
        "Time Stamp,Rodent Sleep\n"
        "23:59:40,5\n"
        "23:59:50,3\n"
        "00:00:00,3\n"
        "00:00:10,4\n",
    )
    result = read_annotations(
        path,
        format="epoch_csv",
        stage_column="Rodent Sleep",
        time_column="Time Stamp",
        time_format="clock",
        label_map={3: "NREM", 4: "REM", 5: "WAKE"},
    )

    assert list(result["label"]) == ["WAKE", "NREM", "REM"]
    # Times must keep increasing straight through the wrap.
    assert list(result["time"]) == [0.0, 10.0, 30.0]
    assert np.all(np.diff(result["time"]) > 0)


def test_epoch_csv_sippel_layout_with_duration(tmp_path):
    """Datetime timestamps plus an explicit duration (was _load_sippelmorris_hypnogram)."""
    path = _write(
        tmp_path,
        "epochs.txt",
        "time,duration,label\n"
        "2023-01-01 00:00:00,4,WAKE\n"
        "2023-01-01 00:00:04,4,WAKE\n"
        "2023-01-01 00:00:08,4,NREM\n",
    )
    result = read_annotations(
        path,
        format="epoch_csv",
        time_column="time",
        time_format="datetime",
        duration_column="duration",
        stage_column="label",
    )

    # Times are made relative to the first epoch, and runs are merged.
    assert list(result["label"]) == ["WAKE", "NREM"]
    assert list(result["time"]) == [0.0, 8.0]
    assert list(result["duration"]) == [8.0, 4.0]


def test_epoch_csv_label_map_is_type_tolerant(tmp_path):
    """Stage codes read as floats must still match int (or JSON string) keys."""
    path = _write(tmp_path, "s.csv", "stage\n3.0\n4.0\n")

    from_int = read_annotations(
        path, format="epoch_csv", epoch_length=10.0, label_map={3: "NREM", 4: "REM"}
    )
    from_str = read_annotations(
        path, format="epoch_csv", epoch_length=10.0, label_map={"3": "NREM", "4": "REM"}
    )

    assert list(from_int["label"]) == ["NREM", "REM"]
    assert list(from_str["label"]) == ["NREM", "REM"]


def test_epoch_csv_unmapped_codes_become_unknown(tmp_path):
    path = _write(tmp_path, "s.csv", "stage\n1\n99\n")
    result = read_annotations(
        path, format="epoch_csv", epoch_length=4.0, label_map={1: "WAKE"}
    )
    assert list(result["label"]) == ["WAKE", "UNKNOWN"]


def test_epoch_csv_requires_an_epoch_length_it_cannot_infer(tmp_path):
    path = _write(tmp_path, "s.csv", "stage\n1\n2\n")
    with pytest.raises(ValueError, match="epoch length"):
        read_annotations(path, format="epoch_csv")


# ---------------------------------------------------------------------------
# column_csv -- one column per recording (the ellen_dash layout)
# ---------------------------------------------------------------------------


def test_column_csv_selects_the_requested_recording(tmp_path):
    path = _write(
        tmp_path,
        "all_rats.csv",
        "ignored header row\n"
        "idx,'rat1','rat2'\n"
        "0,0,2\n"
        "1,0,2\n"
        "2,1,1\n",
    )
    result = read_annotations(
        path,
        format="column_csv",
        column="rat1",
        epoch_length=4.0,
        label_map={0: "WAKE", 1: "NREM", 2: "REM"},
    )

    assert list(result["label"]) == ["WAKE", "NREM"]
    assert list(result["duration"]) == [8.0, 4.0]


def test_column_csv_requires_a_column(tmp_path):
    """The old loader raised a bare ValueError here and was called without the
    column at all, so this path could never succeed."""
    path = _write(tmp_path, "all.csv", "hdr\nidx,'rat1'\n0,0\n")
    with pytest.raises(ValueError, match="column="):
        read_annotations(path, format="column_csv")


def test_column_csv_reports_available_columns(tmp_path):
    path = _write(tmp_path, "all.csv", "hdr\nidx,'rat1','rat2'\n0,0,1\n")
    with pytest.raises(KeyError, match="rat3"):
        read_annotations(path, format="column_csv", column="rat3")


# ---------------------------------------------------------------------------
# visbrain_hyp -- the Oxford benchmark layout
# ---------------------------------------------------------------------------


def test_visbrain_hyp(tmp_path):
    path = _write(
        tmp_path,
        "scores.hyp",
        "*Datafile\tunknown\n*Duration_sec\t40\nWAKE\t10.0\nNREM\t30.0\nREM\t40.0\n",
    )
    result = read_annotations(path)  # auto-detected from the .hyp extension

    assert list(result["label"]) == ["WAKE", "NREM", "REM"]
    assert list(result["time"]) == [0.0, 10.0, 30.0]
    assert list(result["duration"]) == [10.0, 20.0, 10.0]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        read_annotations(str(tmp_path / "nope.csv"))


def test_unknown_format_lists_the_available_ones(tmp_path):
    path = _write(tmp_path, "h.csv", "time,duration,label\n0,4,WAKE\n")
    with pytest.raises(ValueError, match="Available"):
        read_annotations(path, format="not_a_format")


def test_custom_reader_can_be_registered(tmp_path):
    from nyx.io.annotations import ANNOTATION_READERS, register_annotation_reader

    @register_annotation_reader("test_only_format")
    def _reader(path, **kwargs):
        return {
            "time": np.array([0.0]),
            "duration": np.array([1.0]),
            "label": np.array(["WAKE"]),
        }

    try:
        path = _write(tmp_path, "x.csv", "anything\n")
        assert list(read_annotations(path, format="test_only_format")["label"]) == ["WAKE"]
    finally:
        ANNOTATION_READERS.pop("test_only_format", None)
