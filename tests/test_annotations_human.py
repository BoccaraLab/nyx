"""Human scoring formats, ported from the human pipeline's bespoke loaders.

`dodh`/`dodo`, `sleeping` and the NSRR datasets each had their own loader, and
three of them re-implemented the same run-length merge. These tests pin the
generic readers against those formats.
"""

from __future__ import annotations

import numpy as np
import pytest

from nyx.io.annotations import NSRR_CONCEPTS, read_annotations

# Dreem Open Dataset coding.
DODH_LABELS = {-1: "NOSIGNAL", 0: "WAKE", 1: "NREM1", 2: "NREM2", 3: "NREM3", 4: "REM"}


# ---------------------------------------------------------------------------
# epoch_npy -- dodh / dodo
# ---------------------------------------------------------------------------


def test_epoch_npy_bare_array(tmp_path):
    path = tmp_path / "hypno.npy"
    np.save(path, np.array([0, 0, 2, 2, 2, 4, -1]))

    result = read_annotations(
        str(path), format="epoch_npy", epoch_length=30.0, label_map=DODH_LABELS
    )

    assert list(result["label"]) == ["WAKE", "NREM2", "REM", "NOSIGNAL"]
    assert list(result["duration"]) == [60.0, 90.0, 30.0, 30.0]
    assert list(result["time"]) == [0.0, 60.0, 150.0, 180.0]


def test_epoch_npy_dict_payload(tmp_path):
    """Some of these files are a pickled dict rather than a bare array."""
    path = tmp_path / "hypno.npy"
    np.save(path, {"label": np.array([0, 4, 4])}, allow_pickle=True)

    result = read_annotations(
        str(path), format="epoch_npy", epoch_length=30.0, label_map=DODH_LABELS
    )
    assert list(result["label"]) == ["WAKE", "REM"]
    assert list(result["duration"]) == [30.0, 60.0]


# ---------------------------------------------------------------------------
# epoch_mat -- the 'sleeping' dataset
# ---------------------------------------------------------------------------


def test_epoch_mat_reads_epoch_length_from_the_struct(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "scores.mat"
    savemat(str(path), {"stageData": {"stages": np.array([0, 0, 2, 5]), "win": 20.0}})

    result = read_annotations(
        str(path),
        format="epoch_mat",
        label_map={0: "WAKE", 1: "NREM1", 2: "NREM2", 3: "NREM3", 5: "REM", 6: "NOSIGNAL"},
    )

    assert list(result["label"]) == ["WAKE", "NREM2", "REM"]
    # 20 s comes from the struct's `win` field, not from a default.
    assert list(result["duration"]) == [40.0, 20.0, 20.0]


def test_epoch_mat_reports_a_missing_struct(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "scores.mat"
    savemat(str(path), {"somethingElse": np.array([1, 2])})

    with pytest.raises(KeyError, match="stageData"):
        read_annotations(str(path), format="epoch_mat")


# ---------------------------------------------------------------------------
# nsrr_xml -- MESA / CHAT / CCSHS / SHHS
# ---------------------------------------------------------------------------


NSRR_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
  <ScoredEvents>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Wake|0</EventConcept>
      <Start>0.0</Start><Duration>60.0</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Arousals|Arousals</EventType>
      <EventConcept>Arousal|Arousal</EventConcept>
      <Start>30.0</Start><Duration>5.0</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Stage 2 sleep|2</EventConcept>
      <Start>60.0</Start><Duration>30.0</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Stage 4 sleep|4</EventConcept>
      <Start>90.0</Start><Duration>30.0</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Stage 3 sleep|3</EventConcept>
      <Start>120.0</Start><Duration>30.0</Duration>
    </ScoredEvent>
  </ScoredEvents>
</PSGAnnotation>
"""


def test_nsrr_xml_reads_stages_and_ignores_other_events(tmp_path):
    path = tmp_path / "sub-nsrr.xml"
    path.write_text(NSRR_SAMPLE, encoding="utf-8")

    result = read_annotations(str(path), format="nsrr_xml")

    # The arousal event must not appear, and stage 4 folds into N3 (AASM), so
    # the last two events merge into one interval.
    assert list(result["label"]) == ["WAKE", "NREM2", "NREM3"]
    assert list(result["time"]) == [0.0, 60.0, 90.0]
    assert list(result["duration"]) == [60.0, 30.0, 60.0]


def test_nsrr_xml_errors_when_there_are_no_stage_events(tmp_path):
    path = tmp_path / "empty-nsrr.xml"
    path.write_text("<PSGAnnotation><ScoredEvents/></PSGAnnotation>", encoding="utf-8")

    with pytest.raises(ValueError, match="No 'Stages' events"):
        read_annotations(str(path), format="nsrr_xml")


def test_nsrr_concept_map_folds_stage_4_into_n3():
    assert NSRR_CONCEPTS["Stage 4 sleep|4"] == "NREM3"
    assert NSRR_CONCEPTS["Stage 3 sleep|3"] == "NREM3"


# ---------------------------------------------------------------------------
# interval_csv -- the tab-separated anphy layout needs no bespoke reader
# ---------------------------------------------------------------------------


def test_interval_csv_headerless_tab_separated(tmp_path):
    """anphy files are `label<TAB>time<TAB>duration` with no header row."""
    path = tmp_path / "scores.txt"
    path.write_text("WAKE\t0\t30\nNREM2\t30\t60\n", encoding="utf-8")

    result = read_annotations(
        str(path),
        format="interval_csv",
        names=["label", "time", "duration"],
        sep="\t",
    )

    assert list(result["label"]) == ["WAKE", "NREM2"]
    assert list(result["time"]) == [0.0, 30.0]
    assert list(result["duration"]) == [30.0, 60.0]
