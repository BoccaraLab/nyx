"""A saved run must be re-runnable from its own record, with nothing to decide."""

from __future__ import annotations

import json

import numpy as np
import pytest

import nyx
from nyx.config import build_run_record, load_config
from nyx.steps import Step


def _saved(recording, params, tmp_path, **kwargs):
    result = nyx.score_recording(recording, params, window=(0, 1800), verbose=False)
    out = tmp_path / "run"
    nyx.save_results(result, str(out), **kwargs)
    return result, out


def test_run_json_is_written(synthetic_recording, params, tmp_path):
    _result, out = _saved(synthetic_recording, params, tmp_path)
    assert (out / "run.json").exists()


def test_run_record_inlines_the_params(synthetic_recording, params, tmp_path):
    """A record that points at a params file is only as reproducible as that
    file, which may change or move."""
    _result, out = _saved(synthetic_recording, params, tmp_path)
    record = json.loads((out / "run.json").read_text())

    assert isinstance(record["params"], dict)
    assert record["params"]["EEG"]["binsize"] == params["EEG"]["binsize"]


def test_run_record_captures_the_decisions(synthetic_recording, params, tmp_path):
    result, out = _saved(synthetic_recording, params, tmp_path)
    record = json.loads((out / "run.json").read_text())

    assert record["window"] == [0.0, 1800.0]
    assert record["decisions"]["emg"]["thresholds"] == result.wake_sleep.thresholds
    assert record["decisions"]["emg"]["source"] == result.wake_sleep.threshold_source

    # These params carry a `steps` list, so the cluster mapping is recorded per
    # step rather than once for the whole run.
    split = next(s for s in record["params"]["steps"] if s["name"] == "split_sleep")
    assert split["cluster_to_stage"] == {
        str(k): v for k, v in result.staging.cluster_to_stage.items()
    }


def test_a_single_step_run_records_the_mapping_outside_the_steps(
    synthetic_recording, params, tmp_path
):
    """Without a `steps` list there is nowhere per-step to put it."""
    single = {k: v for k, v in params.items() if k != "steps"}
    result, out = _saved(synthetic_recording, single, tmp_path)
    record = json.loads((out / "run.json").read_text())

    assert record["decisions"]["cluster_to_stage"] == {
        str(k): v for k, v in result.staging.cluster_to_stage.items()
    }


def test_the_wake_sleep_step_survives_in_the_record(
    synthetic_recording, params, tmp_path
):
    """Only clustering steps are run, so only they come back from the step
    engine. Recording just those would drop the wake/sleep split from the
    record, and a record that cannot replay that cannot replay the scoring."""
    _result, out = _saved(synthetic_recording, params, tmp_path)
    record = json.loads((out / "run.json").read_text())

    names = [s["name"] for s in record["params"]["steps"]]
    assert names == [s["name"] for s in params["steps"]]
    assert record["params"]["steps"][0]["method"] == "emg_threshold"


def test_run_record_notes_the_version(synthetic_recording, params, tmp_path):
    _result, out = _saved(synthetic_recording, params, tmp_path)
    record = json.loads((out / "run.json").read_text())
    assert record["nyx_version"] == nyx.__version__


def test_run_record_reloads_as_a_config(synthetic_recording, params, tmp_path):
    """The point of the record: load it back and everything is already decided."""
    result, out = _saved(synthetic_recording, params, tmp_path)

    # A record from a recording loaded directly has no recording section, so
    # give it one, as a real saved run would have from its config.
    record = json.loads((out / "run.json").read_text())
    record["recording"] = {"path": synthetic_recording.source_path}
    (out / "run.json").write_text(json.dumps(record))

    reloaded = load_config(str(out / "run.json"))

    assert reloaded.window == tuple(result.window)
    assert reloaded.params["EEG"] == params["EEG"]
    assert reloaded.decisions["emg"]["thresholds"] == result.wake_sleep.thresholds


def test_replaying_a_record_needs_no_interaction(synthetic_recording, params, tmp_path):
    """Re-running from the record must reproduce the scoring exactly."""
    first = nyx.score_recording(synthetic_recording, params, window=(0, 1800),
                                verbose=False)
    out = tmp_path / "run"
    nyx.save_results(first, str(out))
    record = json.loads((out / "run.json").read_text())

    # No cluster_overrides: each step's resolved mapping is inside the recorded
    # params, so replaying needs only the params and the thresholds.
    second = nyx.score_recording(
        synthetic_recording,
        record["params"],
        window=tuple(record["window"]),
        emg_threshold=record["decisions"]["emg"]["thresholds"][1],
        nosignal_threshold=record["decisions"]["emg"]["thresholds"][0],
        verbose=False,
    )

    assert second.staging.stage_durations() == first.staging.stage_durations()


def test_replaying_a_single_step_record_needs_no_interaction(
    synthetic_recording, params, tmp_path
):
    single = {k: v for k, v in params.items() if k != "steps"}
    first = nyx.score_recording(synthetic_recording, single, window=(0, 1800),
                                verbose=False)
    out = tmp_path / "run"
    nyx.save_results(first, str(out))
    record = json.loads((out / "run.json").read_text())

    second = nyx.score_recording(
        synthetic_recording,
        record["params"],
        window=tuple(record["window"]),
        emg_threshold=record["decisions"]["emg"]["thresholds"][1],
        nosignal_threshold=record["decisions"]["emg"]["thresholds"][0],
        cluster_overrides={
            int(k): v for k, v in record["decisions"]["cluster_to_stage"].items()
        },
        verbose=False,
    )

    assert second.staging.stage_durations() == first.staging.stage_durations()


# ---------------------------------------------------------------------------
# Granularities
# ---------------------------------------------------------------------------


def test_granularity_files_are_written(synthetic_recording, params, tmp_path):
    _result, out = _saved(synthetic_recording, params, tmp_path, granularities=[3])
    assert (out / "hypnogram_3stage.csv").exists()


def test_granularity_files_agree_with_the_full_hypnogram(
    synthetic_recording, params, tmp_path
):
    """Coarse files are merges of the fine one, so total time must match."""
    import pandas as pd

    _result, out = _saved(synthetic_recording, params, tmp_path, granularities=[3])

    fine = pd.read_csv(out / "hypnogram.csv")
    coarse = pd.read_csv(out / "hypnogram_3stage.csv")
    assert coarse["duration"].sum() == pytest.approx(fine["duration"].sum())


# ---------------------------------------------------------------------------
# Multi-step runs
# ---------------------------------------------------------------------------


def test_multi_step_record_carries_each_step_mapping(synthetic_recording, params):
    """A multi-step run records what each step's clusters were called, which is
    the part that cannot be recovered from the params alone."""
    steps = [
        Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
             stage_order=["REM", "NREM"]),
    ]
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    outcomes = nyx.run_steps(synthetic_recording, params, wake_sleep.hypnogram,
                             steps, emg=emg, verbose=False)

    record = build_run_record(None, params, steps=outcomes, window=(0, 100))

    saved_step = next(
        s for s in record["params"]["steps"] if s["name"] == "split_sleep"
    )
    assert saved_step["within"] == "SLEEP"
    assert set(saved_step["cluster_to_stage"].values()) <= {"REM", "NREM"}


def test_recorded_steps_reload_as_steps(synthetic_recording, params):
    steps = [Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
                  stage_order=["REM", "NREM"])]
    record = build_run_record(None, params, steps=steps, window=(0, 100))

    rebuilt = [Step.from_dict(s) for s in record["params"]["steps"]]
    assert steps[0] in rebuilt


def test_record_is_json_serialisable_with_numpy_values():
    """Cluster ids come back from numpy as np.int64, which json rejects."""
    steps = [Step("s", method="kmeans",
                  cluster_to_stage={np.int64(0): "REM", np.int64(1): "NREM"})]
    record = build_run_record(None, {"EEG": {}, "EMG": {}}, steps=steps)

    text = json.dumps(record, default=str)
    assert "REM" in text
