"""Working through a manifest one recording at a time, resumably."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import nyx
from nyx.queue import DONE, FAILED, PENDING, SKIPPED


@pytest.fixture
def study(tmp_path, synthetic_recording, params):
    """A three-recording manifest pointing at copies of the synthetic signal."""
    data = tmp_path / "data"
    data.mkdir()

    eeg = synthetic_recording.eeg_trace()
    emg = synthetic_recording.emg_trace()
    rows = []
    for i in range(3):
        path = data / f"m{i:02d}.npz"
        np.savez(path, eeg=eeg, emg=emg, fs=synthetic_recording.fs)
        rows.append({"recording_path": str(path), "name": f"m{i:02d}"})

    manifest = tmp_path / "recordings.csv"
    nyx.write_manifest(str(manifest), rows)

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    return {
        "manifest": str(manifest),
        "params": str(params_path),
        "output": str(tmp_path / "results"),
        "config": str(tmp_path / "config.json"),
        "tmp": tmp_path,
    }


def open_it(study, **kwargs):
    return nyx.open_queue(
        study["manifest"],
        params=study["params"],
        output_root=study["output"],
        config_path=study["config"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# What is left
# ---------------------------------------------------------------------------


def test_a_fresh_queue_has_everything_pending(study):
    queue = open_it(study)

    assert len(queue) == 3
    assert [c.name for c in queue.pending] == ["m00", "m01", "m02"]
    assert queue.scored == []


def test_output_root_is_relative_to_you_not_to_the_manifest(study, tmp_path,
                                                            monkeypatch):
    """A relative output_root belongs where you typed it.

    Resolving it against the manifest instead nests the results inside the
    manifest's folder -- and with a relative path on both sides you get the
    directory name twice: results/study/results/study/m00.
    """
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    queue = nyx.open_queue(study["manifest"], params=study["params"],
                           output_root="results")

    assert queue.configs[0].output_dir == str(work / "results" / "m00")
    assert queue.state_path == str(work / "results" / "queue_state.json")


def test_summary_names_what_is_next(study):
    text = open_it(study).summary()

    assert "3 recordings" in text
    assert "next: m00" in text


# ---------------------------------------------------------------------------
# One at a time
# ---------------------------------------------------------------------------


def test_next_writes_the_config_the_notebook_reads(study):
    """The whole point: the notebook is never edited between recordings."""
    queue = open_it(study)
    config = queue.next()

    assert config.name == "m00"
    assert os.path.exists(study["config"])

    reloaded = nyx.load_config(study["config"])
    assert reloaded.name == "m00"
    recording, _reference = reloaded.load()
    assert recording.duration > 0


def test_marking_done_advances(study):
    queue = open_it(study)

    assert queue.next().name == "m00"
    queue.done()
    assert queue.next().name == "m01"

    assert [c.name for c in queue.scored] == ["m00"]


def test_the_config_follows_the_queue(study):
    queue = open_it(study)
    queue.next()
    queue.done()
    queue.next()

    assert nyx.load_config(study["config"]).name == "m01"


def test_skipping_records_the_reason(study):
    queue = open_it(study)
    queue.next()
    queue.skip("EMG disconnected")

    assert [c.name for c in queue.skipped] == ["m00"]
    assert "EMG disconnected" in queue.summary()


def test_skipping_without_a_reason_is_refused(study):
    """A gap with no explanation is the thing this is meant to prevent."""
    queue = open_it(study)
    queue.next()

    with pytest.raises(ValueError, match="needs a reason"):
        queue.skip("")


def test_a_skipped_recording_is_not_handed_out_again(study):
    queue = open_it(study)
    queue.next()
    queue.skip("bad channel")

    assert queue.next().name == "m01"


def test_marking_with_nothing_open_says_so(study):
    with pytest.raises(RuntimeError, match="No recording is open"):
        open_it(study).done()


def test_an_unknown_status_is_refused(study):
    with pytest.raises(ValueError, match="Unknown status"):
        open_it(study).mark("m00", "finished-ish")


# ---------------------------------------------------------------------------
# Looping
# ---------------------------------------------------------------------------


def test_iterating_walks_the_whole_queue(study):
    queue = open_it(study)

    seen = []
    for config in queue:
        seen.append(config.name)
        queue.done()

    assert seen == ["m00", "m01", "m02"]
    assert queue.pending == []


def test_an_unmarked_recording_stops_the_loop_rather_than_repeating(study):
    """Otherwise the same recording comes round forever."""
    queue = open_it(study)

    with pytest.raises(RuntimeError, match="still pending"):
        for _config in queue:
            pass  # forgot to mark it


def test_a_loop_can_mix_done_and_skip(study):
    queue = open_it(study)

    for i, _config in enumerate(queue):
        queue.skip("odd one out") if i == 1 else queue.done()

    assert [c.name for c in queue.scored] == ["m00", "m02"]
    assert [c.name for c in queue.skipped] == ["m01"]


# ---------------------------------------------------------------------------
# Picking up where you left off
# ---------------------------------------------------------------------------


def test_progress_survives_reopening(study):
    queue = open_it(study)
    queue.next()
    queue.done()
    queue.next()
    queue.skip("noisy")

    reopened = open_it(study)
    assert [c.name for c in reopened.pending] == ["m02"]
    assert [c.name for c in reopened.skipped] == ["m01"]


def test_a_written_result_counts_even_without_the_state_file(study):
    """So the queue is still right after the state file is deleted, or when
    someone else scored part of the list elsewhere."""
    queue = open_it(study)
    queue.next()
    queue.done()

    os.remove(queue.state_path)
    # Fake the output a real run would have left.
    scored = os.path.join(study["output"], "m00")
    os.makedirs(scored, exist_ok=True)
    with open(os.path.join(scored, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"params": {}}, fh)

    reopened = open_it(study)
    assert [c.name for c in reopened.scored] == ["m00"]
    assert [c.name for c in reopened.pending] == ["m01", "m02"]


def test_reset_puts_one_recording_back(study):
    queue = open_it(study)
    queue.next()
    queue.skip("mistake")

    queue.reset("m00")
    assert [c.name for c in queue.pending] == ["m00", "m01", "m02"]


def test_reset_clears_everything(study):
    queue = open_it(study)
    for _config in queue:
        queue.done()

    queue.reset()
    assert len(queue.pending) == 3


# ---------------------------------------------------------------------------
# Headless
# ---------------------------------------------------------------------------


def test_run_all_scores_everything(study):
    queue = open_it(study)
    rows = queue.run_all(verbose=False)

    assert len(rows) == 3
    assert all(row["status"] == DONE for row in rows)
    assert queue.pending == []
    for config in queue.configs:
        assert os.path.exists(os.path.join(config.output_dir, "run.json"))
        assert os.path.exists(os.path.join(config.output_dir, "hypnogram.csv"))


def test_run_all_resumes_rather_than_redoing(study):
    queue = open_it(study)
    queue.next()
    queue.skip("not this one")

    rows = queue.run_all(verbose=False)
    assert [row["name"] for row in rows] == ["m01", "m02"]


def test_one_bad_recording_does_not_stop_the_batch(study):
    """Forty good recordings should not be lost to one unreadable file."""
    os.remove(study["tmp"] / "data" / "m01.npz")

    queue = open_it(study)
    rows = queue.run_all(verbose=False)

    statuses = {row["name"]: row["status"] for row in rows}
    assert statuses == {"m00": DONE, "m01": FAILED, "m02": DONE}
    assert "m01" in [c.name for c in queue.failed]


def test_the_failure_reason_is_kept(study):
    os.remove(study["tmp"] / "data" / "m01.npz")

    queue = open_it(study)
    queue.run_all(verbose=False)

    assert "FileNotFound" in queue.entry("m01").note
    assert "m01" in queue.summary()


def test_stop_on_error_raises(study):
    os.remove(study["tmp"] / "data" / "m01.npz")

    queue = open_it(study)
    with pytest.raises(FileNotFoundError):
        queue.run_all(verbose=False, stop_on_error=True)


def test_run_all_reports_agreement_when_there_is_a_reference(study, synthetic, tmp_path):
    import pandas as pd

    reference_path = tmp_path / "m00_scores.csv"
    pd.DataFrame(synthetic[3]).to_csv(reference_path, index=False)

    rows = [{"recording_path": str(tmp_path / "data" / "m00.npz"), "name": "m00",
             "annotation_path": str(reference_path),
             "annotation_format": "interval_csv"}]
    manifest = tmp_path / "one.csv"
    nyx.write_manifest(str(manifest), rows)

    queue = nyx.open_queue(str(manifest), params=study["params"],
                           output_root=str(tmp_path / "out"),
                           config_path=str(tmp_path / "c.json"))
    result = queue.run_all(verbose=False)

    assert result[0]["mf1"] is not None
    assert 0.0 <= result[0]["mf1"] <= 1.0
