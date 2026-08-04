"""The step engine: sequential clustering steps, each scoped to a label."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.steps import Refinement, Step, epoch_labels, run_step, run_steps


@pytest.fixture(scope="module")
def scored(synthetic_recording, params):
    """A wake/sleep split to run further steps on top of."""
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    return emg, wake_sleep


# ---------------------------------------------------------------------------
# epoch_labels
# ---------------------------------------------------------------------------


def test_epoch_labels_expands_intervals():
    hypnogram = {
        "time": np.array([0.0, 4.0]),
        "duration": np.array([4.0, 4.0]),
        "label": np.array(["WAKE", "SLEEP"]),
    }
    labels = epoch_labels(hypnogram, fs=1.0, n_epochs=8)
    assert list(labels) == ["WAKE"] * 4 + ["SLEEP"] * 4


def test_epoch_labels_fills_gaps_with_nosignal():
    hypnogram = {
        "time": np.array([2.0]),
        "duration": np.array([2.0]),
        "label": np.array(["SLEEP"]),
    }
    labels = epoch_labels(hypnogram, fs=1.0, n_epochs=5)
    assert list(labels) == ["NOSIGNAL", "NOSIGNAL", "SLEEP", "SLEEP", "NOSIGNAL"]


def test_epoch_labels_keeps_long_stage_names():
    """Human vocabularies (NREM1/2/3) must not be truncated by the array dtype."""
    hypnogram = {
        "time": np.array([0.0]),
        "duration": np.array([2.0]),
        "label": np.array(["NREM3"]),
    }
    assert list(epoch_labels(hypnogram, 1.0, 2)) == ["NREM3", "NREM3"]


# ---------------------------------------------------------------------------
# A single step
# ---------------------------------------------------------------------------


def test_step_subdivides_only_its_own_scope(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    step = Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
                pcs_to_use=[0, 1], stage_order=["REM", "NREM"])

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)

    labels = set(outcome.hypnogram["label"].tolist())
    # SLEEP has been replaced by its subdivisions...
    assert "SLEEP" not in labels
    assert {"NREM", "REM"} <= labels
    # ...and WAKE, which the step did not touch, survives untouched.
    assert "WAKE" in labels


def test_step_summary_reports_the_mapping_and_sizes(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    step = Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
                stage_order=["REM", "NREM"])

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)
    summary = outcome.summary()

    assert "split_sleep" in summary
    assert "NREM" in summary and "REM" in summary
    assert sum(outcome.clusters.sizes().values()) > 0


def test_mapping_every_cluster_to_one_stage_makes_a_step_a_no_op(
    synthetic_recording, params, scored
):
    """How you decide, after looking, that a split was not worth keeping."""
    emg, wake_sleep = scored
    step = Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
                cluster_to_stage={0: "SLEEP", 1: "SLEEP"})

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)

    assert set(outcome.hypnogram["label"].tolist()) <= {"WAKE", "SLEEP", "NOSIGNAL"}


def test_unnamed_clusters_are_the_default(synthetic_recording, params, scored):
    """With no stage names given, clusters keep placeholder labels rather than
    being guessed at -- the honest default for an unfamiliar species."""
    emg, wake_sleep = scored
    step = Step("explore", within="SLEEP", method="kmeans", n_clusters=2)

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)

    assert any(lbl.startswith("SLEEP_C") for lbl in outcome.hypnogram["label"])


# ---------------------------------------------------------------------------
# Several steps in sequence
# ---------------------------------------------------------------------------


def test_steps_compose(synthetic_recording, params, scored):
    """A second step refines what the first produced, not the original labels."""
    emg, wake_sleep = scored
    steps = [
        Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
             stage_order=["REM", "NREM"]),
        Step("split_nrem", within="NREM", method="kmeans", n_clusters=2,
             stage_order=["NREM2", "NREM3"]),
    ]

    outcomes = run_steps(
        synthetic_recording, params, wake_sleep.hypnogram, steps, emg=emg, verbose=False
    )

    assert len(outcomes) == 2
    final = set(outcomes[-1].hypnogram["label"].tolist())
    assert "NREM" not in final          # consumed by the second step
    assert {"NREM2", "NREM3"} <= final  # ...into these
    assert "REM" in final               # untouched by the second step
    assert "WAKE" in final              # untouched by both


def test_total_time_is_conserved_across_steps(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    before = float(np.sum(wake_sleep.hypnogram["duration"]))

    outcomes = run_steps(
        synthetic_recording,
        params,
        wake_sleep.hypnogram,
        [Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2)],
        emg=emg,
        verbose=False,
    )
    after = float(np.sum(outcomes[-1].hypnogram["duration"]))

    assert after == pytest.approx(before, rel=0.02)


def test_step_on_a_missing_label_is_reported_clearly(
    synthetic_recording, params, scored
):
    emg, wake_sleep = scored
    step = Step("nope", within="N3", method="kmeans", n_clusters=2)

    with pytest.raises(ValueError, match="No epochs labelled 'N3'"):
        run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)


# ---------------------------------------------------------------------------
# Refinements
# ---------------------------------------------------------------------------


def test_refinement_splits_a_stage_by_a_pc(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    step = Step(
        "split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
        pcs_to_use=[0, 1], stage_order=["REM", "NREM"],
        refinements=[
            Refinement(split_stage="NREM", pc=0, threshold=0.0,
                       high="NREM2", low="NREM3")
        ],
    )

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)

    labels = set(outcome.hypnogram["label"].tolist())
    assert "NREM" not in labels
    assert labels & {"NREM2", "NREM3"}
    assert "REM" in labels  # the refinement must not touch other stages


def test_refinement_on_a_missing_pc_is_reported(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    step = Step(
        "split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
        stage_order=["REM", "NREM"],
        refinements=[Refinement("NREM", pc=99, threshold=0.0, high="A", low="B")],
    )

    with pytest.raises(ValueError, match="PC99"):
        run_step(synthetic_recording, params, wake_sleep.hypnogram, step, emg=emg)
