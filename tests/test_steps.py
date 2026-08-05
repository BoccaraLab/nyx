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


# ---------------------------------------------------------------------------
# elliptic numbers its clusters by meaning, not by position
# ---------------------------------------------------------------------------


def test_elliptic_maps_the_dense_cluster_first(synthetic_recording, params, scored):
    """Cluster 0 is the bulk and cluster 1 the tail, whatever their centroids.

    This is the whole point of using elliptic for the human wake/sleep split:
    sleep is one dense blob and wake is the scattered minority. Ordering by
    centroid instead flips the mapping depending on which side of the first
    component the tail happens to sit -- which varies by recording, and on a
    dodh night put 90% of the recording into WAKE.
    """
    emg, wake_sleep = scored
    step = Step("wake_sleep", within="SLEEP", method="elliptic", use_emg=True,
                pcs_to_use=[0, 1], stage_order=["SLEEP", "WAKE"],
                options={"elliptic_contamination": 0.1,
                         "elliptic_support_fraction": 0.75})

    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step,
                       emg=emg)

    sizes = outcome.clusters.sizes()
    assert outcome.cluster_to_stage[0] == "SLEEP"
    assert outcome.cluster_to_stage[1] == "WAKE"
    # The tail is the contamination fraction, so it must be the smaller one.
    assert sizes[1] < sizes[0]


def test_elliptic_cluster_ids_are_not_renumbered(synthetic_recording, params, scored):
    """Canonicalising by centroid would throw away a stronger guarantee."""
    from nyx.pipeline import cluster_sleep, compute_sleep_pca

    emg, wake_sleep = scored
    pca = compute_sleep_pca(synthetic_recording, params, wake_sleep)
    clusters = cluster_sleep(
        pca, wake_sleep,
        clustering={"method": "elliptic", "pcs_to_use": [0, 1], "use_emg": True,
                    "elliptic_contamination": 0.1,
                    "canonical_cluster_ids": True},
        emg=emg,
    )

    sizes = clusters.sizes()
    assert sizes[0] > sizes[1], "cluster 0 must stay the dense component"


# ---------------------------------------------------------------------------
# Driven from the params
# ---------------------------------------------------------------------------


def test_score_recording_runs_the_steps_in_the_params(synthetic_recording):
    """Every shipped params file carries a `steps` list; it has to be honoured.

    This is what makes rodent substages and human five-stage scoring reachable
    through the ordinary entry point rather than by calling run_steps by hand.
    """
    import nyx

    params = {**nyx.demo_params()}
    params["steps"] = [
        params["steps"][0],
        params["steps"][1],
        {"name": "split_nrem", "within": "NREM", "method": "kmeans",
         "n_clusters": 2, "pcs_to_use": [0, 1],
         "stage_order": ["NREM2", "NREM3"]},
    ]

    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 verbose=False)

    labels = set(result.hypnogram["label"])
    assert {"NREM2", "NREM3"} <= labels
    assert "NREM" not in labels, "the third step should have consumed it"
    assert len(result.steps) == 2, "the emg_threshold step is not a clustering step"


def test_substages_collapse_back_to_the_ordinary_scoring(synthetic_recording):
    """A substage run must still be comparable against a 3-stage reference."""
    import nyx

    two_step = nyx.demo_params()
    three_step = {**two_step}
    three_step["steps"] = [
        *two_step["steps"],
        {"name": "split_nrem", "within": "NREM", "method": "kmeans",
         "n_clusters": 3, "pcs_to_use": [0, 1],
         "stage_order": ["TR", "NREM3", "NREM2"]},
    ]

    plain = nyx.score_recording(synthetic_recording, two_step, window=(0, 2400),
                                verbose=False)
    subs = nyx.score_recording(synthetic_recording, three_step, window=(0, 2400),
                               verbose=False)
    collapsed = nyx.collapse(subs.hypnogram, 3)

    assert {"TR", "NREM2", "NREM3"} <= set(subs.hypnogram["label"])
    assert set(collapsed["label"]) <= {"WAKE", "NREM", "REM", "NOSIGNAL"}

    def nrem_seconds(hypnogram):
        return sum(float(d) for d, l in zip(hypnogram["duration"], hypnogram["label"])
                   if l == "NREM")

    # Not identical: the extra step brings its own outlier rejection, so a few
    # epochs become NOSIGNAL that the two-step run scored.
    assert nrem_seconds(collapsed) == pytest.approx(
        nrem_seconds(plain.hypnogram), rel=0.02
    )


def test_the_params_emg_threshold_is_used(synthetic_recording):
    """human.json sets a deliberately permissive threshold so everything lands
    in SLEEP and a clustering step splits wake off. Auto-fitting instead would
    quietly do something else."""
    import nyx

    params = {**nyx.demo_params()}
    params["steps"] = [
        {"name": "wake_sleep", "method": "emg_threshold", "threshold": 1.5},
        params["steps"][1],
    ]

    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 verbose=False)

    assert result.wake_sleep.threshold == 1.5
    assert result.wake_sleep.threshold_source == "manual"
    # Above the top of the min-max scaled power, so nothing is wake.
    assert "WAKE" not in set(result.wake_sleep.hypnogram["label"])


def test_an_explicit_threshold_still_wins_over_the_params(synthetic_recording):
    import nyx

    params = {**nyx.demo_params()}
    params["steps"] = [
        {"name": "wake_sleep", "method": "emg_threshold", "threshold": 1.5},
        params["steps"][1],
    ]

    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 emg_threshold=0.4, verbose=False)

    assert result.wake_sleep.threshold == 0.4


def test_the_emg_threshold_step_is_not_run_twice(synthetic_recording):
    """classify_wake_sleep has already done it by the time the steps are read."""
    import nyx

    result = nyx.score_recording(synthetic_recording, nyx.demo_params(),
                                 window=(0, 2400), verbose=False)

    assert [outcome.step.name for outcome in result.steps] == ["split_sleep"]


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


# ---------------------------------------------------------------------------
# Renaming clusters after the fact
# ---------------------------------------------------------------------------
#
# Naming clusters is a decision you make after seeing the per-cluster spectra,
# and getting it wrong first time is normal. Editing the Step and running it
# again would recompute the PCA and the clustering to arrive at exactly the same
# clusters -- minutes on a night of data, for a relabelling.


@pytest.fixture
def one_step(synthetic_recording, params, scored):
    emg, wake_sleep = scored
    step = Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
                pcs_to_use=[0, 1], stage_order=["REM", "NREM"])
    outcome = run_step(synthetic_recording, params, wake_sleep.hypnogram, step,
                       emg=emg)
    return wake_sleep, outcome


def test_renaming_by_position(one_step):
    from nyx.steps import rename_clusters

    _ws, outcome = one_step
    flipped = rename_clusters(outcome, stage_order=["NREM", "REM"])

    assert outcome.cluster_to_stage == {0: "REM", 1: "NREM"}
    assert flipped.cluster_to_stage == {0: "NREM", 1: "REM"}


def test_renaming_by_cluster_id(one_step):
    from nyx.steps import rename_clusters

    _ws, outcome = one_step
    named = rename_clusters(outcome, cluster_to_stage={0: "NREM", 1: "REM"})

    assert named.cluster_to_stage == {0: "NREM", 1: "REM"}


def test_renaming_does_not_recluster(one_step):
    """The whole point: the expensive part is reused, not repeated."""
    from nyx.steps import rename_clusters

    _ws, outcome = one_step
    renamed = rename_clusters(outcome, stage_order=["NREM", "REM"])

    assert renamed.clusters is outcome.clusters
    assert renamed.pca is outcome.pca


def test_renaming_leaves_the_original_alone(one_step):
    from nyx.steps import rename_clusters

    _ws, outcome = one_step
    before = dict(outcome.cluster_to_stage)
    labels_before = outcome.hypnogram["label"].copy()

    rename_clusters(outcome, stage_order=["NREM", "REM"])

    assert outcome.cluster_to_stage == before
    assert np.array_equal(outcome.hypnogram["label"], labels_before)


def test_renaming_rebuilds_the_hypnogram(one_step):
    """Not just the mapping -- the labelled output has to follow."""
    from nyx.steps import rename_clusters

    _ws, outcome = one_step

    def seconds(outcome, stage):
        return sum(float(d) for d, l in zip(outcome.hypnogram["duration"],
                                            outcome.hypnogram["label"]) if l == stage)

    flipped = rename_clusters(outcome, stage_order=["NREM", "REM"])

    assert seconds(flipped, "REM") == pytest.approx(seconds(outcome, "NREM"))
    assert seconds(flipped, "NREM") == pytest.approx(seconds(outcome, "REM"))


def test_renaming_matches_running_the_step_again(synthetic_recording, params,
                                                 scored, one_step):
    """It must be indistinguishable from the slow route, or it is a second
    implementation of the naming rule."""
    from nyx.steps import rename_clusters

    emg, wake_sleep = scored
    _ws, outcome = one_step

    renamed = rename_clusters(outcome, stage_order=["NREM", "REM"])
    rerun = run_step(
        synthetic_recording, params, wake_sleep.hypnogram,
        Step("split_sleep", within="SLEEP", method="kmeans", n_clusters=2,
             pcs_to_use=[0, 1], stage_order=["NREM", "REM"]),
        emg=emg,
    )

    assert renamed.cluster_to_stage == rerun.cluster_to_stage
    assert np.array_equal(renamed.labels, rerun.labels)
    assert np.array_equal(renamed.hypnogram["label"], rerun.hypnogram["label"])


def test_refinements_can_be_added_without_reclustering(one_step):
    from nyx.steps import Refinement, rename_clusters

    _ws, outcome = one_step
    refined = rename_clusters(
        outcome,
        stage_order=["REM", "NREM"],
        refinements=[Refinement(split_stage="NREM", pc=0, threshold=0.0,
                                high="NREM3", low="NREM2")],
    )

    assert {"NREM2", "NREM3"} & set(refined.hypnogram["label"])
    assert refined.clusters is outcome.clusters


def test_an_outcome_built_by_hand_says_why_it_cannot_be_renamed(one_step):
    from dataclasses import replace

    from nyx.steps import rename_clusters

    _ws, outcome = one_step
    orphan = replace(outcome, input_hypnogram=None)

    with pytest.raises(ValueError, match="cannot be relabelled"):
        rename_clusters(orphan)
