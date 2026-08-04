"""End-to-end pipeline tests on synthetic signals with a known hypnogram."""

from __future__ import annotations

import numpy as np
import pytest

import nyx
from nyx.pipeline import DEFAULT_CLUSTERING, resolve_clustering_params

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_recording_reports_its_shape(synthetic_recording, synthetic):
    _eeg, _emg, fs, hypnogram = synthetic
    expected = float(hypnogram["duration"].sum())

    assert synthetic_recording.fs == fs
    assert synthetic_recording.duration == pytest.approx(expected, rel=1e-3)
    assert "synthetic" in synthetic_recording.describe()


def test_time_slice_shortens_the_recording(synthetic_recording):
    sliced = synthetic_recording.time_slice(100, 400)
    assert sliced.duration == pytest.approx(300, rel=1e-2)


def test_time_slice_rejects_an_empty_window(synthetic_recording):
    with pytest.raises(ValueError, match="Empty analysis window"):
        synthetic_recording.time_slice(500, 100)


def test_missing_recording_names_the_path():
    with pytest.raises(FileNotFoundError, match="nope.edf"):
        nyx.read_recording("nope.edf")


def test_unknown_extension_asks_for_an_explicit_format(tmp_path):
    path = tmp_path / "x.weird"
    path.write_text("data")
    with pytest.raises(ValueError, match="Pass format="):
        nyx.read_recording(str(path))


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


def test_emg_power_is_scaled_to_unit_range(synthetic_recording, params):
    emg = nyx.compute_emg_features(synthetic_recording, params)

    assert emg.power.ndim == 1
    assert emg.power.min() == pytest.approx(0.0, abs=1e-6)
    assert emg.power.max() == pytest.approx(1.0, abs=1e-6)
    assert emg.fs < synthetic_recording.fs  # one value per spectrogram bin


def test_wake_sleep_split_finds_both_states(synthetic_recording, params):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)

    assert wake_sleep.threshold_source == "auto"
    assert set(wake_sleep.hypnogram["label"]) >= {"WAKE", "SLEEP"}


def test_manual_threshold_is_recorded_as_manual(synthetic_recording, params):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg, threshold=0.5)

    assert wake_sleep.threshold == 0.5
    assert wake_sleep.threshold_source == "manual"
    assert wake_sleep.thresholds == [0.0, 0.5]


def test_threshold_search_rejects_an_impossible_nosignal_cut(synthetic_recording, params):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    with pytest.raises(ValueError, match="nosignal_threshold"):
        nyx.find_wake_sleep_threshold(emg, nosignal_threshold=2.0)


def test_pca_is_fitted_on_sleep_only(synthetic_recording, params):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    pca = nyx.compute_sleep_pca(synthetic_recording, params, wake_sleep)

    assert pca.scores.shape[1] == params["scoring"]["pc_components"]
    # Fewer rows than time bins, because wake epochs are excluded...
    assert pca.scores.shape[0] < pca.signal.shape[0]
    # ...and those excluded bins are NaN in the full-length signal.
    assert np.isnan(pca.signal).any()


def test_cluster_sleep_rejects_components_the_pca_does_not_have(
    synthetic_recording, params
):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    pca = nyx.compute_sleep_pca(synthetic_recording, params, wake_sleep)

    with pytest.raises(ValueError, match="needs at least"):
        nyx.cluster_sleep(pca, wake_sleep, clustering={"pcs_to_use": [0, 1, 99]})


def test_assign_stages_rejects_an_override_for_a_missing_cluster(
    synthetic_recording, params
):
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    pca = nyx.compute_sleep_pca(synthetic_recording, params, wake_sleep)
    clusters = nyx.cluster_sleep(pca, wake_sleep, clustering=params)

    with pytest.raises(KeyError, match="no such cluster"):
        nyx.assign_stages(clusters, pca, wake_sleep, overrides={99: "REM"})


# ---------------------------------------------------------------------------
# Reproducible cluster ids
# ---------------------------------------------------------------------------


def _cluster(recording, params, method="kmeans", n_clusters=3, **extra):
    emg = nyx.compute_emg_features(recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    pca = nyx.compute_sleep_pca(recording, params, wake_sleep)
    settings = {"method": method, "n_clusters": n_clusters, "pcs_to_use": [0, 1], **extra}
    return nyx.cluster_sleep(pca, wake_sleep, clustering=settings, emg=emg)


def test_cluster_ids_are_ordered_by_centroid(synthetic_recording, params):
    """Cluster ids must be a property of the data, not of how the algorithm
    happened to initialise -- otherwise a saved cluster_to_stage mapping cannot
    be replayed."""
    clusters = _cluster(synthetic_recording, params)

    first_pc = clusters.centers[:, 0]
    assert list(first_pc) == sorted(first_pc), (
        f"cluster centroids are not ordered: {first_pc}"
    )
    assert list(clusters.unique_labels) == list(range(len(clusters.centers)))


def test_cluster_ids_are_stable_across_runs(synthetic_recording, params):
    a = _cluster(synthetic_recording, params)
    b = _cluster(synthetic_recording, params)

    assert np.array_equal(a.labels, b.labels)
    assert np.allclose(a.centers, b.centers)


def test_canonicalisation_only_renumbers_it_does_not_repartition(
    synthetic_recording, params
):
    """Turning canonical ids off must give the same grouping of epochs, just
    with different id numbers."""
    canonical = _cluster(synthetic_recording, params, canonical_cluster_ids=True)
    raw = _cluster(synthetic_recording, params, canonical_cluster_ids=False)

    # Compare as partitions: the multiset of cluster sizes is unchanged.
    assert sorted(canonical.sizes().values()) == sorted(raw.sizes().values())
    # And every canonical cluster is exactly some raw cluster.
    raw_groups = {frozenset(np.flatnonzero(raw.labels == c)) for c in raw.unique_labels}
    for cid in canonical.unique_labels:
        assert frozenset(np.flatnonzero(canonical.labels == cid)) in raw_groups


# ---------------------------------------------------------------------------
# Clustering settings
# ---------------------------------------------------------------------------


def test_full_params_dict_without_clustering_section_uses_defaults():
    resolved = resolve_clustering_params({"EEG": {}, "EMG": {}, "clustering": {}})
    assert resolved == DEFAULT_CLUSTERING


def test_bare_section_is_applied_over_the_defaults():
    resolved = resolve_clustering_params({"method": "kmeans", "n_clusters": 3})
    assert resolved["method"] == "kmeans"
    assert resolved["n_clusters"] == 3
    assert resolved["pcs_to_use"] == DEFAULT_CLUSTERING["pcs_to_use"]


def test_legacy_clustering_keys_warn_rather_than_being_applied_silently():
    """Old params files used min_cluster_size for a different routine; applying
    it to the hdbscan settings would silently change results."""
    with pytest.warns(UserWarning, match="min_cluster_size"):
        resolved = resolve_clustering_params(
            {"EEG": {}, "EMG": {}, "clustering": {"min_cluster_size": 200, "nrem_weight": 3.0}}
        )
    assert resolved["hdbscan_min_cluster_size"] == DEFAULT_CLUSTERING["hdbscan_min_cluster_size"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_scores_without_a_reference(synthetic_recording, params):
    """The whole point: nyx must work with no manual scoring at all."""
    result = nyx.score_recording(synthetic_recording, params, verbose=False)

    assert result.agreement is None
    assert set(result.hypnogram["label"]) <= {"WAKE", "NREM", "REM", "NOSIGNAL"}

    durations = result.staging.stage_durations()
    for stage in ("WAKE", "NREM", "REM"):
        assert durations.get(stage, 0) > 0, f"{stage} was never scored"


def test_recovers_the_known_hypnogram(synthetic_recording, params, synthetic):
    _eeg, _emg, _fs, truth = synthetic
    result = nyx.score_recording(synthetic_recording, params, reference=truth, verbose=False)

    assert result.agreement is not None
    assert result.agreement.accuracy > 0.75, result.agreement.summary()
    assert result.agreement.kappa > 0.6, result.agreement.summary()


def test_evaluate_is_a_separate_optional_step(synthetic_recording, params, synthetic):
    """Scoring first, comparing later, must match doing both at once."""
    _eeg, _emg, _fs, truth = synthetic
    scored = nyx.score_recording(synthetic_recording, params, verbose=False)
    agreement = nyx.evaluate(scored.staging, truth, verbose=False)

    combined = nyx.score_recording(
        synthetic_recording, params, reference=truth, verbose=False
    )
    assert agreement.accuracy == pytest.approx(combined.agreement.accuracy)


def test_window_restricts_the_analysis(synthetic_recording, params):
    result = nyx.score_recording(synthetic_recording, params, window=(0, 1800), verbose=False)

    assert result.window == (0.0, 1800.0)
    assert result.total_duration > 1800.0
    assert result.recording.duration == pytest.approx(1800, rel=1e-2)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_results_writes_the_expected_files(synthetic_recording, params, tmp_path):
    result = nyx.score_recording(synthetic_recording, params, window=(0, 1800), verbose=False)
    out = tmp_path / "run"
    nyx.save_results(result, str(out), config={"note": "test"})

    written = {p.name for p in out.iterdir()}
    # run.json is the self-contained record: params inline plus every decision.
    assert {"run.json", "hypnogram.csv", "wake_sleep.csv",
            "emg_power.npy", "plots"} <= written


def test_saved_hypnogram_covers_the_whole_recording(synthetic_recording, params, tmp_path):
    """Epochs outside the analysis window must be written as NOSIGNAL, so the
    saved file always spans the full recording."""
    import pandas as pd

    result = nyx.score_recording(synthetic_recording, params, window=(0, 1800), verbose=False)
    out = tmp_path / "run"
    nyx.save_results(result, str(out))

    saved = pd.read_csv(out / "hypnogram.csv")
    span = float((saved["time"] + saved["duration"]).max())
    assert span == pytest.approx(result.total_duration, rel=1e-2)
    assert "NOSIGNAL" in set(saved["label"])
