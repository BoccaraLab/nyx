"""Summary figures, and the MF1 metric they report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nyx
import nyx.report as report
from nyx.report import plot_confusion, plot_summary, save_report

# Read from report rather than listed here, so a panel added to the library is
# covered by these tests without anyone remembering to add it.
PANELS = list(nyx.report.AX_PANELS)


@pytest.fixture(scope="module")
def scored(synthetic_recording, params):
    _eeg = None
    return nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                               verbose=False)


@pytest.fixture(scope="module")
def scored_with_reference(synthetic_recording, params, synthetic):
    _e, _m, _fs, truth = synthetic
    return nyx.score_recording(synthetic_recording, params, reference=truth,
                               verbose=False)


# ---------------------------------------------------------------------------
# MF1
# ---------------------------------------------------------------------------


def test_mf1_is_the_unweighted_mean_of_per_stage_f1(scored_with_reference):
    agreement = scored_with_reference.agreement
    present = agreement.per_stage["support"] > 0
    expected = agreement.per_stage.loc[present, "f1-score"].mean()

    assert agreement.mf1 == pytest.approx(expected)


def test_mf1_ignores_stages_absent_from_the_reference(synthetic_recording, params):
    """A recording with no REM should not be marked down for missing a stage
    that was never there."""
    truth = {
        "time": np.array([0.0, 1200.0]),
        "duration": np.array([1200.0, 1200.0]),
        "label": np.array(["WAKE", "NREM"]),
    }
    agreement = nyx.evaluate(
        nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                            verbose=False).staging,
        truth, verbose=False,
    )
    # REM carries no support, so it must not enter the average.
    assert agreement.per_stage.loc["REM", "support"] == 0
    present = agreement.per_stage["support"] > 0
    assert agreement.mf1 == pytest.approx(
        agreement.per_stage.loc[present, "f1-score"].mean()
    )


def test_summary_reports_mf1_and_not_a_weighted_average(scored_with_reference):
    text = scored_with_reference.agreement.summary()

    assert "MF1" in text
    assert "Macro average" in text
    assert "eighted average" not in text


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("draw", PANELS, ids=lambda f: f.__name__)
def test_panel_draws(scored, draw):
    fig, ax = plt.subplots()
    try:
        result = draw(scored, ax=ax)
        assert result is ax
        assert ax.get_title() or ax.get_xlabel()
    finally:
        plt.close(fig)


def test_panels_accept_ax_positionally(scored):
    """save_report calls them uniformly, so the second argument must be ax."""
    for draw in PANELS:
        fig, ax = plt.subplots()
        try:
            assert draw(scored, ax) is ax
        finally:
            plt.close(fig)


def test_confusion_needs_a_reference(scored):
    with pytest.raises(ValueError, match="no reference scoring"):
        plot_confusion(scored)


def test_confusion_draws_when_a_reference_exists(scored_with_reference):
    fig, ax = plt.subplots()
    try:
        plot_confusion(scored_with_reference, ax=ax)
        assert "MF1" in ax.get_title()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary and saving
# ---------------------------------------------------------------------------


def test_summary_works_without_a_reference(scored):
    fig = plot_summary(scored)
    try:
        assert len(fig.axes) > 4
    finally:
        plt.close(fig)


def test_save_report_writes_every_panel(scored_with_reference, tmp_path):
    plots = save_report(scored_with_reference, str(tmp_path))

    written = {p.name for p in (tmp_path / "plots").iterdir()}
    assert {"summary.png", "emg_threshold.png", "pca_components.png",
            "clusters.png", "psd_per_cluster.png", "hypnogram.png",
            "confusion_matrix.png"} <= written
    assert plots.endswith("plots")


def test_save_results_writes_figures(synthetic_recording, params, tmp_path):
    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 verbose=False)
    nyx.save_results(result, str(tmp_path / "run"))

    assert (tmp_path / "run" / "plots" / "summary.png").exists()


def test_plots_can_be_turned_off(synthetic_recording, params, tmp_path):
    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 verbose=False)
    nyx.save_results(result, str(tmp_path / "run"), plots=False)

    assert not (tmp_path / "run" / "plots" / "summary.png").exists()


def test_a_failing_panel_does_not_lose_the_run(scored, tmp_path, monkeypatch):
    """Scoring is expensive; a broken figure must never discard it."""
    import nyx.report as report

    def boom(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(report, "plot_clusters", boom)
    with pytest.warns(UserWarning, match="clusters"):
        save_report(scored, str(tmp_path))

    # The others still got written.
    assert (tmp_path / "plots" / "summary.png").exists()


# ---------------------------------------------------------------------------
# The panels take a piece, not only a finished run
# ---------------------------------------------------------------------------
#
# This is what a step-by-step notebook needs: partway through a run there is no
# ScoringResult yet, only the pieces. Without it the notebooks have to
# reimplement every figure inline.


@pytest.fixture(scope="module")
def pieces(synthetic_recording, params):
    """The intermediate objects, as a notebook has them between steps."""
    emg = nyx.compute_emg_features(synthetic_recording, params)
    wake_sleep = nyx.classify_wake_sleep(emg)
    pca = nyx.compute_sleep_pca(synthetic_recording, params, wake_sleep)
    clusters = nyx.cluster_sleep(pca, wake_sleep, clustering=params, emg=emg)
    staging = nyx.assign_stages(clusters, pca, wake_sleep)
    return emg, wake_sleep, pca, clusters, staging


def test_emg_panels_take_the_features_alone(pieces):
    emg, wake_sleep, *_ = pieces

    report.plot_emg_threshold(emg, threshold=0.5)
    report.plot_emg_threshold(emg, wake_sleep=wake_sleep)
    report.plot_emg_power(emg, threshold=0.5)
    report.plot_emg_check(emg, wake_sleep=wake_sleep)


def test_a_threshold_must_come_from_somewhere(pieces):
    """Silently drawing no line would look like a threshold of zero."""
    emg = pieces[0]
    with pytest.raises(ValueError, match="threshold"):
        report.plot_emg_threshold(emg)


def test_pca_panels_take_the_pca_alone(pieces):
    pca = pieces[2]

    report.plot_pca_components(pca)
    figure = report.plot_pca_grid(pca, n=4)
    assert len([ax for ax in figure.axes if ax.get_visible()]) == 4


def test_cluster_panels_work_before_the_stages_are_named(pieces):
    """The state you are in when deciding what to call them."""
    _emg, _ws, pca, clusters, _staging = pieces

    ax = report.plot_clusters(clusters)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    # The cluster entries are bare ids; "outlier -> NOSIGNAL" is not one of them
    # and does carry an arrow.
    cluster_labels = [label for label in labels if not label.startswith("outlier")]
    assert cluster_labels
    assert not any("->" in label for label in cluster_labels), "no mapping was given"

    report.plot_psd_per_cluster(pca, clusters=clusters)
    report.plot_cluster_check(pca, clusters=clusters)


def test_a_given_mapping_is_shown(pieces):
    _emg, _ws, pca, clusters, staging = pieces

    ax = report.plot_clusters(clusters, stages=staging.cluster_to_stage)
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "->" in labels


def test_psd_without_clusters_says_so(pieces):
    with pytest.raises(ValueError, match="clusters"):
        report.plot_psd_per_cluster(pieces[2])


def test_wake_sleep_panel_takes_the_split_alone(pieces):
    report.plot_wake_sleep(pieces[1])


def test_the_same_functions_still_take_a_whole_result(scored):
    """The finished-run form must keep working, positionally as save_report
    calls it."""
    import matplotlib.pyplot as plt

    for draw in (report.plot_emg_threshold, report.plot_pca_components,
                 report.plot_clusters, report.plot_psd_per_cluster,
                 report.plot_wake_sleep):
        draw(scored, plt.subplots()[1])

    report.plot_emg_check(scored)
    report.plot_cluster_check(scored)
    report.plot_pca_grid(scored)


# ---------------------------------------------------------------------------
# A reference scoring is normalised before it is drawn
# ---------------------------------------------------------------------------


def _hyp(*bouts):
    return {
        "time": np.cumsum([0.0] + [float(d) for _, d in bouts][:-1]),
        "duration": np.asarray([float(d) for _, d in bouts]),
        "label": np.asarray([label for label, _ in bouts], dtype="U"),
    }


def test_foreign_spellings_become_nyx_stages():
    """The Oxford benchmark writes 'awake' and 'non-REM'; somnotate agrees."""
    from nyx.metrics import normalise_labels

    result = normalise_labels(_hyp(("awake", 30), ("non-REM", 60), ("REM", 30)))

    assert list(result["label"]) == ["WAKE", "NREM", "REM"]
    assert list(result["duration"]) == [30.0, 60.0, 30.0]


def test_undefined_is_not_silently_called_nrem():
    """It used to fall through to the NREM default, inventing sleep where the
    scorer had explicitly declined to name a stage."""
    from nyx.metrics import normalise_labels

    result = normalise_labels(_hyp(("awake", 30), ("undefined", 10)))

    assert list(result["label"]) == ["WAKE", "UNCLASSIFIED"]


def test_no_signal_stays_distinct_from_unclassified():
    """Different things: no usable signal, versus signal nobody could name."""
    from nyx.metrics import normalise_labels

    result = normalise_labels(_hyp(("NOSIGNAL", 10), ("undefined", 10)))

    assert list(result["label"]) == ["NOSIGNAL", "UNCLASSIFIED"]


def test_normalising_merges_what_it_makes_adjacent():
    from nyx.metrics import normalise_labels

    result = normalise_labels(_hyp(("awake", 30), ("WAKE", 30), ("W", 30)))

    assert list(result["label"]) == ["WAKE"]
    assert list(result["duration"]) == [90.0]


def test_both_hypnograms_share_their_rows(synthetic_recording, params):
    """A reference drawn in its own vocabulary gets no colour from the stage
    palette, and a separate row for every spelling of the same stage."""
    import nyx.report as report

    foreign = _hyp(("awake", 900), ("non-REM", 900), ("REM", 600))
    result = nyx.score_recording(synthetic_recording, params, window=(0, 2400),
                                 reference=foreign, verbose=False)

    figure = report.plot_scoring_overview(result)
    rows = {
        text.get_text()
        for ax in figure.axes for text in ax.get_yticklabels()
        if text.get_text()
    }
    assert "awake" not in rows and "non-REM" not in rows
    assert {"WAKE", "NREM", "REM"} <= rows


# ---------------------------------------------------------------------------
# The ordering stage_order maps onto
# ---------------------------------------------------------------------------


def test_ordering_table_is_in_naming_order(pieces):
    """`stage_order` names clusters by position in this table, so the table has
    to be in that order and say which position each cluster holds."""
    from nyx.pipeline import cluster_order

    clusters = pieces[3]
    table = nyx.cluster_ordering(clusters)

    assert list(table.index) == list(range(len(clusters.unique_labels)))
    assert list(table["cluster"]) == cluster_order(clusters)


def test_ordering_table_carries_the_centroids(pieces):
    """The point of it: two centroids a hundredth apart look identical in the
    scatter plot, and the order between them decides two stage names."""
    clusters = pieces[3]
    table = nyx.cluster_ordering(clusters)

    assert "PC1" in table.columns
    assert table["epochs"].sum() > 0
    for position, cid in enumerate(table["cluster"]):
        assert table.loc[position, "PC1"] == pytest.approx(
            clusters.centers[cid, 0], abs=1e-4
        )
    # PC1 non-decreasing, since that is what the order is by.
    assert list(table["PC1"]) == sorted(table["PC1"])


def test_ordering_table_shows_stages_when_they_are_known(pieces):
    clusters, staging = pieces[3], pieces[4]

    assert "stage" not in nyx.cluster_ordering(clusters).columns
    named = nyx.cluster_ordering(clusters, staging.cluster_to_stage)
    assert set(named["stage"]) == set(staging.cluster_to_stage.values())


def test_the_plot_shows_the_same_order_as_the_table(pieces):
    """They have to agree, or the table cannot be used to read the plot."""
    import nyx.report as report

    clusters = pieces[3]
    ax = report.plot_clusters(clusters)

    legend = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    for position in range(len(clusters.unique_labels)):
        assert f"#{position}" in legend

    # Centroid annotations carry the rank and the coordinates.
    annotations = " ".join(t.get_text() for t in ax.texts)
    assert "#0" in annotations
    assert "(" in annotations and ")" in annotations


def test_elliptic_orders_by_id_in_the_table_too(synthetic_recording, params, pieces):
    """assign_stages orders elliptic by id, not centroid; the table must agree
    or it would describe a mapping that is not the one applied."""
    from nyx.pipeline import cluster_order

    emg, wake_sleep, pca, *_ = pieces
    clusters = nyx.cluster_sleep(
        pca, wake_sleep,
        clustering={"method": "elliptic", "pcs_to_use": [0, 1], "use_emg": True,
                    "elliptic_contamination": 0.1},
        emg=emg,
    )
    assert cluster_order(clusters) == sorted(int(c) for c in clusters.unique_labels)
    assert list(nyx.cluster_ordering(clusters)["cluster"]) == cluster_order(clusters)


# ---------------------------------------------------------------------------
# The dimensions the scatter plot does not show
# ---------------------------------------------------------------------------


def test_cluster_features_draws_one_panel_per_other_feature(pieces):
    """The clustering used more dimensions than the scatter shows; this is where
    you see which one actually drew the boundary."""
    import nyx.report as report

    emg, wake_sleep, pca, *_ = pieces
    clusters = nyx.cluster_sleep(
        pca, wake_sleep,
        clustering={"method": "kmeans", "n_clusters": 2,
                    "pcs_to_use": [0, 1, 2, 3], "use_emg": True},
        emg=emg,
    )
    titles = [ax.get_title() for ax in report.plot_cluster_features(clusters).axes]

    assert any("Clusters" in t for t in titles)
    for name in ("PC3", "PC4", "EMG power"):
        assert any(f"coloured by {name}" == t for t in titles), name

    # PC1 and PC2 are the axes; colouring by them would restate the plot.
    assert not any("coloured by PC1" == t or "coloured by PC2" == t for t in titles)


def test_cluster_features_says_so_when_the_axes_are_the_whole_space(pieces):
    """Two components and no EMG: nothing is left to colour by, and a lone panel
    would look like something failed to draw."""
    import nyx.report as report

    emg, wake_sleep, pca, *_ = pieces
    clusters = nyx.cluster_sleep(
        pca, wake_sleep,
        clustering={"method": "kmeans", "n_clusters": 2, "pcs_to_use": [0, 1],
                    "use_emg": False},
        emg=emg,
    )
    titles = [ax.get_title() for ax in report.plot_cluster_features(clusters).axes]

    assert any("only features used" in t for t in titles)
    assert not any("coloured by" in t for t in titles)


def test_emg_gets_a_panel_even_with_two_components(pieces):
    import nyx.report as report

    emg, wake_sleep, pca, *_ = pieces
    clusters = nyx.cluster_sleep(
        pca, wake_sleep,
        clustering={"method": "kmeans", "n_clusters": 2, "pcs_to_use": [0, 1],
                    "use_emg": True},
        emg=emg,
    )
    titles = [ax.get_title() for ax in report.plot_cluster_features(clusters).axes]

    assert any("coloured by EMG power" == t for t in titles)


def test_cluster_features_takes_a_whole_result_too(scored):
    import nyx.report as report

    assert report.plot_cluster_features(scored).axes


def test_save_report_writes_the_feature_panels(scored, tmp_path):
    save_report(scored, str(tmp_path))

    assert (tmp_path / "plots" / "cluster_features.png").exists()


# ---------------------------------------------------------------------------
# Display knobs: smoothing, and the spectrogram colour range
# ---------------------------------------------------------------------------


def test_psd_smoothing_does_not_change_the_data(pieces):
    """Display only. A human recording is analysed at a fine frequency
    resolution and the raw spectra are too noisy to read the delta-versus-theta
    comparison off, but the numbers behind them must be untouched."""
    import nyx.report as report

    _emg, _ws, pca, clusters, _staging = pieces
    before = pca.spectrogram.copy()

    ax = report.plot_psd_per_cluster(pca, clusters=clusters, smooth=5)

    assert np.array_equal(pca.spectrogram, before)
    assert "smoothed" in ax.get_title()


def test_smoothing_reduces_the_wiggle(pieces):
    import nyx.report as report

    _emg, _ws, pca, clusters, _staging = pieces

    def roughness(ax):
        line = ax.get_lines()[0].get_ydata()
        return float(np.abs(np.diff(line)).mean())

    raw = roughness(report.plot_psd_per_cluster(pca, clusters=clusters))
    smoothed = roughness(report.plot_psd_per_cluster(pca, clusters=clusters, smooth=8))

    assert smoothed < raw


def test_cluster_check_passes_smoothing_through(pieces):
    import nyx.report as report

    _emg, _ws, pca, clusters, _staging = pieces
    figure = report.plot_cluster_check(pca, clusters=clusters, smooth=5)

    assert any("smoothed" in ax.get_title() for ax in figure.axes)


def test_spectrogram_colour_limits_can_be_set(scored):
    """The default range is set from the data, which washes out when one band is
    very loud -- so it has to be overridable per channel."""
    from matplotlib.collections import QuadMesh

    import nyx.report as report

    def limits(figure):
        return [tuple(round(float(v), 3) for v in mesh.get_clim())
                for ax in figure.axes for mesh in ax.collections
                if isinstance(mesh, QuadMesh)]

    automatic = limits(report.plot_scoring_overview(scored))
    explicit = limits(report.plot_scoring_overview(
        scored, eeg_range=(-5, 25), emg_range=(0, 10)))

    assert automatic != explicit
    assert (0.0, 10.0) in explicit and (-5.0, 25.0) in explicit


def test_spectrograms_can_be_capped_in_frequency(scored):
    import nyx.report as report

    figure = report.plot_scoring_overview(scored, eeg_fmax=12)
    tops = [ax.get_ylim()[1] for ax in figure.axes
            if ax.get_ylabel().startswith("EEG (")]

    assert tops and max(tops) <= 13


def test_palette_is_configurable(scored):
    from matplotlib.collections import QuadMesh

    import nyx.report as report

    figure = report.plot_scoring_overview(scored, palette="viridis")
    meshes = [m for ax in figure.axes for m in ax.collections
              if isinstance(m, QuadMesh)]

    assert meshes and all(m.get_cmap().name == "viridis" for m in meshes)
