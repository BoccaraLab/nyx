"""Summary figures, and the MF1 metric they report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nyx
from nyx.report import (
    plot_clusters,
    plot_confusion,
    plot_emg_threshold,
    plot_hypnogram_result,
    plot_pca_components,
    plot_psd_per_cluster,
    plot_summary,
    save_report,
)

PANELS = [
    plot_emg_threshold,
    plot_pca_components,
    plot_clusters,
    plot_psd_per_cluster,
    plot_hypnogram_result,
]


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
