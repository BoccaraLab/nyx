"""Summary figures for a scoring run.

The plots that tell you whether a run went well, in the order you would look at
them:

* :func:`plot_emg_threshold` -- did the wake/sleep cut land in a real valley?
* :func:`plot_pca_components` -- do the components look like spectral shapes,
  or like noise?
* :func:`plot_clusters` -- are the clusters separated, or one smear?
* :func:`plot_psd_per_cluster` -- **the one that matters.** NREM has more
  low-frequency power than REM; if the per-cluster spectra do not show that,
  the stage assignment is wrong however good the clustering looks.
* :func:`plot_hypnogram_result` -- does the resulting night look plausible?
* :func:`plot_confusion` -- only when a reference scoring exists.

:func:`plot_summary` puts them on one page, and :func:`save_report` writes both
the overview and the individual panels.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from nyx.stages import COLORS

__all__ = [
    "plot_scoring_overview",
    "plot_emg_threshold",
    "plot_pca_components",
    "plot_clusters",
    "plot_psd_per_cluster",
    "plot_hypnogram_result",
    "plot_confusion",
    "plot_summary",
    "save_report",
]

CLUSTER_CMAP = "Set1"

#: Stage order for hypnograms, top row first. NOSIGNAL sits above WAKE so that
#: the rows run from "not scored" down through progressively deeper sleep.
STAGE_ROW_ORDER = ("NOSIGNAL", "WAKE", "REM", "NREM1", "NREM", "NREM2", "NREM3")


def _cluster_colour(index: int):
    return plt.get_cmap(CLUSTER_CMAP)(index % 9)


def _despine(ax, keep=("left", "bottom")):
    """Remove the box around a plot, keeping only the axes you read values off."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    return ax


def _legend(ax, **kwargs):
    """Legend without a frame, everywhere."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, **kwargs)
    return ax


def _stage_rows(labels) -> list[str]:
    """Stages present, in display order."""
    present = set(np.asarray(labels).tolist())
    rows = [s for s in STAGE_ROW_ORDER if s in present]
    # Anything unrecognised (unnamed clusters, say) goes underneath.
    return rows + sorted(present - set(rows))


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------


def plot_emg_threshold(result, ax=None):
    """EMG power distribution with the wake/sleep cut drawn on it."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    power = result.emg.power
    nosignal = result.wake_sleep.nosignal_threshold
    threshold = result.wake_sleep.threshold

    # Show the whole distribution, including what the no-signal cut removes --
    # otherwise you cannot see whether that cut is in a sensible place.
    ax.hist(power, bins=100, density=True, alpha=0.55, color="#4C72B0",
            label="all epochs")

    if threshold <= power.max():
        ax.axvline(threshold, color="crimson", ls="--", lw=2,
                   label=f"wake/sleep = {threshold:.3f}")
    else:
        # Above the maximum of the min-max scaled power: every epoch is sleep,
        # and wake is recovered by a later clustering step instead.
        ax.text(0.5, 0.9, "no EMG split\n(all epochs -> SLEEP)", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="crimson")

    if nosignal > 0:
        ax.axvline(nosignal, color="darkorange", ls=":", lw=2,
                   label=f"no-signal = {nosignal:.3f}")
        ax.axvspan(power.min(), nosignal, color="darkorange", alpha=0.10)
        dropped = 100.0 * float((power < nosignal).mean())
        ax.text(0.02, 0.98, f"{dropped:.1f}% below no-signal", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="darkorange")
    else:
        # It was disabled, which is worth saying rather than leaving the reader
        # to wonder where the line went.
        ax.text(0.02, 0.98, "no-signal threshold disabled", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color="0.5")

    ax.set_xlabel("EMG power (scaled)")
    ax.set_ylabel("density")
    ax.set_title(f"EMG threshold ({result.wake_sleep.threshold_source})")
    _legend(ax, fontsize=8)
    return _despine(ax)


def plot_pca_components(result, ax=None, n: int = 4):
    """Component loadings across frequency: what each PC actually measures.

    ``ax`` is second in every panel function here, so they can be called
    uniformly as ``draw(result, ax)``.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    pca = result.pca

    for i in range(min(n, pca.pca.n_components_)):
        variance = 100 * pca.explained_variance_ratio[i]
        ax.plot(pca.freqs, pca.pca.components_[i],
                label=f"PC{i + 1} ({variance:.0f}%)", lw=1.4)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("loading")
    ax.set_title("PCA components")
    _legend(ax, fontsize=8)
    return _despine(ax)


def plot_clusters(result, ax=None):
    """Sleep epochs in the space the clustering actually used."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    clusters = result.clusters
    stages = result.staging.cluster_to_stage

    for i, cid in enumerate(clusters.unique_labels):
        mask = clusters.labels == cid
        ax.scatter(clusters.features_scaled[mask, 0], clusters.features_scaled[mask, 1],
                   s=5, alpha=0.45, color=_cluster_colour(i),
                   label=f"C{cid} -> {stages.get(int(cid), '?')}")
    if clusters.outlier_mask.any():
        ax.scatter(clusters.features_scaled[clusters.outlier_mask, 0],
                   clusters.features_scaled[clusters.outlier_mask, 1],
                   s=5, alpha=0.3, color="0.6", label="outlier -> NOSIGNAL")

    for i, centre in enumerate(clusters.centers):
        ax.scatter(centre[0], centre[1], marker="x", s=110, c="k", lw=2, zorder=5)
        ax.annotate(f"C{i}", (centre[0], centre[1]), xytext=(6, 4),
                    textcoords="offset points", fontweight="bold", fontsize=9)

    pcs = clusters.pcs_to_use
    ax.set_xlabel(f"PC{pcs[0] + 1}")
    ax.set_ylabel(f"PC{pcs[1] + 1}" if len(pcs) > 1 else "")
    ax.set_title("Clusters")
    _legend(ax, fontsize=8, markerscale=2)
    return _despine(ax)


def plot_psd_per_cluster(result, ax=None):
    """Mean spectrum of each cluster, with a 95% interval.

    The check that decides whether the stage assignment is right: NREM should
    carry more low-frequency power than REM.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    pca, clusters = result.pca, result.clusters
    stages = result.staging.cluster_to_stage

    sleep_bins = ~np.any(np.isnan(pca.signal), axis=1)
    spectra = pca.spectrogram[:, sleep_bins][:, clusters.valid_mask]

    for i, cid in enumerate(clusters.unique_labels):
        mask = clusters.labels == cid
        if mask.sum() < 2:
            continue
        block = spectra[:, mask]
        mean = block.mean(axis=1)
        ci = 1.96 * block.std(axis=1) / np.sqrt(mask.sum())
        colour = _cluster_colour(i)
        ax.plot(pca.freqs, mean, color=colour, lw=1.5,
                label=f"C{cid} -> {stages.get(int(cid), '?')} (n={int(mask.sum())})")
        ax.fill_between(pca.freqs, mean - ci, mean + ci, color=colour, alpha=0.25)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Mean spectrum per cluster")
    _legend(ax, fontsize=8)
    return _despine(ax)


def plot_hypnogram_result(result, ax=None, max_hours: float | None = None):
    """The scored night as a stepped hypnogram."""
    import pandas as pd

    from nyx.plotting import plot_hypnogram

    ax = ax if ax is not None else plt.subplots(figsize=(12, 3))[1]
    plot_hypnogram(
        pd.DataFrame(result.staging.hypnogram),
        possible_labels=_stage_rows(result.staging.hypnogram["label"]),
        title="Hypnogram",
        ax=ax,
        xaxis_format="Hours",
    )
    return ax


def plot_scoring_overview(result, max_points: int = 4000, figsize=None):
    """The signals, their spectrograms and the resulting hypnograms, aligned.

    The figure to read a whole recording from. Panels share a time axis, so a
    disagreement between the two hypnograms can be traced straight up to what
    the signal was doing at that moment.

    Rows, top to bottom: EMG trace, EMG spectrogram, EEG trace, EEG
    spectrogram, the reference hypnogram if there is one, and nyx's.

    The spectrograms are the ones the pipeline computed, not fresh ones -- so
    this shows the features scoring actually used, including any notch filter
    and normalisation. Built with :func:`nyx.plotting.generate_custom_plot`.
    """
    import pandas as pd

    from nyx.plotting import generate_custom_plot

    recording = result.recording
    reference = result.reference

    rows = _stage_rows(
        np.concatenate([
            np.asarray(result.staging.hypnogram["label"]),
            np.asarray(reference["label"]) if reference is not None else np.array([]),
        ])
    )

    emg_times = np.arange(len(recording.emg_trace())) / recording.emg_fs
    eeg_times = np.arange(len(recording.eeg_trace())) / recording.fs

    subplots = [
        # EMG first: it is what separates wake from sleep.
        {"type": "trace", "data": [emg_times, recording.emg_trace()],
         "label": "EMG", "lw": 0.3, "color": "#333333", "height": 1},
        {"type": "spectrogram",
         "data": [result.emg.freqs, result.emg.times, result.emg.spectrogram],
         "label": "power", "ylabel": "EMG (Hz)", "height": 2},
        {"type": "trace", "data": [eeg_times, recording.eeg_trace()],
         "label": "EEG", "lw": 0.3, "color": "#333333", "height": 1},
        {"type": "spectrogram",
         "data": [result.pca.freqs, result.pca.times, result.pca.spectrogram],
         "label": "power", "ylabel": "EEG (Hz)", "height": 2},
    ]
    if reference is not None:
        subplots.append({"type": "hypnogram", "data": pd.DataFrame(reference),
                         "possible_labels": rows, "title": "reference",
                         "xaxis_format": "Hours", "height": 1.1,
                         "drop_absent": False})
    subplots.append({"type": "hypnogram",
                     "data": pd.DataFrame(result.staging.hypnogram),
                     "possible_labels": rows, "title": "nyx",
                     "xaxis_format": "Hours", "height": 1.1,
                     "drop_absent": False})

    title = recording.name or "recording"
    if result.agreement is not None:
        title += f"   MF1 {result.agreement.mf1:.3f}"

    return generate_custom_plot({
        "figsize": figsize or (14, 1.7 * sum(s["height"] for s in subplots)),
        "subplots": subplots,
        "max_points": max_points,
        "suptitle": title,
    })


def plot_confusion(result, ax=None, normalise: bool = True):
    """Agreement with the reference scoring, as row-normalised percentages."""
    if result.agreement is None:
        raise ValueError(
            "This run has no reference scoring, so there is no confusion matrix. "
            "Pass reference=... to score_recording to get one."
        )
    ax = ax if ax is not None else plt.subplots(figsize=(5, 4.5))[1]
    agreement = result.agreement
    cm = np.asarray(agreement.confusion_matrix, dtype=float)

    shown = cm
    if normalise:
        # Percentages are comparable across recordings; seconds are not.
        totals = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0) * 100

    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=100 if normalise else None)
    labels = agreement.label_order
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("nyx")
    ax.set_ylabel("reference")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalise and cm[i].sum() == 0:
                continue
            ax.text(j, i, f"{shown[i, j]:.0f}" + ("%" if normalise else ""),
                    ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > shown.max() / 2 else "black")

    ax.set_title(f"MF1 {agreement.mf1:.3f}   acc {agreement.accuracy:.3f}   "
                 f"$\\kappa$ {agreement.kappa:.3f}", fontsize=10)
    plt.colorbar(image, ax=ax, fraction=0.046, label="% of reference stage")
    return _despine(ax, keep=())


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _safe_draw(draw, result, ax, name: str):
    """Draw a panel, turning a failure into a note on the axes.

    Scoring is expensive and the figures are a convenience; one panel that
    cannot be drawn must never cost you the rest of the page.
    """
    try:
        return draw(result, ax=ax)
    except Exception as exc:  # noqa: BLE001
        ax.clear()
        ax.text(0.5, 0.5, f"{name}\ncould not be drawn\n({type(exc).__name__})",
                ha="center", va="center", fontsize=9, color="crimson")
        ax.axis("off")
        warnings.warn(
            f"Could not draw the {name!r} panel: {type(exc).__name__}: {exc}",
            stacklevel=3,
        )
        return ax


def plot_summary(result, figsize: tuple[float, float] | None = None):
    """One page covering the whole run."""
    has_reference = result.agreement is not None
    figsize = figsize or (16, 11)

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7], hspace=0.42, wspace=0.28)

    _safe_draw(plot_emg_threshold, result, fig.add_subplot(grid[0, 0]), "emg_threshold")
    _safe_draw(plot_pca_components, result, fig.add_subplot(grid[0, 1]), "pca_components")
    _safe_draw(plot_clusters, result, fig.add_subplot(grid[0, 2]), "clusters")

    _safe_draw(plot_psd_per_cluster, result, fig.add_subplot(grid[1, 0]),
               "psd_per_cluster")
    _safe_draw(_plot_stage_durations, result, fig.add_subplot(grid[1, 1]),
               "stage_durations")
    if has_reference:
        _safe_draw(plot_confusion, result, fig.add_subplot(grid[1, 2]), "confusion")
    else:
        blank = fig.add_subplot(grid[1, 2])
        blank.text(0.5, 0.5, "no reference scoring\n(agreement not computed)",
                   ha="center", va="center", fontsize=10, color="0.4")
        blank.axis("off")

    _safe_draw(plot_hypnogram_result, result, fig.add_subplot(grid[2, :]), "hypnogram")

    title = f"nyx - {result.recording.name or 'recording'}"
    if has_reference:
        title += f"   (MF1 {result.agreement.mf1:.3f})"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    return fig


def _plot_stage_durations(result, ax=None):
    """How the recording divides between stages."""
    ax = ax if ax is not None else plt.subplots(figsize=(6, 4))[1]
    durations = result.staging.stage_durations()
    total = sum(durations.values()) or 1.0
    stages = sorted(durations, key=lambda s: -durations[s])

    ax.bar(range(len(stages)), [durations[s] / 3600.0 for s in stages],
           color=[COLORS.get(s, "#888888") for s in stages])
    for i, stage in enumerate(stages):
        ax.text(i, durations[stage] / 3600.0, f"{100 * durations[stage] / total:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(stages)), stages, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("hours")
    ax.set_title("Time per stage")
    return _despine(ax)


def _signal_check_figure(result):
    """The signal check of what was actually scored, for the record."""
    from nyx.inspect import check_signals

    notch = result.params.get("EEG", {}).get("notch")
    return check_signals(result.recording, preview=None, notch=notch).plot()


def save_report(result, output_dir: str, dpi: int = 150, panels: bool = True) -> str:
    """Write ``summary.png`` and, optionally, the individual panels.

    Returns the plots directory.
    """
    plots = os.path.join(output_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    def _write(name: str, build):
        try:
            figure = build()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not draw {name!r}: {type(exc).__name__}: {exc}",
                          stacklevel=3)
            return
        try:
            figure.savefig(os.path.join(plots, f"{name}.png"), dpi=dpi,
                           bbox_inches="tight")
        finally:
            plt.close(figure)

    _write("summary", lambda: plot_summary(result))

    # The whole recording on one time axis, with the hypnograms beneath it.
    _write("scoring_overview", lambda: plot_scoring_overview(result))

    # And the raw signal as it came in, for the record of what was scored.
    _write("signal_check", lambda: _signal_check_figure(result))

    if panels:
        individual = {
            "emg_threshold": plot_emg_threshold,
            "pca_components": plot_pca_components,
            "clusters": plot_clusters,
            "psd_per_cluster": plot_psd_per_cluster,
            "hypnogram": plot_hypnogram_result,
        }
        if result.agreement is not None:
            individual["confusion_matrix"] = plot_confusion

        for name, draw in individual.items():
            size = (12, 3) if name == "hypnogram" else (6, 4.5)
            fig, ax = plt.subplots(figsize=size)
            try:
                draw(result, ax=ax)
                fig.savefig(os.path.join(plots, f"{name}.png"), dpi=dpi,
                            bbox_inches="tight")
            except Exception as exc:  # noqa: BLE001
                # One panel failing must not cost you the others.
                warnings.warn(
                    f"Could not draw the {name!r} panel: "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
            finally:
                plt.close(fig)

    return plots
