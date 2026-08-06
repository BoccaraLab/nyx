"""Figures for a scoring run.

The plots that tell you whether a run went well, in the order you would look at
them:

* :func:`plot_emg_threshold` -- did the wake/sleep cut land in a real valley?
* :func:`plot_emg_power` -- and does it track the animal being awake?
* :func:`plot_wake_sleep` -- does the split look like a night?
* :func:`plot_pca_components` / :func:`plot_pca_grid` -- do the components look
  like spectral shapes, or like noise?
* :func:`plot_clusters` -- are the clusters separated, or one smear?
* :func:`plot_cluster_features` -- the same scatter coloured by each of the
  other features in turn. Clusters that overlap in the PC1/PC2 view are often
  separated cleanly by a further component or by the EMG, and this shows which.
* :func:`plot_psd_per_cluster` -- **the one that matters.** NREM has more
  low-frequency power than REM; if the per-cluster spectra do not show that,
  the stage assignment is wrong however good the clustering looks.
* :func:`plot_hypnogram_result` -- does the resulting night look plausible?
* :func:`plot_confusion` -- only when a reference scoring exists.

Every one of them takes **either a whole**
:class:`~nyx.pipeline.ScoringResult` **or the piece it draws**::

    plot_clusters(result)                      # after a full run
    plot_clusters(clusters, stages=mapping)    # partway through one

That is what lets a notebook working step by step use the same figures as the
finished report, rather than reimplementing them inline: partway through a run
there is no ``ScoringResult`` yet, only the pieces.

Three pairs come ready-combined, because they are only useful read together:
:func:`plot_emg_check`, :func:`plot_cluster_check`, and :func:`plot_summary`,
which puts the whole lot on one page. :func:`save_report` writes the overview
and the individual panels.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from nyx.stages import COLORS, STAGE_ROW_ORDER

__all__ = [
    "plot_scoring_overview",
    "plot_emg_threshold",
    "plot_emg_power",
    "plot_emg_check",
    "plot_pca_components",
    "plot_pca_grid",
    "plot_clusters",
    "plot_cluster_features",
    "cluster_ordering",
    "plot_psd_per_cluster",
    "plot_cluster_check",
    "plot_wake_sleep",
    "plot_hypnogram_result",
    "plot_confusion",
    "plot_summary",
    "save_report",
]

CLUSTER_CMAP = "Set1"

def _cluster_colour(index: int):
    return plt.get_cmap(CLUSTER_CMAP)(index % 9)


def _is_result(obj) -> bool:
    """Whether this is a whole :class:`~nyx.pipeline.ScoringResult`."""
    return obj is not None and hasattr(obj, "staging") and hasattr(obj, "clusters")


def _piece(obj, attribute: str):
    """The panel's main argument: a ``ScoringResult``, or the piece itself.

    Every panel below takes the object it draws -- the EMG features, the PCA,
    the clusters -- and also accepts a whole ``ScoringResult``, pulling that
    piece out of it. That is what lets a step-by-step notebook use the same
    functions as the finished report: partway through a run there is no
    ``ScoringResult`` yet, only the pieces.
    """
    return getattr(obj, attribute) if _is_result(obj) else obj


def _extra(obj, attribute: str, given=None):
    """A *second* piece a panel needs, e.g. the clusters alongside the PCA.

    Distinct from :func:`_piece`: here the caller's own value wins, and the
    result is only consulted when one was passed. Reading it off ``obj`` the way
    ``_piece`` does would hand back the main argument under another name.
    """
    if given is not None:
        return given
    return getattr(obj, attribute, None) if _is_result(obj) else None


def _stages_of(obj, given=None) -> dict:
    """Cluster-to-stage mapping, from a result or from what the caller passed.

    Before :func:`nyx.assign_stages` has run there is no mapping, so the
    clusters are drawn as bare ids -- which is exactly the state you are in when
    deciding what to call them.
    """
    if given is not None:
        return dict(given)
    return dict(obj.staging.cluster_to_stage) if _is_result(obj) else {}


def _cluster_label(cid: int, stages: dict) -> str:
    stage = stages.get(int(cid))
    return f"C{cid} -> {stage}" if stage else f"C{cid}"


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


def plot_emg_threshold(emg, ax=None, *, wake_sleep=None, threshold=None,
                       nosignal=0.0, source=""):
    """EMG power distribution with the wake/sleep cut drawn on it.

    Takes either a :class:`~nyx.pipeline.ScoringResult` or the
    :class:`~nyx.types.EmgFeatures` on their own -- in which case pass the
    ``threshold`` you are considering, or a ``wake_sleep`` to read it from.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]

    wake_sleep = _extra(emg, "wake_sleep", wake_sleep)
    emg = _piece(emg, "emg")

    if emg is None:
        ax.text(0.5, 0.5, "no EMG channel\n(wake came out of the EEG)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color="crimson")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("EMG threshold")
        return _despine(ax)

    if wake_sleep is not None:
        threshold = wake_sleep.threshold
        nosignal = wake_sleep.nosignal_threshold
        source = source or wake_sleep.threshold_source
    if threshold is None:
        raise ValueError(
            "Pass a threshold to draw, or a wake_sleep (or a whole "
            "ScoringResult) to read one from."
        )

    power = emg.power

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
    ax.set_title("EMG threshold" + (f" ({source})" if source else ""))
    _legend(ax, fontsize=8)
    return _despine(ax)


def plot_emg_power(emg, ax=None, *, wake_sleep=None, threshold=None):
    """EMG band power over time, with the wake/sleep cut across it.

    The companion to :func:`plot_emg_threshold`: the histogram says whether the
    cut sits in a valley, this says whether it tracks the animal actually being
    awake. A threshold can look perfect in a histogram and still be in the wrong
    place once you see it against time.
    """
    ax = ax or plt.subplots(figsize=(11, 3))[1]

    wake_sleep = _extra(emg, "wake_sleep", wake_sleep)
    emg = _piece(emg, "emg")
    if emg is None:
        ax.text(0.5, 0.5, "no EMG channel", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="crimson")
        ax.set_xticks([])
        ax.set_yticks([])
        return _despine(ax)

    if wake_sleep is not None:
        threshold = wake_sleep.threshold

    times = np.arange(len(emg.power)) / emg.fs
    ax.plot(times, emg.power, lw=0.4, color="#333333")

    title = "EMG power over time"
    if threshold is not None and threshold <= emg.power.max():
        ax.axhline(threshold, color="crimson", ls="--", lw=1.5,
                   label=f"wake/sleep = {threshold:.3f}")
        # In the title rather than annotated on the axes: the threshold is
        # often near the top of the range, so a corner label lands on the line.
        above = 100.0 * float((emg.power > threshold).mean())
        title += f"  --  {above:.1f}% above the cut"

    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG power")
    ax.set_title(title)
    _legend(ax, fontsize=8, loc="upper right")
    return _despine(ax)


def plot_emg_check(emg, *, wake_sleep=None, threshold=None, nosignal=0.0,
                   figsize=(14, 4)):
    """The two EMG panels side by side: the distribution and the time course.

    What you look at before accepting a wake/sleep threshold.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2]}
    )
    plot_emg_threshold(emg, left, wake_sleep=wake_sleep, threshold=threshold,
                       nosignal=nosignal)
    plot_emg_power(emg, right, wake_sleep=wake_sleep, threshold=threshold)
    fig.tight_layout()
    return fig


def plot_pca_components(pca, ax=None, *, n: int = 4):
    """Component loadings across frequency: what each PC actually measures.

    ``ax`` is second in every panel function here, so they can all be called
    uniformly as ``draw(result, ax)``.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    pca = _piece(pca, "pca")

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


def _subsample(n: int, max_points: int) -> np.ndarray:
    """Indices of at most ``max_points`` rows, in order and reproducibly."""
    if max_points is None or n <= max_points:
        return np.arange(n)
    keep = np.random.default_rng(0).choice(n, max_points, replace=False)
    keep.sort()
    return keep


def plot_pca_grid(pca, *, n: int | None = None, smooth: float = 0.0,
                  figsize=(14, 6)):
    """One panel per principal component, rather than all on one axis.

    Easier to read than :func:`plot_pca_components` when you are deciding
    whether the components describe sleep at all -- a component dominated by one
    narrow peak is describing an artefact or residual mains, and that is obvious
    here and easy to miss in an overlay.

    Parameters
    ----------
    smooth
        Gaussian sigma, in frequency bins, applied to the curves **for display
        only**. Human recordings analysed at a fine frequency resolution give
        visibly noisy loadings, and the shape is what you are reading. Leave at
        0 for rodents, where the bins are wide enough already.
    """
    from scipy.ndimage import gaussian_filter1d

    pca = _piece(pca, "pca")
    total = pca.pca.n_components_
    n = total if n is None else min(n, total)

    columns = min(n, 3)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True,
                             squeeze=False)

    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_visible(False)
            continue
        loading = pca.pca.components_[i]
        if smooth:
            loading = gaussian_filter1d(loading, sigma=float(smooth))
        ax.plot(pca.freqs, loading, lw=1.4)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"PC{i + 1}: {100 * pca.explained_variance_ratio[i]:.1f}% var",
                     fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        _despine(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("loading")

    fig.tight_layout()
    return fig


def cluster_ordering(clusters, stages=None):
    """The cluster ordering ``stage_order`` maps onto, with the centroids.

    ``stage_order`` names clusters by their position in this order, so this
    table is what tells you which name is going where. That matters because the
    order is often **not** readable off the scatter plot: two centroids a
    hundredth apart on PC1 look identical there, and swapping them silently
    swaps two stages.

    Returns a DataFrame, one row per cluster, ordered as the names are applied.
    """
    import pandas as pd

    from nyx.pipeline import cluster_order

    stages = _stages_of(clusters, stages)
    clusters = _piece(clusters, "clusters")

    sizes = clusters.sizes()
    rows = []
    for position, cid in enumerate(cluster_order(clusters)):
        row = {"order": position, "cluster": cid}
        for axis, pc in enumerate(clusters.pcs_to_use):
            row[f"PC{pc + 1}"] = round(float(clusters.centers[cid, axis]), 4)
        if clusters.centers.shape[1] > len(clusters.pcs_to_use):
            row["EMG"] = round(float(clusters.centers[cid, len(clusters.pcs_to_use)]), 4)
        row["epochs"] = sizes.get(cid, 0)
        if stages:
            row["stage"] = stages.get(cid, "")
        rows.append(row)

    return pd.DataFrame(rows).set_index("order")


def plot_clusters(clusters, ax=None, *, stages=None, max_points: int = 20_000):
    """Sleep epochs in the space the clustering actually used.

    Takes either a :class:`~nyx.pipeline.ScoringResult` or the
    :class:`~nyx.types.SleepClusters` on their own. Without a stage mapping the
    clusters are drawn as bare ids, which is the state you are in when deciding
    what to call them.

    ``max_points`` caps how many epochs are drawn. The scalogram backend
    produces one column per sample rather than per epoch, which is millions of
    points -- more than a scatter plot can show anything with, and enough to
    make the figure unopenable. The subsample is a fixed random draw, so the
    picture is the same every time; the clustering itself always used every
    point.
    """
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    stages = _stages_of(clusters, stages)
    clusters = _piece(clusters, "clusters")

    keep = _subsample(len(clusters.features_scaled), max_points)
    points = clusters.features_scaled[keep]
    labels = clusters.labels[keep]
    outliers = clusters.outlier_mask[keep]

    from nyx.pipeline import cluster_order

    # The position of each cluster in the order stage_order names them. Shown
    # because it is what the naming actually depends on, and it is not readable
    # off the plot when two centroids sit close together.
    rank = {cid: position for position, cid in enumerate(cluster_order(clusters))}

    for i, cid in enumerate(clusters.unique_labels):
        mask = labels == cid
        ax.scatter(points[mask, 0], points[mask, 1],
                   s=5, alpha=0.45, color=_cluster_colour(i), rasterized=True,
                   label=f"#{rank[int(cid)]}  {_cluster_label(cid, stages)}")
    if outliers.any():
        ax.scatter(points[outliers, 0], points[outliers, 1],
                   s=5, alpha=0.3, color="0.6", rasterized=True,
                   label="outlier -> NOSIGNAL")

    for cid in clusters.unique_labels:
        centre = clusters.centers[int(cid)]
        ax.scatter(centre[0], centre[1], marker="x", s=110, c="k", lw=2, zorder=5)
        ax.annotate(
            f"#{rank[int(cid)]} C{int(cid)}\n({centre[0]:+.2f}, {centre[1]:+.2f})"
            if clusters.centers.shape[1] > 1
            else f"#{rank[int(cid)]} C{int(cid)}\n({centre[0]:+.2f})",
            (centre[0], centre[1]), xytext=(7, 5), textcoords="offset points",
            fontweight="bold", fontsize=8, zorder=6,
        )

    pcs = clusters.pcs_to_use
    ax.set_xlabel(f"PC{pcs[0] + 1}")
    ax.set_ylabel(f"PC{pcs[1] + 1}" if len(pcs) > 1 else "")
    ax.set_title("Clusters  (#n = the order stage_order names them in)")
    _legend(ax, fontsize=8, markerscale=2)
    return _despine(ax)


def _feature_names(clusters) -> list[str]:
    """What each column of ``features_scaled`` is."""
    names = [f"PC{pc + 1}" for pc in clusters.pcs_to_use]
    # cluster_sleep appends EMG power as a trailing column when use_emg is set.
    if clusters.features_scaled.shape[1] > len(names):
        names.append("EMG power")
    return names


def plot_cluster_features(clusters, *, stages=None, max_points: int = 20_000,
                          cmap: str = "RdYlBu_r", columns: int = 3, figsize=None):
    """The same scatter again, coloured by each feature the axes do not show.

    The cluster plot shows two dimensions, but the clustering used more --
    further components, and the EMG power when ``use_emg`` is set. Two clusters
    that overlap in the PC1/PC2 view are often cleanly separated by one of those
    others, and this is where you see which.

    Read it when the clusters look like one smear, or when you cannot tell why
    the algorithm split where it did: the panel whose colour changes across the
    boundary is the feature that drew it.

    Only the features beyond the two on the axes get a panel -- colouring by
    PC1 or PC2 would restate the axes rather than add anything. A clustering on
    two components with no EMG therefore has nothing further to show, and the
    figure says so.
    """
    stages = _stages_of(clusters, stages)
    clusters = _piece(clusters, "clusters")

    names = _feature_names(clusters)
    keep = _subsample(len(clusters.features_scaled), max_points)
    points = clusters.features_scaled[keep]
    labels = clusters.labels[keep]
    inliers = ~clusters.outlier_mask[keep]

    x, y = points[:, 0], points[:, 1] if points.shape[1] > 1 else np.zeros(len(points))
    # Only the features the scatter does not already show. Colouring by PC1 or
    # PC2 would just restate the axes.
    extras = list(range(2, len(names)))
    panels = 1 + len(extras)

    columns = min(panels, columns)
    rows = (panels + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=figsize or (5.5 * columns, 4.6 * rows))
    flat = axes.ravel()

    def centroids(ax):
        for cid in clusters.unique_labels:
            centre = clusters.centers[int(cid)]
            ax.scatter(centre[0], centre[1] if len(centre) > 1 else 0.0,
                       marker="x", s=110, c="k", lw=2, zorder=5)

    plot_clusters(clusters, flat[0], stages=stages, max_points=max_points)

    if not extras:
        # Two components and no EMG: the axes already are the whole feature
        # space. Say that rather than returning a lone panel that looks like
        # something failed to draw.
        flat[0].set_title(
            f"Clusters -- {names[0]} and {names[-1]} are the only features used"
        )

    for panel, column in enumerate(extras, start=1):
        ax = flat[panel]
        scatter = ax.scatter(x[inliers], y[inliers], c=points[inliers, column],
                             s=5, cmap=cmap, rasterized=True)
        fig.colorbar(scatter, ax=ax, label=names[column])
        centroids(ax)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1] if len(names) > 1 else "")
        ax.set_title(f"coloured by {names[column]}")
        _despine(ax)

    for ax in flat[panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_psd_per_cluster(pca, ax=None, *, clusters=None, stages=None,
                         smooth: float = 0.0):
    """Mean spectrum of each cluster, with a 95% interval.

    **The panel that decides whether the stage assignment is right**: NREM
    carries more low-frequency power than REM, and if these spectra do not show
    that, the assignment is wrong however clean the clusters look in PC space.

    Takes either a :class:`~nyx.pipeline.ScoringResult` or the ``pca`` and
    ``clusters`` separately.

    Parameters
    ----------
    smooth
        Gaussian sigma, in frequency bins, applied **for display only** -- the
        same knob as :func:`plot_pca_grid`. Human recordings are analysed at a
        fine frequency resolution, and the resulting spectra are noisy enough
        that the delta-versus-theta comparison this panel exists for is hard to
        read. Leave at 0 for rodents, whose bins are already wide.

        The confidence interval is smoothed with it, so the band still means
        what it says relative to the line it surrounds.
    """
    from scipy.ndimage import gaussian_filter1d

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    stages = _stages_of(pca, stages)
    clusters = _extra(pca, "clusters", clusters)
    pca = _piece(pca, "pca")
    if clusters is None:
        raise ValueError(
            "Pass the clusters to draw, or a whole ScoringResult to read them from."
        )

    sleep_bins = ~np.any(np.isnan(pca.signal), axis=1)
    spectra = pca.spectrogram[:, sleep_bins][:, clusters.valid_mask]

    for i, cid in enumerate(clusters.unique_labels):
        mask = clusters.labels == cid
        if mask.sum() < 2:
            continue
        block = spectra[:, mask]
        mean = block.mean(axis=1)
        ci = 1.96 * block.std(axis=1) / np.sqrt(mask.sum())
        if smooth:
            # Both, so the band keeps its meaning relative to the line.
            mean = gaussian_filter1d(mean, sigma=float(smooth))
            ci = gaussian_filter1d(ci, sigma=float(smooth))
        colour = _cluster_colour(i)
        ax.plot(pca.freqs, mean, color=colour, lw=1.5,
                label=f"{_cluster_label(cid, stages)} (n={int(mask.sum())})")
        ax.fill_between(pca.freqs, mean - ci, mean + ci, color=colour, alpha=0.25)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Mean spectrum per cluster"
                 + (f"  (smoothed, sigma {smooth:g})" if smooth else ""))
    _legend(ax, fontsize=8)
    return _despine(ax)


def plot_cluster_check(pca, *, clusters=None, stages=None, smooth: float = 0.0,
                       figsize=(14, 5)):
    """Clusters in PC space beside their mean spectra.

    The pair you look at before naming clusters. Read the **right** panel: the
    left one only tells you the clustering separated something, the right one
    tells you what that something is.

    ``smooth`` goes to :func:`plot_psd_per_cluster`; use it on human recordings,
    where the fine frequency resolution makes the raw spectra hard to read.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=figsize)
    plot_clusters(_extra(pca, "clusters", clusters) or pca, left,
                  stages=_stages_of(pca, stages))
    plot_psd_per_cluster(pca, right, clusters=clusters, stages=stages,
                         smooth=smooth)
    right.set_title(right.get_title() + "  <- read this one")
    fig.tight_layout()
    return fig


def plot_wake_sleep(wake_sleep, ax=None):
    """The wake/sleep/no-signal split, before any stage clustering.

    Worth looking at on its own: everything after it only subdivides SLEEP, so
    wake that is wrong here stays wrong.
    """
    import pandas as pd

    from nyx.plotting import plot_hypnogram

    ax = ax if ax is not None else plt.subplots(figsize=(12, 2.5))[1]
    hypnogram = _piece(wake_sleep, "wake_sleep").hypnogram
    plot_hypnogram(
        pd.DataFrame(hypnogram),
        possible_labels=_stage_rows(hypnogram["label"]),
        title="Wake / sleep",
        ax=ax,
        xaxis_format="Hours",
    )
    return ax


def plot_hypnogram_result(staging, ax=None, *, title: str = "Hypnogram"):
    """The scored night as a stepped hypnogram.

    Takes a :class:`~nyx.pipeline.ScoringResult`, a
    :class:`~nyx.types.Staging`, or a plain hypnogram dict.
    """
    import pandas as pd

    from nyx.plotting import plot_hypnogram

    ax = ax if ax is not None else plt.subplots(figsize=(12, 3))[1]

    staging = _piece(staging, "staging")
    hypnogram = getattr(staging, "hypnogram", staging)

    plot_hypnogram(
        pd.DataFrame(hypnogram),
        possible_labels=_stage_rows(hypnogram["label"]),
        title=title,
        ax=ax,
        xaxis_format="Hours",
    )
    return ax


def plot_scoring_overview(result, max_points: int = 4000, figsize=None, *,
                          eeg_range=None, emg_range=None, palette: str = "jet",
                          eeg_fmax=None, emg_fmax=None):
    """The signals, their spectrograms and the resulting hypnograms, aligned.

    The figure to read a whole recording from. Panels share a time axis, so a
    disagreement between the two hypnograms can be traced straight up to what
    the signal was doing at that moment.

    Rows, top to bottom: EMG trace, EMG spectrogram, EEG trace, EEG
    spectrogram, the reference hypnogram if there is one, and nyx's.

    The spectrograms are the ones the pipeline computed, not fresh ones -- so
    this shows the features scoring actually used, including any notch filter
    and normalisation. Built with :func:`nyx.plotting.generate_custom_plot`.

    Parameters
    ----------
    eeg_range, emg_range
        Colour limits for the spectrograms, as ``(vmin, vmax)`` in the units of
        the spectrogram -- dB, normally. Left out, each is set from the data,
        which is usually right but washes out when a recording carries one very
        loud band. Read the colorbar on a first draw to see what range to ask
        for.
    palette
        Any matplotlib colormap name.
    eeg_fmax, emg_fmax
        Highest frequency to draw. The default shows the whole computed band.

    Examples
    --------
    Each knob is per channel, since the two rarely want the same treatment::

        nyx.plot_scoring_overview(result, eeg_range=(-10, 30), eeg_fmax=25,
                                  palette="viridis")
    """
    import pandas as pd

    from nyx.plotting import generate_custom_plot

    from nyx.metrics import normalise_labels

    recording = result.recording
    reference = result.reference

    # The reference keeps whatever spellings its file used -- 'awake',
    # 'non-REM' -- which have no entry in the stage palette and would each get a
    # hypnogram row of their own beside nyx's WAKE and NREM. Normalise it for
    # drawing, exactly as the comparison does before measuring agreement.
    if reference is not None:
        reference = normalise_labels(reference)

    rows = _stage_rows(
        np.concatenate([
            np.asarray(result.staging.hypnogram["label"]),
            np.asarray(reference["label"]) if reference is not None else np.array([]),
        ])
    )

    eeg_times = np.arange(len(recording.eeg_trace())) / recording.fs

    def spectrogram(freqs, times, values, ylabel, im_range, fmax):
        band = slice(None) if fmax is None else (freqs <= float(fmax))
        panel = {
            "type": "spectrogram",
            "data": [freqs[band], times, values[band]],
            "label": "power", "ylabel": ylabel, "height": 2,
            "palette": palette,
        }
        if im_range is not None:
            panel["im_range"] = tuple(im_range)
        return panel

    subplots = []
    if result.emg is not None:
        # EMG first: it is what separates wake from sleep.
        emg_times = np.arange(len(recording.emg_trace())) / recording.emg_fs
        subplots += [
            {"type": "trace", "data": [emg_times, recording.emg_trace()],
             "label": "EMG", "lw": 0.3, "color": "#333333", "height": 1},
            spectrogram(result.emg.freqs, result.emg.times,
                        result.emg.spectrogram, "EMG (Hz)", emg_range, emg_fmax),
        ]
    subplots += [
        {"type": "trace", "data": [eeg_times, recording.eeg_trace()],
         "label": "EEG", "lw": 0.3, "color": "#333333", "height": 1},
        spectrogram(result.pca.freqs, result.pca.times, result.pca.spectrogram,
                    "EEG (Hz)", eeg_range, eeg_fmax),
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

    # The clustering's other dimensions. The summary shows PC1 against PC2, but
    # the split may have been drawn by a component or by the EMG that view does
    # not show at all.
    _write("cluster_features", lambda: plot_cluster_features(result))

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
