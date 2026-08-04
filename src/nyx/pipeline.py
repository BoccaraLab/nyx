"""The scoring pipeline, one function per step.

Each step takes the previous step's result and returns a new one, so a caller can
stop after any step, show it to a user, change something, and re-run only what
comes after. The notebook, a batch script and a GUI all drive the same
functions.

The steps, in order::

    compute_emg_features   EMG spectrogram -> band-power trace
    find_wake_sleep_threshold      pick the wake/sleep cut on that trace
    classify_wake_sleep    -> WAKE / SLEEP / NOSIGNAL hypnogram
    compute_sleep_pca      EEG spectrogram -> PCA fitted on sleep epochs only
    cluster_sleep          cluster sleep epochs in PC space
    assign_stages          map clusters onto NREM / REM -> final hypnogram
    evaluate               OPTIONAL: agreement with a manual scoring

:func:`score_recording` runs all of them with automatic choices at each
decision point.

Three points in this pipeline are genuine judgement calls: 
the analysis window, the EMG wake/sleep threshold, and clustering parameters. 
Each has an automatic default and an explicit override.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nyx.clustering import reconstruct_signal_multiclass, run_clustering_step
from nyx.features import compute_eeg_pca_feature, compute_emg_power_trace
from nyx.metrics import compare_sleep, trim_manual_scores
from nyx.postprocess import apply_rules, rules_from_params
from nyx.scoring import (
    classify_wakesleep,
    prepare_epoch_data,
    save_hypno_with_padding,
)
from nyx.thresholds import find_emg_threshold
from nyx.types import (
    Agreement,
    EmgFeatures,
    Recording,
    SleepClusters,
    SleepPca,
    Staging,
    WakeSleep,
)

__all__ = [
    "DEFAULT_CLUSTERING",
    "ScoringResult",
    "compute_emg_features",
    "find_wake_sleep_threshold",
    "classify_wake_sleep",
    "compute_sleep_pca",
    "cluster_sleep",
    "assign_stages",
    "evaluate",
    "score_recording",
    "save_results",
]


# Defaults for :func:`cluster_sleep`. These are the values the published
# analyses used; a `clustering` section in the params file overrides them.
DEFAULT_CLUSTERING: dict[str, Any] = {
    "method": "hdbscan",  # "hdbscan", "kmeans", "gmm" or "elliptic"
    "n_clusters": 2,  # used by kmeans and gmm only
    "pcs_to_use": [0, 1, 2, 3],
    "use_emg": False,  # add EMG power as an extra clustering feature
    "outlier_z_threshold": 6,  # epochs beyond this |z| are marked NOSIGNAL
    "hdbscan_min_cluster_size": 300,
    "hdbscan_min_samples": 30,
    "elliptic_contamination": 0.1,  # elliptic only
    "elliptic_support_fraction": 0.75,  # elliptic only
    # Renumber clusters by centroid so that ids are a property of the data, not
    # of how the algorithm happened to initialise. Keep this on: without it a
    # saved cluster_to_stage mapping is not reproducible.
    "canonical_cluster_ids": True,
}

# Integer codes used for the final stage array.
_STAGE_TO_INT = {"NOSIGNAL": -2, "WAKE": -1, "NREM": 0, "REM": 1}
_INT_TO_STAGE = {value: key for key, value in _STAGE_TO_INT.items()}


@dataclass
class ScoringResult:
    """Everything :func:`score_recording` produced, in one object."""

    recording: Recording
    emg: EmgFeatures
    wake_sleep: WakeSleep
    pca: SleepPca
    clusters: SleepClusters
    staging: Staging
    agreement: Agreement | None = None
    window: tuple[float, float] = (0.0, 0.0)
    total_duration: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    #: The reference scoring, trimmed to the analysis window, when one was
    #: given. Kept so figures can show it beside nyx's own hypnogram.
    reference: dict[str, np.ndarray] | None = None

    @property
    def hypnogram(self) -> dict[str, np.ndarray]:
        """The final hypnogram (WAKE / NREM / REM / NOSIGNAL)."""
        return self.staging.hypnogram


# ---------------------------------------------------------------------------
# Step 1 -- EMG band power
# ---------------------------------------------------------------------------


def compute_emg_features(
    recording: Recording, params: dict, return_in_uV: bool = False
) -> EmgFeatures:
    """Compute the EMG spectrogram and collapse it to a band-power trace.

    Parameters
    ----------
    recording
        Recording, already restricted to the analysis window.
    params
        Full parameter dict; the ``"EMG"`` section is used.
    return_in_uV
        Convert the raw trace to microvolts first. Only meaningful if the file
        carries a valid gain; the published analyses used ``False``.
    """
    # The EMG may be sampled differently from the EEG, as it is in several
    # polysomnography formats, so use its own rate rather than the recording's.
    trace = recording.emg_trace(return_in_uV=return_in_uV)
    spectrogram, freqs, times, power, fs = compute_emg_power_trace(
        trace, recording.emg_fs, params["EMG"]
    )
    return EmgFeatures(
        spectrogram=spectrogram, freqs=freqs, times=times, power=power, fs=fs
    )


# ---------------------------------------------------------------------------
# Step 2 -- wake/sleep threshold
# ---------------------------------------------------------------------------


def find_wake_sleep_threshold(
    emg: EmgFeatures, nosignal_threshold: float = 0.0, random_state: int = 0
) -> float:
    """Find the EMG power that best separates wake from sleep.

    Fits a Gaussian mixture to the EMG power distribution and cuts between the
    two groups of components. Epochs at or below ``nosignal_threshold`` are
    excluded first, so flat or disconnected stretches do not drag the cut down.

    This is the first of the pipeline's three judgement calls: check the
    histogram, and if the cut does not sit in a clear valley between two modes,
    pass your own value to :func:`classify_wake_sleep`.
    """
    feature = emg.power[emg.power > nosignal_threshold]
    if feature.size == 0:
        raise ValueError(
            f"No EMG samples above the no-signal threshold ({nosignal_threshold}). "
            f"EMG power ranges from {emg.power.min():.3f} to {emg.power.max():.3f}; "
            f"lower nosignal_threshold."
        )
    return float(find_emg_threshold(feature, random_state=random_state))


def classify_wake_sleep(
    emg: EmgFeatures,
    threshold: float | None = None,
    nosignal_threshold: float = 0.0,
    min_duration: float = 4.0,
    random_state: int = 0,
) -> WakeSleep:
    """Split the recording into WAKE, SLEEP and NOSIGNAL.

    Parameters
    ----------
    threshold
        EMG power above which an epoch is wake. ``None`` (the default) finds it
        automatically with :func:`find_wake_sleep_threshold`.
    nosignal_threshold
        EMG power at or below which an epoch is treated as signal loss rather
        than as sleep. ``0.0`` disables it.
    min_duration
        Segments shorter than this many seconds are absorbed into their
        neighbours, which removes single-epoch flicker.
    """
    source = "manual"
    if threshold is None:
        threshold = find_wake_sleep_threshold(
            emg, nosignal_threshold=nosignal_threshold, random_state=random_state
        )
        source = "auto"

    hypnogram = classify_wakesleep(
        emg.power,
        [nosignal_threshold, float(threshold)],
        emg.fs,
        min_duration=min_duration,
    )
    return WakeSleep(
        hypnogram=hypnogram,
        threshold=float(threshold),
        nosignal_threshold=float(nosignal_threshold),
        threshold_source=source,
    )


# ---------------------------------------------------------------------------
# Step 3 -- EEG PCA over sleep epochs
# ---------------------------------------------------------------------------


def compute_sleep_pca(
    recording: Recording,
    params: dict,
    wake_sleep: WakeSleep,
    return_in_uV: bool = False,
    within: str = "SLEEP",
) -> SleepPca:
    """Compute the EEG spectrogram and run PCA on the epochs labelled ``within``.

    Fitting inside one label keeps epochs that are already separated -- wake,
    when splitting sleep -- from dominating the variance and defining the
    components. ``within`` is what lets a later step refit inside WAKE or inside
    a specific NREM stage; see :mod:`nyx.steps`.
    """
    trace = recording.eeg_trace(return_in_uV=return_in_uV)
    spectrogram, freqs, times, pca, scores, signal, fs = compute_eeg_pca_feature(
        trace, recording.fs, params, wake_sleep.hypnogram, within=within
    )
    return SleepPca(
        spectrogram=spectrogram,
        freqs=freqs,
        times=times,
        pca=pca,
        scores=scores,
        signal=signal,
        fs=fs,
    )


# ---------------------------------------------------------------------------
# Step 4 -- cluster the sleep epochs
# ---------------------------------------------------------------------------


def resolve_clustering_params(params: dict | None) -> dict[str, Any]:
    """Merge a ``clustering`` section over :data:`DEFAULT_CLUSTERING`."""
    resolved = dict(DEFAULT_CLUSTERING)
    if not params:
        return resolved

    # Accept either a full params dict or just its clustering section.
    is_full_params = "EEG" in params or "EMG" in params
    section = params.get("clustering", {}) if is_full_params else params
    if not section:
        return resolved

    unknown = [key for key in section if key not in DEFAULT_CLUSTERING]
    if unknown:
        warnings.warn(
            f"Ignoring unrecognised clustering setting(s) {unknown}. "
            f"Recognised settings: {sorted(DEFAULT_CLUSTERING)}. "
            f"('min_cluster_size', 'min_samples', 'nrem_weight' and 'rem_frac' "
            f"belonged to an older LDA-based routine that has been removed; they "
            f"are not equivalent to the 'hdbscan_*' settings. Delete them from "
            f"your parameter file.)",
            stacklevel=2,
        )
    resolved.update({k: v for k, v in section.items() if k in DEFAULT_CLUSTERING})
    return resolved


def cluster_sleep(
    pca: SleepPca,
    wake_sleep: WakeSleep,
    clustering: dict | None = None,
    emg: EmgFeatures | None = None,
) -> SleepClusters:
    """Cluster sleep epochs in PC space.

    Parameters
    ----------
    pca
        Result of :func:`compute_sleep_pca`.
    wake_sleep
        Result of :func:`classify_wake_sleep`, used to align EMG to EEG epochs.
    clustering
        Settings, merged over :data:`DEFAULT_CLUSTERING`. May be a full params
        dict (its ``"clustering"`` section is used) or just the section.
    emg
        Required only when ``use_emg`` is enabled.
    """
    settings = resolve_clustering_params(clustering)
    pcs_to_use = list(settings["pcs_to_use"])
    use_emg = bool(settings["use_emg"])

    if pca.scores.size == 0:
        raise ValueError(
            "No sleep epochs to cluster. Either the recording contains no sleep, "
            "or the EMG wake/sleep threshold is too low."
        )
    max_pc = max(pcs_to_use)
    if max_pc >= pca.scores.shape[1]:
        raise ValueError(
            f"pcs_to_use={pcs_to_use} needs at least {max_pc + 1} components but the "
            f"PCA has {pca.scores.shape[1]}. Raise params['scoring']['pc_components'] "
            f"or drop the higher components."
        )

    features = pca.scores[:, pcs_to_use]

    if use_emg:
        if emg is None:
            raise ValueError("use_emg is enabled but no EmgFeatures were passed.")
        emg_aligned = _emg_aligned_to_sleep_epochs(emg, wake_sleep, pca.fs)
        n = min(len(features), len(emg_aligned))
        features = np.column_stack([features[:n], emg_aligned[:n]])

    valid_mask = ~np.any(np.isnan(features), axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[valid_mask])

    # Extreme epochs are almost always artefact rather than a real stage; they
    # are set aside as NOSIGNAL instead of distorting the clusters.
    z_threshold = settings.get("outlier_z_threshold")
    if z_threshold is not None:
        outlier_mask = np.max(np.abs(features_scaled), axis=1) > float(z_threshold)
    else:
        outlier_mask = np.zeros(len(features_scaled), dtype=bool)

    inlier_labels, _raw_labels, centers = run_clustering_step(
        features_scaled[~outlier_mask], scaler, settings
    )

    if settings.get("canonical_cluster_ids", True):
        inlier_labels, centers = _canonicalise_cluster_ids(inlier_labels, centers)

    labels = np.full(len(features_scaled), _STAGE_TO_INT["NOSIGNAL"], dtype=int)
    labels[~outlier_mask] = inlier_labels

    return SleepClusters(
        labels=labels,
        centers=centers,
        unique_labels=np.unique(inlier_labels),
        features_scaled=features_scaled,
        valid_mask=valid_mask,
        outlier_mask=outlier_mask,
        scaler=scaler,
        pcs_to_use=pcs_to_use,
        params=settings,
    )


def _canonicalise_cluster_ids(
    labels: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Renumber clusters so that id 0 has the lowest centroid on the first feature.

    Clustering algorithms number their output arbitrarily -- k-means and GMM by
    initialisation, HDBSCAN by internal traversal order -- so the *same*
    physical cluster can come back as 0 on one run and 2 on the next. That makes
    a saved ``cluster_to_stage`` mapping unreproducible: replaying the exact
    parameters can attach the labels to the wrong clusters.

    Sorting by centroid position makes the ids a property of the data rather
    than of the run, so ``cluster_to_stage`` means the same thing every time.
    Cluster 0 is always the one furthest along the negative direction of the
    first component used.

    This rarely bites with two or three clusters, where the ordering usually
    comes out the same anyway, but it does with the four-plus clusters human
    scoring needs.
    """
    order = np.argsort(centers[:, 0], kind="stable")
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    return remap[labels], centers[order]


def _emg_aligned_to_sleep_epochs(
    emg: EmgFeatures, wake_sleep: WakeSleep, fs_eeg: float
) -> np.ndarray:
    """Resample EMG power onto the EEG epoch grid, over sleep segments only."""
    from scipy.interpolate import interp1d

    segments = []
    for _, row in pd.DataFrame(wake_sleep.hypnogram).iterrows():
        if row["label"] != "SLEEP":
            continue

        start_emg = max(0, min(int(row["time"] * emg.fs), len(emg.power)))
        end_emg = max(0, min(int((row["time"] + row["duration"]) * emg.fs), len(emg.power)))
        start_eeg = int(row["time"] * fs_eeg)
        end_eeg = int((row["time"] + row["duration"]) * fs_eeg)
        n_eeg = end_eeg - start_eeg

        segment = emg.power[start_emg:end_emg]
        if fs_eeg == emg.fs:
            segments.append(segment[:n_eeg])
            continue

        eeg_times = np.arange(n_eeg) / fs_eeg
        if len(segment) > 1:
            interpolate = interp1d(
                np.arange(len(segment)) / emg.fs,
                segment,
                kind="linear",
                bounds_error=False,
                fill_value=(segment[0], segment[-1]),
            )
            segments.append(interpolate(eeg_times))
        else:
            fill = segment[0] if len(segment) else np.nan
            segments.append(np.full(n_eeg, fill))

    if not segments:
        return np.array([], dtype=float)
    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Step 5 -- clusters to stages
# ---------------------------------------------------------------------------


def assign_stages(
    clusters: SleepClusters,
    pca: SleepPca,
    wake_sleep: WakeSleep,
    stage_order: Sequence[str] = ("REM", "NREM"),
    overrides: dict[int, str] | None = None,
    within: str = "SLEEP",
) -> Staging:
    """Map clusters onto sleep stages and build the final hypnogram.

    Clusters are ordered by their centroid along the first PC used, and
    ``stage_order`` names them from lowest to highest. This is the pipeline's
    third judgement call: check the per-cluster PSDs, and if the assignment is
    wrong, correct it with ``overrides``.

    Parameters
    ----------
    stage_order
        Stage names from lowest to highest centroid on the first PC. The
        default ``("REM", "NREM")`` reflects REM having less low-frequency power.
    overrides
        ``{cluster_id: stage}``, applied after the automatic assignment.

    Rules that rewrite implausible *sequences* -- REM out of wake, segments too
    short to be a real bout -- are not applied here. They belong to the params'
    ``postprocess`` list; see :mod:`nyx.postprocess`.
    """
    ordered = sorted(
        clusters.unique_labels, key=lambda cid: clusters.centers[cid, 0]
    )
    cluster_to_stage: dict[int, str] = {}
    for position, cluster_id in enumerate(ordered):
        if position < len(stage_order):
            cluster_to_stage[int(cluster_id)] = str(stage_order[position])
        else:
            # More clusters than names: leave the extras unnamed rather than
            # silently folding them into the last stage.
            cluster_to_stage[int(cluster_id)] = f"C{int(cluster_id)}"

    if overrides:
        unknown = set(overrides) - set(cluster_to_stage)
        if unknown:
            raise KeyError(
                f"Cannot override cluster(s) {sorted(unknown)}: no such cluster. "
                f"Clusters found: {sorted(cluster_to_stage)}."
            )
        cluster_to_stage.update({int(k): str(v) for k, v in overrides.items()})

    # Stage name per covered epoch. Names rather than integer codes, so that
    # vocabularies beyond WAKE/NREM/REM (NREM1/2/3, or unnamed clusters) survive.
    names_valid = np.full(len(clusters.features_scaled), "NOSIGNAL", dtype="<U16")
    for cluster_id, stage in cluster_to_stage.items():
        names_valid[clusters.labels == cluster_id] = stage

    stage_labels = np.full(len(clusters.valid_mask), "NOSIGNAL", dtype="<U16")
    stage_labels[clusters.valid_mask] = names_valid

    # Integer codes as well, for the single-step path that reconstructs a
    # WAKE/NREM/REM signal over the whole recording.
    stages_valid = np.full(len(clusters.features_scaled), _STAGE_TO_INT["NOSIGNAL"], dtype=int)
    for cluster_id, stage in cluster_to_stage.items():
        stages_valid[clusters.labels == cluster_id] = _STAGE_TO_INT.get(
            stage, _STAGE_TO_INT["NREM"]
        )

    sleep_stages = np.full(len(clusters.valid_mask), _STAGE_TO_INT["NOSIGNAL"], dtype=int)
    sleep_stages[clusters.valid_mask] = stages_valid

    stage_signal = reconstruct_signal_multiclass(
        sleep_stages, wake_sleep.hypnogram, pca.fs, pca.signal.shape[0], within=within
    )
    hypnogram = prepare_epoch_data(stage_signal, pca.fs, state_names=_INT_TO_STAGE)

    return Staging(
        hypnogram=hypnogram,
        cluster_to_stage=cluster_to_stage,
        stage_signal=stage_signal,
        stage_labels=stage_labels,
    )


# ---------------------------------------------------------------------------
# Optional step -- agreement with a manual scoring
# ---------------------------------------------------------------------------


def evaluate(
    hypnogram: dict | Staging,
    reference: dict,
    label_order: Iterable[str] = ("WAKE", "REM", "NREM"),
    window: tuple[float, float] | None = None,
    verbose: bool = True,
) -> Agreement | None:
    """Compare a nyx hypnogram against a manually scored one.

    This step is **optional**: nyx produces a hypnogram without it. Use it only
    when a reference scoring exists to check against.

    Parameters
    ----------
    hypnogram
        A :class:`~nyx.types.Staging` or a raw hypnogram dict.
    reference
        Manual scoring, in the same ``time``/``duration``/``label`` format.
    window
        ``(start, end)`` in seconds. If given, the reference is trimmed to the
        analysis window and shifted to start at zero first.

    Returns
    -------
    Agreement or None
        ``None`` when the two scorings do not overlap at all.
    """
    if isinstance(hypnogram, Staging):
        hypnogram = hypnogram.hypnogram

    if window is not None:
        reference = trim_manual_scores(reference, window[0], window[1])

    label_order = list(label_order)
    results, per_stage, overall = compare_sleep(
        hypnogram,
        reference,
        label_order=label_order,
        plot=False,
        debug=False,
        verbose=verbose,
    )
    if results is None:
        return None

    return Agreement(
        accuracy=results["accuracy"],
        kappa=results["kappa"],
        confusion_matrix=results["confusion_matrix"],
        per_stage=per_stage,
        overall=overall,
        label_order=label_order,
        details=results,
    )


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------


def score_recording(
    recording: Recording,
    params: dict,
    window: tuple[float, float] | None = None,
    reference: dict | None = None,
    emg_threshold: float | None = None,
    nosignal_threshold: float = 0.0,
    stage_order: Sequence[str] = ("REM", "NREM"),
    cluster_overrides: dict[int, str] | None = None,
    return_in_uV: bool = False,
    verbose: bool = True,
) -> ScoringResult:
    """Run the whole pipeline with automatic choices at every decision point.

    Convenience wrapper; call the individual steps when you need to inspect or
    override something partway through.

    Parameters
    ----------
    recording
        The full recording. It is sliced to ``window`` internally.
    params
        Parameter dict with ``"EEG"``, ``"EMG"`` and optionally ``"scoring"``
        and ``"clustering"`` sections.
    window
        ``(start, end)`` seconds to analyse. Defaults to the whole recording.
    reference
        Manual scoring. When given, agreement is computed as a final step;
        when omitted, that step is skipped entirely.
    emg_threshold
        Override the automatic wake/sleep threshold.
    """
    from nyx.preprocessing import preprocess_recording

    recording = preprocess_recording(recording, params)

    total_duration = recording.duration
    start, end = (0.0, total_duration) if window is None else window
    end = min(float(end), total_duration)
    windowed = recording.time_slice(start, end)

    min_duration = float(params.get("scoring", {}).get("min_duration", 4.0))

    if verbose:
        print(f"Analysing {windowed.name or 'recording'}  window {start:g}-{end:g}s")

    emg = compute_emg_features(windowed, params, return_in_uV=return_in_uV)
    wake_sleep = classify_wake_sleep(
        emg,
        threshold=emg_threshold,
        nosignal_threshold=nosignal_threshold,
        min_duration=min_duration,
    )
    if verbose:
        print(
            f"  EMG threshold: {wake_sleep.threshold:.3f} ({wake_sleep.threshold_source})"
        )

    pca = compute_sleep_pca(windowed, params, wake_sleep, return_in_uV=return_in_uV)
    clusters = cluster_sleep(pca, wake_sleep, clustering=params, emg=emg)
    if verbose:
        print(f"  clusters: {clusters.sizes()}")

    staging = assign_stages(
        clusters,
        pca,
        wake_sleep,
        stage_order=stage_order,
        overrides=cluster_overrides,
    )

    rules = rules_from_params(params)
    if rules:
        if verbose:
            print("  postprocessing:")
        staging = replace(
            staging,
            hypnogram=apply_rules(staging.hypnogram, rules, verbose=verbose),
        )

    if verbose:
        durations = staging.stage_durations()
        total = sum(durations.values()) or 1.0
        for stage, seconds in sorted(durations.items()):
            print(f"  {stage:<10} {seconds:9.1f}s ({100 * seconds / total:.1f}%)")

    agreement = None
    trimmed_reference = None
    if reference is not None:
        trimmed_reference = trim_manual_scores(reference, start, end)
        agreement = evaluate(
            staging, reference, window=(start, end), verbose=verbose
        )

    return ScoringResult(
        recording=windowed,
        emg=emg,
        wake_sleep=wake_sleep,
        pca=pca,
        clusters=clusters,
        staging=staging,
        agreement=agreement,
        window=(float(start), float(end)),
        total_duration=float(total_duration),
        params=params,
        reference=trimmed_reference,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_results(
    result: ScoringResult,
    output_dir: str,
    config=None,
    save_emg_power: bool = True,
    steps: list | None = None,
    granularities: Sequence[int] | None = None,
    plots: bool = True,
) -> str:
    """Write the hypnograms, run record and metrics of a run to ``output_dir``.

    Segments outside the analysis window are written as NOSIGNAL, so the saved
    hypnograms always span the full recording.

    The ``run.json`` written here is a self-contained config: loading it with
    :func:`nyx.load_config` and running it reproduces this scoring exactly, with
    no decisions left to make. That is what makes a result re-runnable rather
    than merely documented.

    Parameters
    ----------
    steps
        The steps as run, for a multi-step scoring. Their resolved cluster
        mappings go into the run record.
    granularities
        Extra stage counts to write, e.g. ``[4, 3]`` alongside a 5-stage
        scoring. Written as ``hypnogram_4stage.csv`` and so on.
    """
    from nyx.config import RunConfig, build_run_record

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    start, end = result.window

    run_config = config if isinstance(config, RunConfig) else None
    record = build_run_record(
        run_config,
        result.params,
        window=(start, end),
        thresholds={
            "thresholds": result.wake_sleep.thresholds,
            "source": result.wake_sleep.threshold_source,
        },
        steps=steps,
        recording_name=result.recording.name,
    )
    if steps is None:
        # Single-step scoring: the mapping lives outside the step list.
        record.setdefault("decisions", {})["cluster_to_stage"] = {
            str(k): v for k, v in result.staging.cluster_to_stage.items()
        }
        record.setdefault("decisions", {})["clustering"] = result.clusters.params

    with open(os.path.join(output_dir, "run.json"), "w") as handle:
        json.dump(record, handle, indent=4, default=str)

    if config is not None and run_config is None:
        with open(os.path.join(output_dir, "config.json"), "w") as handle:
            json.dump(config, handle, indent=4, default=str)

    if save_emg_power:
        np.save(os.path.join(output_dir, "emg_power.npy"), result.emg.power)

    save_hypno_with_padding(
        result.wake_sleep.hypnogram,
        os.path.join(output_dir, "wake_sleep.csv"),
        start,
        end,
        result.total_duration,
    )
    save_hypno_with_padding(
        result.staging.hypnogram,
        os.path.join(output_dir, "hypnogram.csv"),
        start,
        end,
        result.total_duration,
    )

    # Coarser hypnograms are merges of the one above, never a second scoring,
    # so they are guaranteed consistent with it.
    for n_stages in granularities or []:
        from nyx.granularity import collapse

        save_hypno_with_padding(
            collapse(result.staging.hypnogram, int(n_stages)),
            os.path.join(output_dir, f"hypnogram_{int(n_stages)}stage.csv"),
            start,
            end,
            result.total_duration,
        )

    if result.agreement is not None:
        with open(os.path.join(output_dir, "agreement.txt"), "w") as handle:
            handle.write(result.agreement.summary())
        with open(os.path.join(output_dir, "agreement.pkl"), "wb") as handle:
            pickle.dump(result.agreement, handle)

    if plots:
        from nyx.report import save_report

        # Plotting must never lose a result that has already been computed.
        try:
            save_report(result, output_dir)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Scoring succeeded but the figures could not be drawn: "
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )

    return output_dir
