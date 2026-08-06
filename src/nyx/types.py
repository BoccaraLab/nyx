"""Data containers passed between pipeline steps.

Every step of the pipeline takes and returns one of these, so that a caller
(a notebook, a script, or a GUI) can run one step, show the result to a user,
let them adjust something, and re-run only the steps downstream of the change.

The canonical hypnogram format used everywhere in nyx is a plain dict::

    {"time": np.ndarray, "duration": np.ndarray, "label": np.ndarray}

one entry per interval, times in seconds relative to the start of the analysis
window. Every annotation reader converts to it and every writer consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # avoid paying the spikeinterface import cost at module load
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from spikeinterface.core import BaseRecording

__all__ = [
    "Recording",
    "EmgFeatures",
    "WakeSleep",
    "SleepPca",
    "SleepClusters",
    "Staging",
    "Agreement",
]


@dataclass
class Recording:
    """One EEG channel, and usually one EMG channel, from a single recording.

    ``eeg`` and ``emg`` are single-channel spikeinterface recordings, which keeps
    the data lazy on disk until a trace is actually requested.

    ``emg`` may be ``None``. Scoring without it is possible but **markedly
    worse**: EMG power is what separates wake from sleep, and without it wake
    has to be recovered from the EEG spectrum alone, where quiet wake and REM
    look much alike. Prefer a surrogate over nothing -- see
    :func:`nyx.emg_from_lfp`, which builds an EMG-like trace out of the
    high-frequency correlation between wideband channels.
    """

    eeg: BaseRecording
    emg: BaseRecording | None
    fs: float
    name: str = ""
    source_path: str = ""
    #: EMG sampling rate, when it differs from the EEG's. Polysomnography files
    #: often store EMG in a separate stream at a different rate (CCSHS, for
    #: instance). ``None`` means the two share :attr:`fs`.
    emg_fs_: float | None = None

    @property
    def has_emg(self) -> bool:
        """Whether this recording carries an EMG channel at all."""
        return self.emg is not None

    @property
    def emg_fs(self) -> float:
        """Sampling rate of the EMG channel."""
        return float(self.emg_fs_) if self.emg_fs_ else float(self.fs)

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return float(self.eeg.get_duration())

    @property
    def eeg_channel_name(self) -> str:
        return self._channel_name(self.eeg)

    @property
    def emg_channel_name(self) -> str:
        return self._channel_name(self.emg) if self.emg is not None else "none"

    @staticmethod
    def _channel_name(rec: BaseRecording) -> str:
        for key in ("channel_name", "channel_names"):
            try:
                values = rec.get_property(key)
            except Exception:
                continue
            if values is not None and len(values) > 0:
                return str(values[0])
        return str(rec.get_channel_ids()[0])

    def eeg_trace(self, return_in_uV: bool = False) -> np.ndarray:
        """EEG as a 1-D float array."""
        return self.eeg.get_traces(return_in_uV=return_in_uV)[:, 0]

    def emg_trace(self, return_in_uV: bool = False) -> np.ndarray:
        """EMG as a 1-D float array. Raises if there is no EMG channel."""
        if self.emg is None:
            raise ValueError(
                f"{self.name or 'This recording'} has no EMG channel, so there is "
                f"no EMG trace to read. Load one with emg_channel=..., or build a "
                f"surrogate with nyx.emg_from_lfp. To score without any EMG at all, "
                f"use a params file whose steps do not include an 'emg_threshold' "
                f"step and whose clustering has use_emg false -- see "
                f"nyx.load_params('mouse_no_emg')."
            )
        return self.emg.get_traces(return_in_uV=return_in_uV)[:, 0]

    def time_slice(self, start_time: float, end_time: float) -> Recording:
        """Return a new Recording restricted to ``[start_time, end_time]`` seconds.

        ``end_time`` is clamped to the time of the last sample. Note that this
        is one sample *before* :attr:`duration` (which is ``n_samples / fs``),
        and asking spikeinterface for the full duration is an error -- so
        clamping is what makes "analyse the whole recording" work.
        """
        start_time = max(0.0, float(start_time))
        last_sample_time = (self.eeg.get_num_samples() - 1) / self.fs
        end_time = min(float(end_time), last_sample_time)

        if start_time >= end_time:
            raise ValueError(
                f"Empty analysis window: start={start_time}s is not before end={end_time}s "
                f"(recording is {self.duration:.1f}s long)."
            )
        if start_time == 0.0 and end_time >= last_sample_time:
            return self  # nothing to trim

        emg = None
        if self.emg is not None:
            # The EMG is sliced by time, so a different sampling rate is fine.
            emg_end = min(end_time, (self.emg.get_num_samples() - 1) / self.emg_fs)
            emg = self.emg.time_slice(start_time, max(emg_end, start_time + 1e-6))

        return Recording(
            eeg=self.eeg.time_slice(start_time, end_time),
            emg=emg,
            fs=self.fs,
            name=self.name,
            source_path=self.source_path,
            emg_fs_=self.emg_fs_,
        )

    def describe(self) -> str:
        """Human-readable summary, useful before choosing channel indices."""
        header = (
            f"{self.name or 'recording'}: {self.duration:.1f}s at {self.fs:g} Hz\n"
            f"  EEG channel: {self.eeg_channel_name}\n"
        )
        if self.emg is None:
            return header + (
                "  EMG channel: none -- wake will have to come out of the EEG "
                "alone, which is much less reliable"
            )
        emg_rate = "" if self.emg_fs == self.fs else f" at {self.emg_fs:g} Hz"
        return header + f"  EMG channel: {self.emg_channel_name}{emg_rate}"


@dataclass
class EmgFeatures:
    """EMG spectrogram collapsed to a smoothed, min-max scaled band-power trace."""

    spectrogram: np.ndarray  # (n_freqs, n_bins)
    freqs: np.ndarray
    times: np.ndarray
    power: np.ndarray  # (n_bins,) scaled to [0, 1]
    fs: float  # sampling rate of `power`, in bins per second


@dataclass
class WakeSleep:
    """Wake / sleep / no-signal segmentation derived from EMG power."""

    hypnogram: dict[str, np.ndarray]
    threshold: float
    nosignal_threshold: float
    threshold_source: str = "auto"  # "auto" or "manual"

    @property
    def thresholds(self) -> list[float]:
        """``[nosignal_threshold, threshold]``, the order the scoring code expects."""
        return [self.nosignal_threshold, self.threshold]


@dataclass
class SleepPca:
    """PCA of the EEG spectrogram, fitted on sleep epochs only."""

    spectrogram: np.ndarray  # (n_freqs, n_bins), band-limited
    freqs: np.ndarray
    times: np.ndarray
    pca: PCA
    scores: np.ndarray  # (n_sleep_bins, n_components), sleep epochs only
    signal: np.ndarray  # (n_bins, n_components), NaN outside sleep
    fs: float  # bins per second

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_


@dataclass
class SleepClusters:
    """Clustering of sleep epochs in PC space."""

    labels: np.ndarray  # per valid sleep epoch; -2 marks outliers
    centers: np.ndarray  # (n_clusters, n_features) in standardised space
    unique_labels: np.ndarray
    features_scaled: np.ndarray
    valid_mask: np.ndarray  # rows of `scores` without NaNs
    outlier_mask: np.ndarray  # within the valid rows
    scaler: StandardScaler
    pcs_to_use: list[int]
    params: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[int, int]:
        """Number of epochs per cluster."""
        return {int(lbl): int((self.labels == lbl).sum()) for lbl in self.unique_labels}


@dataclass
class Staging:
    """Final hypnogram, after clusters have been mapped to sleep stages."""

    hypnogram: dict[str, np.ndarray]
    cluster_to_stage: dict[int, str]
    stage_signal: np.ndarray  # per epoch: -2 NOSIGNAL, -1 WAKE, 0 NREM, 1 REM
    #: Stage name per epoch covered by this step, in order. Unlike
    #: ``stage_signal`` this carries arbitrary stage vocabularies (NREM1/2/3,
    #: or unnamed clusters), so it is what multi-step scoring builds on.
    stage_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype="U"))

    def stage_durations(self) -> dict[str, float]:
        """Total seconds per stage."""
        totals: dict[str, float] = {}
        for duration, label in zip(self.hypnogram["duration"], self.hypnogram["label"]):
            totals[str(label)] = totals.get(str(label), 0.0) + float(duration)
        return totals


@dataclass
class Agreement:
    """Agreement between a nyx hypnogram and a reference (manual) hypnogram."""

    accuracy: float
    kappa: float
    confusion_matrix: np.ndarray
    per_stage: Any  # pandas DataFrame indexed by stage
    overall: Any  # pandas Series, macro (unweighted) averages
    label_order: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def mf1(self) -> float:
        """Macro F1: the unweighted mean of the per-stage F1 scores.

        The headline number. Unweighted on purpose -- weighting by how much of
        the recording each stage occupies lets a dominant stage mask poor
        performance on a rare one, and REM is always a small fraction.
        """
        return float(self.overall["f1-score"])

    def summary(self) -> str:
        lines = [
            "=== Agreement with reference scoring ===",
            f"MF1: {self.mf1:.3f}    Accuracy: {self.accuracy:.3f}    "
            f"Cohen's kappa: {self.kappa:.3f}",
            "",
            self.per_stage.to_string(
                formatters={
                    "precision": "{:.3f}".format,
                    "recall": "{:.3f}".format,
                    "f1-score": "{:.3f}".format,
                    "support": "{:.1f}s".format,
                }
            ),
            "",
            "Macro average (unweighted, over stages present in the reference):",
            f"  precision {self.overall['precision']:.3f}   "
            f"recall {self.overall['recall']:.3f}   "
            f"MF1 {self.mf1:.3f}",
            "",
            "Confusion matrix (rows=reference, cols=nyx, values in seconds):",
        ]
        import pandas as pd

        lines.append(
            pd.DataFrame(
                self.confusion_matrix, index=self.label_order, columns=self.label_order
            )
            .round(1)
            .to_string()
        )
        return "\n".join(lines)
