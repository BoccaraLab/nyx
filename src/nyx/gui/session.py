"""The scoring pipeline, held open partway through.

A notebook runs the pipeline top to bottom and keeps whatever it needs in
variables. A GUI cannot: the user moves back and forth, changes one thing, and
expects everything downstream to follow and everything upstream to survive.
This module is that bookkeeping, and nothing else -- it drives nyx's own
functions and adds no analysis of its own.

**It must not import Qt.** Not PySide, not pyqtgraph, not ephyviewer, directly
or transitively. That is what makes the pipeline state testable without a
display, and it is checked by a test. The Qt layer observes this object; this
object knows nothing about it.

The cost gradient is the whole design
-------------------------------------

============  ==========================================  ==============
stage         produced by                                 cost
============  ==========================================  ==============
LOAD          ``read_recording``                          I/O
PREPROCESS    ``preprocess_recording`` + ``time_slice``   free (lazy)
EMG           ``compute_emg_features``                    moderate
WAKE_SLEEP    ``classify_wake_sleep``                     **instant**
STEPS         ``compute_sleep_pca`` + ``cluster_sleep``   **minutes**
              + ``label_clusters``                        (the last: instant)
RESULT        ``evaluate``                                instant
============  ==========================================  ==============

Changing something at stage *N* drops the cache from *N* down and leaves
everything above it alone. Three cases carry their weight:

* moving the EMG threshold keeps the :class:`~nyx.types.EmgFeatures`, so a
  dragged threshold re-scores in milliseconds;
* changing a clustering setting keeps the :class:`~nyx.types.SleepPca`, which
  is the expensive one;
* renaming clusters keeps the clustering itself, *by identity* -- this is
  :func:`nyx.rename_clusters`' reason for existing, one level down.

Accessors never compute
-----------------------

:meth:`ScoringSession.pca` raises :class:`NotComputed` rather than quietly
computing. If accessors computed, a repaint or a tooltip could start a
five-minute PCA on the UI thread. Only :meth:`ScoringSession.compute` computes,
and the GUI only ever calls it from a worker.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import IntEnum
from typing import Any

import numpy as np

from nyx.config import RecordingSpec, RunConfig, validate_params
from nyx.pipeline import (
    ScoringResult,
    classify_wake_sleep,
    cluster_sleep,
    compute_emg_features,
    compute_sleep_pca,
    evaluate,
    find_wake_sleep_threshold,
    resolve_clustering_params,
)
from nyx.postprocess import apply_rules, rules_from_params
from nyx.preprocessing import preprocess_recording
from nyx.steps import Step, StepOutcome, label_clusters
from nyx.types import Recording, Staging, WakeSleep

__all__ = ["Stage", "NotComputed", "ScoringSession"]


class Stage(IntEnum):
    """Where a value sits in the chain. Order is the dependency order."""

    LOAD = 0
    PREPROCESS = 1
    EMG = 2
    WAKE_SLEEP = 3
    STEPS = 4
    RESULT = 5


class NotComputed(RuntimeError):
    """Asked for something that has not been computed yet.

    Carries the stage, so a caller can decide whether to run it or wait.
    """

    def __init__(self, stage: Stage, detail: str = ""):
        self.stage = stage
        super().__init__(
            f"{stage.name} has not been computed yet"
            f"{': ' + detail if detail else ''}. "
            f"Call session.compute_through(Stage.{stage.name}) first."
        )


class ScoringSession:
    """One recording, scored interactively, with everything cached.

    Setters record a decision and invalidate downstream. :meth:`compute` does
    the work. Accessors hand back what is already there, or raise.
    """

    def __init__(self, params: dict | None = None):
        self._params: dict = {}
        self._recording: Recording | None = None
        self._reference: dict | None = None
        self._window: tuple[float, float] | None = None

        self._emg_threshold: float | None = None
        self._nosignal_threshold: float | None = None
        self._min_duration: float | None = None

        self._steps: list[Step] = []
        self._postprocess: list | None = None
        #: step index -> {position in the clustered epochs: stage}. Hand
        #: assignments, applied after the cluster names.
        self._manual_epochs: dict[int, dict[int, str]] = {}
        self._edited_hypnogram: dict | None = None
        self._edit_note: dict[str, Any] = {}

        # Caches. Everything here is dropped by invalidate().
        self._windowed: Recording | None = None
        self._emg: Any = None
        self._emg_done = False
        self._auto_threshold: float | None = None
        self._wake_sleep: WakeSleep | None = None
        self._pcas: list[Any] = []
        self._clusters: list[Any] = []
        self._outcomes: list[StepOutcome] = []

        self._generation = 0
        self._listeners: list[Callable[[Stage], None]] = []
        self._warnings: list[str] = []

        if params is not None:
            self.set_params(params)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: RunConfig) -> ScoringSession:
        """Start from a config or a previous run's ``run.json``.

        A run record *is* a config, so resuming an interactive session and
        reproducing a finished one are the same operation.
        """
        session = cls(config.params or None)
        recording, reference = config.load()
        session.set_recording(recording, reference=reference)
        if config.window:
            session.set_window(config.window)

        decisions = config.decisions or {}
        if decisions.get("emg_threshold") is not None:
            session.set_emg_threshold(
                float(decisions["emg_threshold"]),
                nosignal=decisions.get("nosignal_threshold"),
            )
        session._config = config
        return session

    # -- observation -------------------------------------------------------

    def add_listener(self, listener: Callable[[Stage], None]) -> None:
        """Call ``listener(stage)`` when *stage* and everything after changes.

        One event, one signature. Listeners run on whichever thread made the
        change, which for the GUI means a worker thread -- so the Qt adapter
        has to re-emit rather than touch widgets. See :mod:`nyx.gui.mainwindow`.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Stage], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, stage: Stage) -> None:
        for listener in list(self._listeners):
            listener(stage)

    def generation(self) -> int:
        """Bumped by every change.

        A worker carries the generation it started under; if it no longer
        matches, its result is stale and the caller drops it.
        """
        return self._generation

    # -- inputs ------------------------------------------------------------

    def set_params(self, params: dict) -> None:
        validate_params(params, source="params")
        self._params = dict(params)
        self._steps = _clustering_steps_of(self._params)
        self._postprocess = None
        self.invalidate(Stage.PREPROCESS)

    def update_params(self, section: str, **values) -> None:
        """Change keys in one params section, e.g. ``update_params("EMG", notch=50)``."""
        params = dict(self._params)
        params[section] = {**params.get(section, {}), **values}
        self.set_params(params)

    def set_recording(self, recording: Recording, *, reference: dict | None = None) -> None:
        self._recording = recording
        if reference is not None:
            self._reference = reference
        self.invalidate(Stage.PREPROCESS)

    def set_reference(self, reference: dict | None) -> None:
        self._reference = reference
        self.invalidate(Stage.RESULT)

    def set_window(self, window: tuple[float, float] | None) -> None:
        self._window = (
            None if window is None else (float(window[0]), float(window[1]))
        )
        self.invalidate(Stage.PREPROCESS)

    def set_emg_threshold(
        self, threshold: float | None, nosignal: float | None = None
    ) -> None:
        """Set the wake/sleep cut. ``None`` returns to the automatic one."""
        self._emg_threshold = None if threshold is None else float(threshold)
        if nosignal is not None:
            self._nosignal_threshold = float(nosignal)
        self.invalidate(Stage.WAKE_SLEEP)

    def set_min_duration(self, seconds: float) -> None:
        self._min_duration = float(seconds)
        self.invalidate(Stage.WAKE_SLEEP)

    def set_clustering(self, settings: dict, step: int = -1) -> None:
        """Change one step's clustering. Keeps that step's PCA."""
        index = self._step_index(step)
        known = set(Step.__dataclass_fields__)
        fields = {k: v for k, v in settings.items() if k in known and k != "options"}
        options = {k: v for k, v in settings.items() if k not in known}
        current = dict(self._steps[index].options)
        current.update(options)
        self._steps[index] = replace(self._steps[index], **fields, options=current)
        self._invalidate_step(index, keep_pca=True)

    def set_stage_order(self, order: Sequence[str] | None, step: int = -1) -> None:
        """Name a step's clusters by position. Instant -- nothing is re-clustered."""
        index = self._step_index(step)
        self._steps[index] = replace(
            self._steps[index], stage_order=None if order is None else list(order)
        )
        self._invalidate_step(index, keep_pca=True, keep_clusters=True)

    def set_cluster_overrides(
        self, overrides: dict[int, str] | None, step: int = -1
    ) -> None:
        """Name a step's clusters by id. Applied over ``stage_order``."""
        index = self._step_index(step)
        self._steps[index] = replace(
            self._steps[index],
            cluster_to_stage=(
                None if not overrides else {int(k): str(v) for k, v in overrides.items()}
            ),
        )
        self._invalidate_step(index, keep_pca=True, keep_clusters=True)

    def assign_epochs(self, mask, stage: str, step: int = -1) -> int:
        """Name individual epochs by hand, overriding what clustering decided.

        ``mask`` is over the clustered epochs, in the order
        :attr:`SleepClusters.features_scaled` holds them -- which is what a
        polygon drawn on the cluster scatter selects.

        This is the escape hatch for the cases clustering will not get on its
        own: a REM cluster that merged into NREM, an artefact lobe that should
        be NOSIGNAL. It is applied *after* the cluster names, so it survives
        renaming, and it is recorded in the run config, since a scoring a human
        edited must not claim to be automatic.

        Returns how many epochs it touched.
        """
        import numpy as np

        index = self._step_index(step)
        mask = np.asarray(mask, dtype=bool)

        overrides = dict(self._manual_epochs.get(index, {}))
        for position in np.nonzero(mask)[0]:
            overrides[int(position)] = str(stage)
        self._manual_epochs[index] = overrides

        self._invalidate_step(index, keep_pca=True, keep_clusters=True)
        return int(mask.sum())

    def clear_manual_epochs(self, step: int | None = None) -> None:
        """Forget hand assignments, for one step or all of them."""
        if step is None:
            self._manual_epochs = {}
        else:
            self._manual_epochs.pop(self._step_index(step), None)
        if self._steps:
            self._invalidate_step(0, keep_pca=True, keep_clusters=True)

    def manual_epochs(self, step: int = -1) -> dict[int, str]:
        return dict(self._manual_epochs.get(self._step_index(step), {}))

    def set_refinements(self, refinements: Sequence | None, step: int = -1) -> None:
        index = self._step_index(step)
        self._steps[index] = replace(
            self._steps[index], refinements=list(refinements or [])
        )
        self._invalidate_step(index, keep_pca=True, keep_clusters=True)

    def set_postprocess(self, rules: Sequence | None) -> None:
        """Override the params' rule list. ``None`` goes back to the params."""
        self._postprocess = None if rules is None else list(rules)
        self.invalidate(Stage.RESULT)

    def set_hypnogram(self, hypnogram: dict, *, source: str = "manual") -> None:
        """Replace the scored hypnogram, typically after editing it by hand.

        The edit wins over everything the pipeline produced, and is recorded in
        the run config so a saved ``run.json`` cannot claim to replay a scoring
        that a human changed.
        """
        from datetime import datetime, timezone

        previous = None
        try:
            previous = self.staging().hypnogram
        except NotComputed:
            pass

        self._edited_hypnogram = hypnogram
        self._edit_note = {
            "edited": True,
            "source": source,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if previous is not None:
            self._edit_note["segments_before"] = int(len(previous["label"]))
            self._edit_note["segments_after"] = int(len(hypnogram["label"]))
        self.invalidate(Stage.RESULT)

    # -- steps -------------------------------------------------------------

    @property
    def steps(self) -> list[Step]:
        """The clustering steps, in order. One for a standard rodent run."""
        return list(self._steps)

    def _step_index(self, step: int) -> int:
        if not self._steps:
            raise ValueError(
                "These params declare no clustering steps, so there is nothing "
                "to configure. Add a 'steps' entry with a clustering method."
            )
        return range(len(self._steps))[step]

    # -- invalidation ------------------------------------------------------

    def invalidate(self, stage: Stage) -> None:
        """Drop everything from ``stage`` down, and say so."""
        self._generation += 1

        if stage <= Stage.PREPROCESS:
            self._windowed = None
        if stage <= Stage.EMG:
            self._emg, self._emg_done, self._auto_threshold = None, False, None
        if stage <= Stage.WAKE_SLEEP:
            self._wake_sleep = None
        if stage <= Stage.STEPS:
            self._pcas, self._clusters, self._outcomes = [], [], []
            # A hand edit sits on top of the last step's output, so re-running
            # the steps discards it -- but a change at RESULT (a different
            # reference, different rules) must not, or setting the edited
            # hypnogram would immediately erase it.
            self._edited_hypnogram = None
            self._edit_note = {}

        self._notify(stage)

    def _invalidate_step(
        self, index: int, *, keep_pca: bool = False, keep_clusters: bool = False
    ) -> None:
        """Drop step *index* onwards, optionally keeping its expensive parts.

        Later steps always go entirely: each one fits its PCA inside the labels
        the previous step produced, so changing even the *names* of step k's
        clusters changes what step k+1 is looking at.
        """
        self._generation += 1
        keep_pca_to = index + 1 if keep_pca else index
        keep_clusters_to = index + 1 if keep_clusters else index

        self._pcas = self._pcas[:keep_pca_to]
        self._clusters = self._clusters[:keep_clusters_to]
        self._outcomes = self._outcomes[:index]
        self._edited_hypnogram = None
        self._edit_note = {}
        self._notify(Stage.STEPS)

    # -- state -------------------------------------------------------------

    def has(self, stage: Stage) -> bool:
        """Whether ``stage`` is computed and current."""
        return {
            Stage.LOAD: self._recording is not None,
            Stage.PREPROCESS: self._windowed is not None,
            Stage.EMG: self._emg_done,
            Stage.WAKE_SLEEP: self._wake_sleep is not None,
            Stage.STEPS: bool(self._steps) and len(self._outcomes) == len(self._steps),
            Stage.RESULT: bool(self._steps) and len(self._outcomes) == len(self._steps),
        }[stage]

    def ready(self) -> Stage:
        """The furthest stage that is computed and current."""
        furthest = Stage.LOAD if self._recording is not None else Stage.LOAD
        for stage in Stage:
            if self.has(stage):
                furthest = stage
            else:
                break
        return furthest

    def steps_done(self) -> int:
        """How many clustering steps have finished."""
        return len(self._outcomes)

    # -- accessors ---------------------------------------------------------

    @property
    def params(self) -> dict:
        return self._params

    @property
    def recording(self) -> Recording:
        if self._recording is None:
            raise NotComputed(Stage.LOAD, "no recording has been loaded")
        return self._recording

    @property
    def reference(self) -> dict | None:
        return self._reference

    @property
    def window(self) -> tuple[float, float]:
        """The analysis window, resolved against the recording's duration."""
        total = self.recording.duration
        if self._window is None:
            return (0.0, float(total))
        start, end = self._window
        return (float(start), float(min(end, total)))

    def windowed(self) -> Recording:
        if self._windowed is None:
            raise NotComputed(Stage.PREPROCESS)
        return self._windowed

    def emg(self):
        """The EMG features, or ``None`` when the recording has no EMG."""
        if not self._emg_done:
            raise NotComputed(Stage.EMG)
        return self._emg

    def auto_threshold(self) -> float:
        if self._auto_threshold is None:
            raise NotComputed(Stage.EMG, "no automatic threshold was fitted")
        return self._auto_threshold

    def emg_threshold(self) -> float | None:
        """The threshold in force, automatic one included once it is known."""
        if self._emg_threshold is not None:
            return self._emg_threshold
        declared = _declared_thresholds(self._params)[0]
        if declared is not None:
            return float(declared)
        return self._auto_threshold

    def nosignal_threshold(self) -> float:
        if self._nosignal_threshold is not None:
            return self._nosignal_threshold
        return float(_declared_thresholds(self._params)[1] or 0.0)

    def min_duration(self) -> float:
        if self._min_duration is not None:
            return self._min_duration
        return float(self._params.get("scoring", {}).get("min_duration", 4.0))

    def wake_sleep(self) -> WakeSleep:
        if self._wake_sleep is None:
            raise NotComputed(Stage.WAKE_SLEEP)
        return self._wake_sleep

    def pca(self, step: int = -1):
        index = self._step_index(step)
        if index >= len(self._pcas):
            raise NotComputed(Stage.STEPS, f"step {index} has no PCA")
        return self._pcas[index]

    def clusters(self, step: int = -1):
        index = self._step_index(step)
        if index >= len(self._clusters):
            raise NotComputed(Stage.STEPS, f"step {index} has not been clustered")
        return self._clusters[index]

    def outcome(self, step: int = -1) -> StepOutcome:
        index = self._step_index(step)
        if index >= len(self._outcomes):
            raise NotComputed(Stage.STEPS, f"step {index} has not been labelled")
        return self._outcomes[index]

    def outcomes(self) -> list[StepOutcome]:
        return list(self._outcomes)

    def cluster_to_stage(self, step: int = -1) -> dict[int, str]:
        return dict(self.outcome(step).cluster_to_stage)

    def cluster_table(self, step: int = -1):
        """Centroids and stage names, one row per cluster, ordered."""
        from nyx.report import cluster_ordering

        return cluster_ordering(self.clusters(step), self.cluster_to_stage(step))

    def staging(self) -> Staging:
        """The final labelling: last step, postprocessed, manual edits applied."""
        if not self.has(Stage.STEPS):
            raise NotComputed(Stage.STEPS)
        last = self._outcomes[-1]

        hypnogram = last.hypnogram
        if self._edited_hypnogram is not None:
            hypnogram = self._edited_hypnogram
        else:
            rules = self.postprocess_rules()
            if rules:
                hypnogram = apply_rules(hypnogram, rules)

        return Staging(
            hypnogram=hypnogram,
            cluster_to_stage=last.cluster_to_stage,
            # Five stages do not fit the WAKE/NREM/REM integer codes, so the
            # multi-step path leaves this empty, as score_recording does.
            stage_signal=np.array([]),
            stage_labels=last.labels,
        )

    def postprocess_rules(self) -> list:
        if self._postprocess is not None:
            return list(self._postprocess)
        return rules_from_params(self._params)

    def was_edited(self) -> bool:
        return self._edited_hypnogram is not None

    def agreement(self):
        """Agreement against the reference, or ``None`` when there is none."""
        if self._reference is None:
            return None
        start, end = self.window
        return evaluate(
            self.staging(), self._reference, window=(start, end), verbose=False
        )

    def result(self) -> ScoringResult:
        """Everything, bundled the way :func:`nyx.score_recording` bundles it."""
        from nyx.metrics import trim_manual_scores

        start, end = self.window
        reference = (
            None if self._reference is None
            else trim_manual_scores(self._reference, start, end)
        )
        return ScoringResult(
            recording=self.windowed(),
            emg=self.emg(),
            wake_sleep=self.wake_sleep(),
            pca=self.pca(),
            clusters=self.clusters(),
            staging=self.staging(),
            agreement=self.agreement(),
            window=(start, end),
            total_duration=float(self.recording.duration),
            params=self._params,
            reference=reference,
            steps=self.outcomes(),
        )

    def warnings(self) -> list[str]:
        """Warnings raised by the last :meth:`compute`, oldest first."""
        return list(self._warnings)

    # -- computation -------------------------------------------------------

    def compute(self, stage: Stage) -> None:
        """Compute exactly one stage. Blocking; safe off the UI thread.

        Raises :class:`NotComputed` if the stage above it is missing, rather
        than quietly computing it -- the caller decides what runs.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._compute(stage)
        self._warnings = [str(w.message) for w in caught]
        for message in self._warnings:
            warnings.warn(message, stacklevel=2)

    def _compute(self, stage: Stage) -> None:
        if stage is Stage.PREPROCESS:
            start, end = self.window
            processed = preprocess_recording(self.recording, self._params)
            self._windowed = processed.time_slice(start, end)

        elif stage is Stage.EMG:
            self._emg = compute_emg_features(self.windowed(), self._params)
            self._emg_done = True
            # Fitted once, here, rather than on every threshold change: it is
            # four GaussianMixtures selected by BIC over the whole power
            # vector, which is seconds on a night -- not something to run
            # behind a slider.
            self._auto_threshold = (
                None if self._emg is None
                else float(
                    find_wake_sleep_threshold(
                        self._emg, nosignal_threshold=self.nosignal_threshold()
                    )
                )
            )

        elif stage is Stage.WAKE_SLEEP:
            self._wake_sleep = self._classify()

        elif stage is Stage.STEPS:
            self._run_steps()

        elif stage in (Stage.LOAD, Stage.RESULT):
            pass  # nothing to compute: LOAD is set, RESULT is assembled on demand

        self._notify(stage)

    def _classify(self) -> WakeSleep:
        from nyx.pipeline import _everything_is_sleep

        emg = self.emg()
        if emg is None:
            # No EMG to threshold, so everything starts as SLEEP and the
            # clustering steps have to find wake in the EEG themselves.
            warnings.warn(
                "Scoring without an EMG channel. Wake has to be recovered from "
                "the EEG spectrum alone, where quiet wake and REM look much "
                "alike, so expect noticeably worse agreement -- REM especially. "
                "If the file has other wideband channels, nyx.emg_from_lfp "
                "builds a surrogate from them.",
                stacklevel=2,
            )
            return _everything_is_sleep(self.windowed(), self._params)

        return classify_wake_sleep(
            emg,
            threshold=self.emg_threshold(),
            nosignal_threshold=self.nosignal_threshold(),
            min_duration=self.min_duration(),
        )

    def _run_steps(self) -> None:
        """Run whatever part of the step chain is missing, and no more."""
        if not self._steps:
            raise ValueError(
                "These params declare no clustering steps, so there is nothing "
                "to score. Add a 'steps' entry with a clustering method -- see "
                "nyx.load_params('mouse')."
            )

        hypnogram = self.wake_sleep().hypnogram
        emg = self.emg()

        for index, step in enumerate(self._steps):
            if index < len(self._outcomes):
                hypnogram = self._outcomes[index].hypnogram
                continue

            scope = WakeSleep(hypnogram=hypnogram, threshold=0.0, nosignal_threshold=0.0)

            if index >= len(self._pcas):
                self._pcas.append(
                    compute_sleep_pca(
                        self.windowed(), self._params, scope, within=step.within
                    )
                )
            if index >= len(self._clusters):
                self._clusters.append(
                    cluster_sleep(
                        self._pcas[index], scope,
                        clustering=step.clustering_params(), emg=emg,
                    )
                )

            outcome = label_clusters(
                step, hypnogram, self._pcas[index], self._clusters[index]
            )
            outcome = self._apply_manual(index, outcome)
            self._outcomes.append(outcome)
            hypnogram = outcome.hypnogram

    def _apply_manual(self, index: int, outcome: StepOutcome) -> StepOutcome:
        """Overwrite the epochs a human named, after the clusters were named.

        The mask the caller gave is over the *clustered* epochs; the outcome's
        labels are over the whole epoch grid. The PCA's non-NaN rows are
        exactly the epochs this step covered, in order, which is the same
        correspondence :func:`nyx.steps.label_clusters` relies on to write its
        own labels back.
        """
        overrides = self._manual_epochs.get(index)
        if not overrides:
            return outcome

        from nyx.io.annotations import intervals_from_labels

        pca = self._pcas[index]
        covered = np.nonzero(~np.any(np.isnan(pca.signal), axis=1))[0]
        valid = self._clusters[index].valid_mask
        selectable = covered[np.asarray(valid, dtype=bool)] if len(valid) else covered

        labels = np.array(outcome.labels, copy=True)
        for position, stage in overrides.items():
            if 0 <= position < len(selectable):
                labels[selectable[position]] = stage

        return replace(
            outcome,
            labels=labels,
            hypnogram=intervals_from_labels(labels, 1.0 / pca.fs),
        )

    def compute_through(
        self, stage: Stage, should_stop: Callable[[], bool] | None = None
    ) -> None:
        """Compute every missing stage up to and including ``stage``.

        ``should_stop`` is checked *between* stages. It cannot interrupt one:
        ``compute_sleep_pca`` is a straight line into scipy and sklearn with
        nothing to poll, and Python cannot preempt a thread. Abandoning a run
        therefore means ignoring its result, not stopping it.
        """
        for step in Stage:
            if step > stage:
                break
            if should_stop is not None and should_stop():
                return
            if not self.has(step):
                self._compute_wrapped(step)

    def _compute_wrapped(self, stage: Stage) -> None:
        self.compute(stage)

    # -- output ------------------------------------------------------------

    def to_run_config(self) -> RunConfig:
        """A config recording every decision, for ``run.json``.

        A hand edit is recorded too. Without that, a saved record would claim
        to replay a scoring it cannot: re-running it reproduces the automatic
        result, not the one a human corrected.
        """
        config = getattr(self, "_config", None) or RunConfig()
        decisions = dict(config.decisions or {})

        # Without a recording section the record cannot be loaded at all, so a
        # session that was not started from a config has to describe what it
        # opened. A demo recording has no path and simply has none.
        recording_spec = config.recording
        if recording_spec is None and self._recording is not None:
            source = self._recording.source_path
            if source:
                # The format and the reader's options too: a Neo file, or an
                # EDF with its EMG in a second stream, cannot be read back
                # from a path and two channel names alone.
                recording_spec = RecordingSpec(
                    path=source,
                    format=self._recording.source_format or "auto",
                    eeg_channel=self._recording.eeg_channel_name,
                    emg_channel=(
                        self._recording.emg_channel_name
                        if self._recording.has_emg else None
                    ),
                    options=dict(self._recording.source_options),
                )

        start, end = self.window
        decisions["window"] = [start, end]
        if self._emg is not None and self._wake_sleep is not None:
            decisions["emg_threshold"] = float(self._wake_sleep.threshold)
            decisions["nosignal_threshold"] = float(self.nosignal_threshold())
        if self._edit_note:
            decisions["manual_edit"] = dict(self._edit_note)
        if any(self._manual_epochs.values()):
            decisions["manual_epochs"] = {
                str(step): {str(k): v for k, v in overrides.items()}
                for step, overrides in self._manual_epochs.items()
                if overrides
            }

        return replace(
            config,
            recording=recording_spec,
            params=self._params,
            window=(start, end),
            decisions=decisions,
        )

    def save(
        self,
        output_dir: str,
        *,
        plots: bool = True,
        granularities: Sequence[int] | None = None,
        save_emg_power: bool = True,
    ) -> str:
        """Write the result out, exactly as :func:`nyx.save_results` would."""
        from nyx.pipeline import save_results

        return save_results(
            self.result(),
            output_dir,
            config=self.to_run_config(),
            save_emg_power=save_emg_power,
            granularities=granularities,
            plots=plots,
        )


# ---------------------------------------------------------------------------
# Reading decisions out of a params dict
# ---------------------------------------------------------------------------


def _clustering_steps_of(params: dict) -> list[Step]:
    """The clustering steps, or one synthesised from the clustering section.

    A params file normally declares its steps. One that does not still has a
    ``clustering`` section, and ``score_recording`` falls back to a single
    NREM/REM split -- so the same fallback is made explicit here rather than
    leaving the session unable to score a file the library can.
    """
    from nyx.pipeline import _clustering_steps

    declared = _clustering_steps(params)
    if declared:
        return list(declared)

    settings = resolve_clustering_params(params)
    settings.pop("canonical_cluster_ids", None)
    known = set(Step.__dataclass_fields__)
    return [
        Step(
            name="split_sleep",
            within="SLEEP",
            stage_order=["REM", "NREM"],
            **{k: v for k, v in settings.items() if k in known},
            options={k: v for k, v in settings.items() if k not in known},
        )
    ]


def _declared_thresholds(params: dict) -> tuple[float | None, float | None]:
    """``(threshold, nosignal)`` from an ``emg_threshold`` step, if there is one.

    human params set a deliberately permissive threshold so that everything
    lands in SLEEP and a later step splits wake off; fitting one automatically
    there would quietly do the wrong thing.
    """
    from nyx.pipeline import _emg_threshold_step

    declared = _emg_threshold_step(params)
    return declared.get("threshold"), declared.get("nosignal_threshold")
