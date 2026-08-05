"""Clustering steps, applied in sequence to refine a labelling.

A scoring run is an ordered list of :class:`Step`. Each one takes the epochs
that currently carry a given label, fits a PCA *inside* those epochs, clusters
them, and replaces that label with finer ones. Which species you are scoring is
a matter of how many steps you run and what you call the clusters -- not of
different code.

Rodent, standard::

    emg_threshold  ->  WAKE / SLEEP
    Step("split_sleep", within="SLEEP", method="hdbscan")   -> NREM / REM

Rodent, when the EMG does not separate cleanly (or the recording is short) --
set the EMG threshold permissively so everything lands in SLEEP, then::

    Step("split_all", within="SLEEP", method="gmm", n_clusters=3, use_emg=True)
        -> WAKE / NREM / REM

Human::

    Step("wake_sleep",  within="SLEEP", method="elliptic", use_emg=True)
        -> WAKE / SLEEP
    Step("split_wake",  within="WAKE",  method="kmeans")     -> WAKE / NREM1
    Step("split_sleep", within="SLEEP", method="hdbscan", n_clusters=4,
         refinements=[...])                                  -> NREM2/NREM3/REM

Mapping every cluster of a step to the same stage makes that step a no-op,
which is the natural way to decide -- after looking at the result -- that a
split was not worth keeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nyx.io.annotations import intervals_from_labels

__all__ = [
    "Step",
    "Refinement",
    "StepOutcome",
    "run_step",
    "run_steps",
    "rename_clusters",
    "steps_from_params",
    "epoch_labels",
]

# Widest label we store on the epoch grid.
_LABEL_DTYPE = "<U16"


@dataclass
class Refinement:
    """Split one output stage in two by thresholding a principal component.

    Used where a clustering leaves two stages merged that a single component
    does separate -- N1 from N2, typically.
    """

    split_stage: str  # stage to divide
    pc: int  # index into the step's PCs
    threshold: float
    high: str  # label for scores above the threshold
    low: str  # label for scores at or below it


@dataclass
class Step:
    """One clustering step."""

    name: str
    within: str = "SLEEP"
    method: str = "hdbscan"
    pcs_to_use: list[int] = field(default_factory=lambda: [0, 1])
    use_emg: bool = False
    n_clusters: int = 2
    outlier_z_threshold: float | None = None
    #: Cluster id -> stage. Omit to name clusters by ``stage_order`` instead.
    cluster_to_stage: dict[int, str] | None = None
    #: Stage names ordered by centroid on the first PC used, lowest first.
    stage_order: list[str] | None = None
    refinements: list[Refinement] = field(default_factory=list)
    #: Method-specific settings passed through to the clustering routine.
    options: dict[str, Any] = field(default_factory=dict)

    def clustering_params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "pcs_to_use": list(self.pcs_to_use),
            "use_emg": self.use_emg,
            "outlier_z_threshold": self.outlier_z_threshold,
            **self.options,
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> Step:
        """Build a Step from its JSON form.

        ``n_pcs: 4`` is accepted as shorthand for ``pcs_to_use: [0, 1, 2, 3]``.
        Any key the Step does not define is passed through to the clustering
        routine as an option, so method-specific settings need no schema change.
        """
        # Keys starting with _ are comments. JSON has none of its own, and a
        # params file with a dozen tuning knobs badly needs them.
        spec = {k: v for k, v in spec.items() if not k.startswith("_")}
        known = {f for f in cls.__dataclass_fields__ if f != "options"}

        if "pcs_to_use" not in spec:
            n_pcs = spec.pop("n_pcs", None)
            if n_pcs is not None:
                spec["pcs_to_use"] = list(range(int(n_pcs)))
        spec.pop("n_pcs", None)
        spec.pop("pc_indices", None)  # legacy alias, always null in the old files

        refinements = [
            r if isinstance(r, Refinement) else Refinement(**r)
            for r in spec.pop("refinements", [])
        ]
        mapping = spec.pop("cluster_to_stage", None)
        if mapping is not None:
            mapping = {int(k): str(v) for k, v in mapping.items()}

        options = {**spec.pop("options", {}),
                   **{k: spec.pop(k) for k in list(spec) if k not in known}}

        if "name" not in spec:
            raise KeyError(f"A step needs a 'name'. Got keys: {sorted(spec)}.")

        return cls(
            **spec,
            cluster_to_stage=mapping,
            refinements=refinements,
            options=options,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form, suitable for writing into a run record."""
        out: dict[str, Any] = {
            "name": self.name,
            "within": self.within,
            "method": self.method,
            "pcs_to_use": list(self.pcs_to_use),
            "use_emg": self.use_emg,
            "n_clusters": self.n_clusters,
        }
        if self.outlier_z_threshold is not None:
            out["outlier_z_threshold"] = self.outlier_z_threshold
        if self.stage_order:
            out["stage_order"] = list(self.stage_order)
        if self.cluster_to_stage:
            out["cluster_to_stage"] = {str(k): v for k, v in self.cluster_to_stage.items()}
        if self.refinements:
            out["refinements"] = [vars(r) for r in self.refinements]
        if self.options:
            out["options"] = dict(self.options)
        return out


def steps_from_params(params: dict) -> list[Step]:
    """Read the ``steps`` list out of a params dict."""
    return [Step.from_dict(spec) for spec in params.get("steps", [])]


@dataclass
class StepOutcome:
    """What one step produced."""

    step: Step
    hypnogram: dict[str, np.ndarray]
    labels: np.ndarray  # per-epoch labels after this step
    pca: Any
    clusters: Any
    cluster_to_stage: dict[int, str]
    #: The labelling this step was run on. Kept so :func:`rename_clusters` can
    #: rebuild the output without re-running anything.
    input_hypnogram: dict[str, np.ndarray] | None = None

    def summary(self) -> str:
        sizes = self.clusters.sizes()
        total = max(sum(sizes.values()), 1)
        lines = [
            f"step {self.step.name!r}: split {self.step.within!r} with {self.step.method}"
        ]
        for cid, stage in sorted(self.cluster_to_stage.items()):
            count = sizes.get(cid, 0)
            lines.append(
                f"    C{cid} -> {stage:<8} {count:>7} epochs ({100 * count / total:.1f}%)"
            )
        return "\n".join(lines)


def epoch_labels(hypnogram: dict, fs: float, n_epochs: int) -> np.ndarray:
    """Expand an interval hypnogram onto a per-epoch label array.

    Uses truncation rather than rounding when converting times to indices, to
    match how the PCA selects its epochs.
    """
    labels = np.full(n_epochs, "NOSIGNAL", dtype=_LABEL_DTYPE)
    for time, duration, label in zip(
        hypnogram["time"], hypnogram["duration"], hypnogram["label"]
    ):
        start = max(0, min(int(float(time) * fs), n_epochs))
        end = max(0, min(int((float(time) + float(duration)) * fs), n_epochs))
        if end > start:
            labels[start:end] = str(label)
    return labels


def run_step(
    recording,
    params: dict,
    hypnogram: dict,
    step: Step,
    emg=None,
    return_in_uV: bool = False,
) -> StepOutcome:
    """Run one clustering step and return the refined labelling.

    Parameters
    ----------
    recording
        The windowed recording.
    params
        Full parameter dict (``EEG`` / ``EMG`` / ``scoring`` sections).
    hypnogram
        The labelling so far. Only epochs labelled ``step.within`` are touched.
    step
        What to do.
    emg
        Required when ``step.use_emg`` is set.
    """
    from nyx.pipeline import assign_stages, cluster_sleep, compute_sleep_pca
    from nyx.types import WakeSleep

    scope = WakeSleep(hypnogram=hypnogram, threshold=0.0, nosignal_threshold=0.0)

    pca = compute_sleep_pca(
        recording, params, scope, return_in_uV=return_in_uV, within=step.within
    )
    clusters = cluster_sleep(pca, scope, clustering=step.clustering_params(), emg=emg)

    return _label(step, hypnogram, pca, clusters)


def _label(step: Step, hypnogram: dict, pca, clusters) -> StepOutcome:
    """Name the clusters and write the result back onto the epoch grid.

    Split out of :func:`run_step` so :func:`rename_clusters` can redo it without
    recomputing the PCA or the clustering, which is the expensive part.
    """
    from nyx.pipeline import assign_stages
    from nyx.types import WakeSleep

    scope = WakeSleep(hypnogram=hypnogram, threshold=0.0, nosignal_threshold=0.0)

    staging = assign_stages(
        clusters,
        pca,
        scope,
        stage_order=step.stage_order or _default_stage_order(step),
        overrides=step.cluster_to_stage,
        within=step.within,
    )

    # The PCA's non-NaN rows are exactly the epochs this step covered, in order,
    # which is what lets the new labels be written back without re-deriving the
    # selection.
    selected = ~np.any(np.isnan(pca.signal), axis=1)
    if int(selected.sum()) != len(staging.stage_labels):
        raise RuntimeError(
            f"Step {step.name!r} produced {len(staging.stage_labels)} labels for "
            f"{int(selected.sum())} epochs. This is a bug in the step engine, not in "
            f"your configuration."
        )

    labels = epoch_labels(hypnogram, pca.fs, pca.signal.shape[0])
    new_labels = np.asarray(staging.stage_labels, dtype=_LABEL_DTYPE)

    if step.refinements:
        new_labels = _apply_refinements(new_labels, pca, clusters, step)

    labels[selected] = new_labels
    refined = intervals_from_labels(labels, 1.0 / pca.fs)

    return StepOutcome(
        step=step,
        hypnogram=refined,
        labels=labels,
        pca=pca,
        clusters=clusters,
        cluster_to_stage=staging.cluster_to_stage,
        input_hypnogram=hypnogram,
    )


def rename_clusters(outcome: StepOutcome, stage_order=None,
                    cluster_to_stage=None, refinements=None) -> StepOutcome:
    """Give a finished step's clusters different names, without re-clustering.

    Naming clusters is a decision you make *after* seeing the per-cluster
    spectra, and it is normal to get it wrong first time. Editing the
    :class:`Step` and calling :func:`run_step` again would recompute the PCA and
    the clustering to reach the same clusters -- minutes, on a night of data,
    for a relabelling that changes nothing about them.

    Pass ``stage_order`` to name clusters by position, ``cluster_to_stage`` to
    name them by id, or both -- ids are applied over positions, as in
    :func:`nyx.assign_stages`.

        out = run_step(recording, params, hypnogram, step, emg=emg)
        nyx.plot_cluster_check(out.pca, clusters=out.clusters)   # ...look...
        out = nyx.rename_clusters(out, stage_order=["NREM", "REM"])

    Returns a new outcome; the original is untouched.
    """
    if outcome.input_hypnogram is None:
        raise ValueError(
            "This outcome does not carry the labelling it was run on, so it "
            "cannot be relabelled. It was probably built by hand rather than by "
            "run_step; pass input_hypnogram when constructing it."
        )

    from dataclasses import replace as _replace

    changes = {}
    if stage_order is not None:
        changes["stage_order"] = list(stage_order)
    if cluster_to_stage is not None:
        changes["cluster_to_stage"] = {int(k): str(v) for k, v in cluster_to_stage.items()}
    if refinements is not None:
        changes["refinements"] = list(refinements)

    step = _replace(outcome.step, **changes)
    return _label(step, outcome.input_hypnogram, outcome.pca, outcome.clusters)


def run_steps(
    recording,
    params: dict,
    hypnogram: dict,
    steps: list[Step],
    emg=None,
    return_in_uV: bool = False,
    verbose: bool = True,
) -> list[StepOutcome]:
    """Run a sequence of steps, each refining the labelling left by the last.

    Returns one outcome per step; the final hypnogram is
    ``outcomes[-1].hypnogram``.
    """
    outcomes: list[StepOutcome] = []
    current = hypnogram
    for step in steps:
        outcome = run_step(
            recording, params, current, step, emg=emg, return_in_uV=return_in_uV
        )
        if verbose:
            print(outcome.summary())
        outcomes.append(outcome)
        current = outcome.hypnogram
    return outcomes


def _default_stage_order(step: Step) -> list[str]:
    """Placeholder names when a step has no mapping yet.

    Leaving clusters unnamed is the honest default for a species whose stages
    you do not know in advance -- inspect the PSDs and the quality metrics,
    then set ``cluster_to_stage``.
    """
    return [f"{step.within}_C{i}" for i in range(max(step.n_clusters, 2))]


def _apply_refinements(
    labels: np.ndarray, pca, clusters, step: Step
) -> np.ndarray:
    """Threshold-split stages after the clustering has been mapped."""
    labels = labels.copy()
    scores = pca.scores  # one row per covered epoch

    for refinement in step.refinements:
        target = labels == refinement.split_stage
        if not target.any():
            continue
        if refinement.pc >= scores.shape[1]:
            raise ValueError(
                f"Refinement on step {step.name!r} asks for PC{refinement.pc} but the "
                f"PCA has {scores.shape[1]} components."
            )
        values = scores[:, refinement.pc]
        labels[target & (values > refinement.threshold)] = refinement.high
        labels[target & (values <= refinement.threshold)] = refinement.low

    return labels
