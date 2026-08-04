"""Configuration: what to load, how to process it, and what a run actually did.

Three layers, separated by how often each changes:

**params** (``params/*.json``) -- reusable per species or setup. Spectral
settings, normalisation, notch, and the *structure* of the scoring steps
(methods, how many clusters, which components). No recording-specific values,
so a params file can be shared as a sensible default.

**config** (``config.json``) -- one recording. Where the files are, which
channels, which params, where results go. Or just a dataset name and a
recording name, and a recipe fills the rest in.

**run record** (``<output>/run.json``) -- written by a run. Everything above
*plus* the decisions made along the way: the analysis window, the EMG
thresholds, each step's cluster-to-stage mapping, each refinement threshold.

The run record is itself a valid config, which is what makes a result
re-runnable: feed its ``run.json`` back in and you get the same scoring, with
no interactive steps.

Paths inside a config are resolved relative to the config file, so a config can
live next to the data it points at.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RecordingSpec",
    "AnnotationSpec",
    "OutputSpec",
    "RunConfig",
    "load_config",
    "load_json",
    "load_params",
    "validate_params",
    "build_run_record",
]

#: Sections a params file must have.
REQUIRED_PARAM_SECTIONS = ("EEG", "EMG")

#: Feature backends. Only the spectrogram is implemented; the scalogram keys are
#: recognised so that a params file using it fails with a clear message rather
#: than a missing-key error deep inside the feature code.
FEATURE_METHODS = ("spectrogram", "scalogram")
_SCALOGRAM_KEYS = {"freq_resolution", "f0", "exp_corr"}


def load_json(path: str) -> dict[str, Any]:
    """Read a JSON file, with a readable error if it is missing or malformed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path} (looked in {os.getcwd()})")
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON: {exc}. A common cause on Windows is a "
            f"single backslash in a path -- write \"C:\\\\data\", not \"C:\\data\"."
        ) from exc


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class RecordingSpec:
    """Where the signal is and which channels to use."""

    path: str
    format: str = "auto"
    eeg_channel: int | str = 0
    emg_channel: int | str = 1
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: dict, base: str = "") -> RecordingSpec:
        spec = dict(spec)
        known = {"path", "format", "eeg_channel", "emg_channel", "options"}
        options = {**spec.pop("options", {}),
                   **{k: spec.pop(k) for k in list(spec) if k not in known}}
        if "path" not in spec:
            raise KeyError("The 'recording' section needs a 'path'.")
        spec["path"] = _resolve(spec["path"], base)
        return cls(**spec, options=options)

    def to_dict(self) -> dict[str, Any]:
        out = {"path": self.path, "format": self.format,
               "eeg_channel": self.eeg_channel, "emg_channel": self.emg_channel}
        if self.options:
            out["options"] = dict(self.options)
        return out


@dataclass
class AnnotationSpec:
    """Where a reference scoring is, if there is one.

    Entirely optional: nyx produces a hypnogram without it, and the agreement
    step is simply skipped.
    """

    path: str | None = None
    format: str = "auto"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: dict, base: str = "") -> AnnotationSpec:
        spec = dict(spec)
        known = {"path", "format", "options"}
        options = {**spec.pop("options", {}),
                   **{k: spec.pop(k) for k in list(spec) if k not in known}}
        if spec.get("path"):
            spec["path"] = _resolve(spec["path"], base)
        return cls(**spec, options=options)

    def to_dict(self) -> dict[str, Any]:
        out = {"path": self.path, "format": self.format}
        if self.options:
            out["options"] = dict(self.options)
        return out


@dataclass
class OutputSpec:
    """Where results go and at what granularities."""

    root: str = "results"
    name: str | None = None
    #: Stage counts to write. Empty means "whatever the scoring resolved".
    granularities: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, spec: dict, base: str = "") -> OutputSpec:
        spec = dict(spec)
        if "root" in spec:
            spec["root"] = _resolve(spec["root"], base)
        return cls(**spec)

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "name": self.name,
                "granularities": list(self.granularities)}


# ---------------------------------------------------------------------------
# The config itself
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """One recording's worth of configuration."""

    recording: RecordingSpec | None = None
    annotations: AnnotationSpec | None = None
    #: Alternative to `recording`/`annotations`: name a dataset recipe.
    dataset: str | None = None
    dataset_options: dict[str, Any] = field(default_factory=dict)

    params_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    output: OutputSpec = field(default_factory=OutputSpec)

    #: Analysis window in seconds; None means the whole recording.
    window: tuple[float, float] | None = None
    #: Decisions recorded by a previous run, replayed when present.
    decisions: dict[str, Any] = field(default_factory=dict)

    source_path: str = ""

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, spec: dict, base: str = "", source_path: str = "") -> RunConfig:
        spec = dict(spec)

        recording = spec.pop("recording", None)
        annotations = spec.pop("annotations", None)
        dataset = spec.pop("dataset", None)

        if recording is None and dataset is None:
            raise KeyError(
                "A config needs either a 'recording' section (path + channels) or a "
                "'dataset' name. Available datasets: see nyx.datasets.list_datasets()."
            )

        params_path = spec.pop("params", None)
        params = {}
        if isinstance(params_path, dict):
            params, params_path = params_path, None  # params given inline
        elif params_path:
            params_path = _resolve(params_path, base)
            params = load_json(params_path)

        window = spec.pop("window", None)
        if window is not None:
            window = (float(window[0]), float(window[1]))

        # Anything left that is not a known key becomes a dataset option, so a
        # recipe config reads naturally: {"dataset": ..., "data_root": ...}.
        known = {"output", "decisions", "dataset_options"}
        dataset_options = {**spec.pop("dataset_options", {}),
                           **{k: spec.pop(k) for k in list(spec) if k not in known}}
        if "data_root" in dataset_options:
            dataset_options["data_root"] = _resolve(dataset_options["data_root"], base)

        return cls(
            recording=(RecordingSpec.from_dict(recording, base) if recording else None),
            annotations=(
                AnnotationSpec.from_dict(annotations, base) if annotations else None
            ),
            dataset=dataset,
            dataset_options=dataset_options,
            params_path=params_path,
            params=params,
            output=OutputSpec.from_dict(spec.pop("output", {}), base),
            window=window,
            decisions=spec.pop("decisions", {}),
            source_path=source_path,
        )

    def load(self, with_annotations: bool = True):
        """Load the recording and, if configured, the reference scoring."""
        from nyx.io import read_annotations, read_recording

        if self.dataset:
            from nyx.datasets import load_dataset

            return load_dataset(
                self.dataset, with_annotations=with_annotations, **self.dataset_options
            )

        recording = read_recording(
            self.recording.path,
            format=self.recording.format,
            eeg_channel=self.recording.eeg_channel,
            emg_channel=self.recording.emg_channel,
            name=self.name,
            **self.recording.options,
        )

        reference = None
        if with_annotations and self.annotations and self.annotations.path:
            reference = read_annotations(
                self.annotations.path,
                format=self.annotations.format,
                **self.annotations.options,
            )
        return recording, reference

    def steps(self):
        """The scoring steps declared by the params."""
        from nyx.steps import steps_from_params

        return steps_from_params(self.params)

    # -- naming ------------------------------------------------------------

    @property
    def name(self) -> str:
        """Label for this recording, used for the output folder."""
        if self.output.name:
            return self.output.name
        if self.dataset_options.get("recording_name"):
            return str(self.dataset_options["recording_name"])
        if self.recording:
            return os.path.splitext(os.path.basename(self.recording.path.rstrip("/\\")))[0]
        return "recording"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output.root, self.name)

    def to_dict(self) -> dict[str, Any]:
        """JSON form. Together with `params` this is the run record."""
        out: dict[str, Any] = {}
        if self.dataset:
            out["dataset"] = self.dataset
            out.update(self.dataset_options)
        if self.recording:
            out["recording"] = self.recording.to_dict()
        if self.annotations:
            out["annotations"] = self.annotations.to_dict()
        out["params"] = self.params_path or self.params
        out["output"] = self.output.to_dict()
        if self.window:
            out["window"] = list(self.window)
        if self.decisions:
            out["decisions"] = self.decisions
        return out


def load_config(path: str) -> RunConfig:
    """Read a config file, resolving its paths relative to the file itself."""
    spec = load_json(path)
    return RunConfig.from_dict(
        spec, base=os.path.dirname(os.path.abspath(path)), source_path=path
    )


def _resolve(path: str, base: str) -> str:
    """Make a config-relative path absolute."""
    if not base or not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def build_run_record(
    config: RunConfig | None = None,
    params: dict | None = None,
    *,
    window: tuple[float, float] | None = None,
    thresholds: dict[str, Any] | None = None,
    steps: list | None = None,
    recording_name: str = "",
) -> dict[str, Any]:
    """Assemble the record of what a run actually did.

    The result is a *self-contained config*: params are written inline rather
    than as a path, and every interactive decision is baked in -- the analysis
    window, the EMG thresholds, and each step's cluster-to-stage mapping and
    refinement thresholds. Loading it with :func:`load_config` and running it
    reproduces the scoring with nothing left to decide.

    Parameters
    ----------
    config
        The config the run started from, if there was one.
    params
        The parameters used. Steps are replaced with their resolved form.
    window
        The analysis window actually used.
    thresholds
        EMG threshold information, as recorded by the wake/sleep step.
    steps
        The steps as run, after their cluster mappings were decided. Accepts
        `Step` objects or `StepOutcome`s.
    """
    from nyx import __version__

    record: dict[str, Any] = {}
    if config is not None:
        record.update(config.to_dict())

    resolved = dict(params or (config.params if config else {}))
    if steps:
        resolved["steps"] = [_step_dict(step) for step in steps]
    # Inline, not a path: a record that points at a params file is only as
    # reproducible as that file.
    record["params"] = resolved

    if window is not None:
        record["window"] = [float(window[0]), float(window[1])]

    decisions: dict[str, Any] = dict(record.get("decisions", {}))
    if thresholds:
        decisions["emg"] = thresholds
    if decisions:
        record["decisions"] = decisions

    record["nyx_version"] = __version__
    if recording_name:
        record.setdefault("output", {})["name"] = recording_name
    return record


def _step_dict(step) -> dict[str, Any]:
    """Accept either a Step or a StepOutcome and return the resolved step."""
    if hasattr(step, "step"):  # StepOutcome
        resolved = step.step.to_dict()
        resolved["cluster_to_stage"] = {
            str(k): v for k, v in step.cluster_to_stage.items()
        }
        return resolved
    return step.to_dict()


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def load_params(path: str) -> dict[str, Any]:
    """Read a parameter file and check it has the sections the pipeline needs."""
    params = load_json(path)
    validate_params(params, source=path)
    return params


def validate_params(params: dict, source: str = "<params>") -> None:
    """Raise if a parameter dict cannot drive the pipeline."""
    missing = [key for key in REQUIRED_PARAM_SECTIONS if key not in params]
    if missing:
        raise KeyError(
            f"{source} is missing the {missing} section(s). A parameter file needs "
            f"at least 'EEG' and 'EMG'; 'scoring' and 'steps' are optional."
        )

    method = params.get("features", {}).get("method", "spectrogram")
    if method not in FEATURE_METHODS:
        raise ValueError(
            f"{source}: unknown features.method {method!r}. Expected one of "
            f"{list(FEATURE_METHODS)}."
        )
    if method == "scalogram":
        raise NotImplementedError(
            f"{source} asks for the scalogram backend, which is not implemented yet. "
            f"Use features.method = 'spectrogram'."
        )

    # Catch a scalogram params file that predates the explicit `features` key.
    for section in ("EEG", "EMG"):
        keys = {k for k in params.get(section, {}) if not k.startswith("_")}
        if _SCALOGRAM_KEYS & keys and "binsize" not in keys:
            raise NotImplementedError(
                f"{source}: the {section} section has scalogram settings "
                f"({sorted(_SCALOGRAM_KEYS & keys)}) but no 'binsize'. The scalogram "
                f"backend is not implemented yet -- convert this file to spectrogram "
                f"settings, or wait for scalogram support."
            )

    # Postprocessing rules: catch a typo here rather than after the scoring has
    # run, which on a long recording is many minutes later.
    from nyx.postprocess import RULES, _as_call

    for i, rule in enumerate(params.get("postprocess") or []):
        try:
            name, _ = _as_call(rule)
        except ValueError as exc:
            raise ValueError(f"{source}: postprocess[{i}]: {exc}") from exc
        if name not in RULES:
            raise ValueError(
                f"{source}: postprocess[{i}] names an unknown rule {name!r}. "
                f"Available: {sorted(RULES)}."
            )

    for i, step in enumerate(params.get("steps", [])):
        if "name" not in step:
            raise KeyError(f"{source}: steps[{i}] has no 'name'.")
        if "method" not in step:
            raise KeyError(f"{source}: step {step['name']!r} has no 'method'.")
