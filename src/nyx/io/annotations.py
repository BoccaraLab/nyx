"""Readers that turn a scoring file into nyx's canonical hypnogram format.

The canonical format is a dict of three equal-length arrays::

    {"time": start of each interval (s),
     "duration": length of each interval (s),
     "label": stage name}

Readers are keyed by *file format*. Four generic readers cover almost every
manually scored file we have seen:

``interval_csv``
    A CSV with ``time``, ``duration`` and ``label`` columns -- nyx's own output
    format, and the easiest one to convert to.
``epoch_csv``
    One row per fixed-length epoch, with a column holding the stage. Covers
    files that store stages as numbers (1/2/3, 3/4/5, ...) and files whose time
    column is a clock time rather than seconds.
``column_csv``
    One CSV holding several recordings side by side, one column per recording.
``visbrain_hyp``
    The stage/stop ``.hyp`` format written by visbrain and somnotate.

Anything else can be plugged in from user code::

    from nyx.io import register_annotation_reader

    @register_annotation_reader("my_format")
    def read_my_format(path, **kwargs):
        ...
        return {"time": ..., "duration": ..., "label": ...}
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "read_annotations",
    "register_annotation_reader",
    "ANNOTATION_READERS",
    "intervals_from_labels",
    "merge_consecutive",
]

ANNOTATION_READERS: dict[str, Callable[..., dict]] = {}


def register_annotation_reader(name: str) -> Callable:
    """Register a reader under ``name`` so it can be used as ``format=name``."""

    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        ANNOTATION_READERS[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _empty() -> dict[str, np.ndarray]:
    return {
        "time": np.array([], dtype="float64"),
        "duration": np.array([], dtype="float64"),
        "label": np.array([], dtype="U"),
    }


def _apply_label_map(values, label_map: dict | None) -> np.ndarray:
    """Map raw stage codes onto stage names.

    Lookup is tolerant about how the code is spelled: ``3``, ``3.0`` and
    ``"3"`` all match a ``label_map`` key of ``3`` or ``"3"``. This matters
    because stage codes read from CSV may be ints or floats, while a label map
    loaded from JSON can only have string keys.
    """
    if label_map is None:
        return np.array([str(v).strip() for v in values], dtype="U")

    normalised = {_normalise_key(k): v for k, v in label_map.items()}
    return np.array(
        [normalised.get(_normalise_key(v), "UNKNOWN") for v in values], dtype="U"
    )


def _normalise_key(key: Any) -> str:
    """Render a stage code as a canonical string so lookups are format-agnostic."""
    if isinstance(key, (float, np.floating)) and float(key).is_integer():
        return str(int(key))
    if isinstance(key, str):
        stripped = key.strip()
        try:
            as_float = float(stripped)
        except ValueError:
            return stripped
        return str(int(as_float)) if as_float.is_integer() else str(as_float)
    return str(key).strip()


def intervals_from_labels(
    labels, epoch_length: float, start_time: float = 0.0
) -> dict[str, np.ndarray]:
    """Collapse a per-epoch label sequence into variable-length intervals.

    Consecutive epochs sharing a label are merged into a single interval, which
    is what the rest of nyx expects.

    Parameters
    ----------
    labels
        One label per epoch, in chronological order.
    epoch_length
        Duration of a single epoch, in seconds.
    start_time
        Time of the first epoch, in seconds.
    """
    labels = np.asarray(labels, dtype="U")
    if labels.size == 0:
        return _empty()

    merged_labels: list[str] = []
    durations: list[float] = []
    previous = labels[0]
    count = 1
    for label in labels[1:]:
        if label == previous:
            count += 1
        else:
            merged_labels.append(previous)
            durations.append(count * epoch_length)
            previous = label
            count = 1
    merged_labels.append(previous)
    durations.append(count * epoch_length)

    times = start_time + np.cumsum([0.0] + durations[:-1])
    return {
        "time": np.asarray(times, dtype="float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(merged_labels, dtype="U"),
    }


def merge_consecutive(hypnogram: dict) -> dict[str, np.ndarray]:
    """Merge adjacent intervals that share a label, keeping start times intact."""
    times = np.asarray(hypnogram["time"], dtype="float64")
    durations = np.asarray(hypnogram["duration"], dtype="float64")
    labels = np.asarray(hypnogram["label"], dtype="U")

    if times.size == 0:
        return _empty()

    out_times = [float(times[0])]
    out_durations = [float(durations[0])]
    out_labels = [str(labels[0])]
    for time, duration, label in zip(times[1:], durations[1:], labels[1:]):
        if str(label) == out_labels[-1]:
            out_durations[-1] += float(duration)
        else:
            out_times.append(float(time))
            out_durations.append(float(duration))
            out_labels.append(str(label))

    return {
        "time": np.asarray(out_times, dtype="float64"),
        "duration": np.asarray(out_durations, dtype="float64"),
        "label": np.asarray(out_labels, dtype="U"),
    }


def _timestamps_to_seconds(values, time_format: str) -> np.ndarray:
    """Convert a time column to seconds since the first entry."""
    if time_format == "seconds":
        seconds = pd.to_numeric(pd.Series(values)).to_numpy(dtype="float64")
    elif time_format == "datetime":
        parsed = pd.to_datetime(pd.Series(values))
        # Subtract before converting, so the result does not depend on whether
        # pandas stored the timestamps as ns, us or ms.
        if parsed.empty:
            return np.array([], dtype="float64")
        seconds = (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy(dtype="float64")
    elif time_format == "clock":
        seconds = np.array([_clock_to_seconds(v) for v in values], dtype="float64")
        seconds = _unwrap_midnight(seconds)
    else:
        raise ValueError(
            f"Unknown time_format {time_format!r}. "
            f"Expected 'seconds', 'datetime' or 'clock'."
        )
    return seconds - seconds[0] if seconds.size else seconds


def _unwrap_midnight(seconds: np.ndarray) -> np.ndarray:
    """Make wall-clock times monotonic across midnight.

    A scoring file that stores the time of day as ``HH:MM:SS`` restarts at
    ``00:00:00`` partway through any recording that runs overnight. Without
    this, every epoch after midnight is placed ~24 h too early and silently
    stops overlapping the recording.
    """
    if seconds.size < 2:
        return seconds
    wrapped = np.diff(seconds) < 0
    return seconds + np.cumsum(np.r_[0.0, wrapped]) * 86400.0


def _clock_to_seconds(value: Any) -> float:
    """Parse ``HH:MM:SS`` or ``MM:SS`` into seconds."""
    parts = str(value).strip().split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0]
    return 0.0


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


@register_annotation_reader("interval_csv")
def _read_interval_csv(
    path: str,
    time_column: str = "time",
    duration_column: str = "duration",
    stage_column: str = "label",
    label_map: dict | None = None,
    sep: str = ",",
    names: list[str] | None = None,
    header: int | None = "infer",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read nyx's own hypnogram format: one row per interval.

    ``sep`` and ``names`` cover headerless variants -- e.g. a tab-separated
    ``label<TAB>time<TAB>duration`` file is
    ``names=["label", "time", "duration"], sep="\\t", header=None``.
    """
    if names is not None and header == "infer":
        header = None
    df = pd.read_csv(path, sep=sep, names=names, header=header)
    missing = [c for c in (time_column, duration_column, stage_column) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path!r} is missing the column(s) {missing}. Found: {list(df.columns)}. "
            f"If your columns are named differently, pass time_column=, "
            f"duration_column= and stage_column=."
        )
    return {
        "time": df[time_column].to_numpy(dtype="float64"),
        "duration": df[duration_column].to_numpy(dtype="float64"),
        "label": _apply_label_map(df[stage_column].to_numpy(), label_map),
    }


@register_annotation_reader("epoch_csv")
def _read_epoch_csv(
    path: str,
    stage_column: str | int | None = None,
    epoch_length: float | None = None,
    time_column: str | None = None,
    duration_column: str | None = None,
    time_format: str = "seconds",
    label_map: dict | None = None,
    header: int | None = 0,
    index_col: int | None = None,
    sep: str = ",",
    names: list[str] | None = None,
    merge: bool = True,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read a file with one row per scored epoch.

    Parameters
    ----------
    stage_column
        Column holding the stage, by name or position. Defaults to the first
        column that is not the time or duration column.
    epoch_length
        Seconds per epoch. Optional if the file has a time or duration column,
        in which case it is inferred.
    time_column, time_format
        Column holding the epoch start time, and how to parse it: ``"seconds"``,
        ``"datetime"`` (a date/time string) or ``"clock"`` (``HH:MM:SS`` or
        ``MM:SS``). Times are made relative to the first epoch.
    duration_column
        Per-row duration, if the file has one. Otherwise ``epoch_length`` is used.
    label_map
        Maps the raw stage codes onto stage names, e.g.
        ``{3: "NREM", 4: "REM", 5: "WAKE"}``. Unmapped codes become ``"UNKNOWN"``.
    merge
        Merge consecutive epochs that share a label (default). Turn this off to
        keep one interval per row.
    """
    df = pd.read_csv(path, header=header, index_col=index_col, sep=sep, names=names)
    if df.empty:
        return _empty()

    stage_col = _pick_stage_column(df, stage_column, time_column, duration_column, path)
    stages = df[stage_col]

    # Rows with no stage carry no information.
    keep = stages.notna()
    df, stages = df[keep], stages[keep]
    if df.empty:
        return _empty()

    labels = _apply_label_map(stages.to_numpy(), label_map)

    times = None
    if time_column is not None:
        if time_column not in df.columns:
            raise KeyError(
                f"{path!r} has no column {time_column!r}. Found: {list(df.columns)}."
            )
        times = _timestamps_to_seconds(df[time_column].to_numpy(), time_format)

    if duration_column is not None:
        if duration_column not in df.columns:
            raise KeyError(
                f"{path!r} has no column {duration_column!r}. Found: {list(df.columns)}."
            )
        durations = df[duration_column].to_numpy(dtype="float64")
    else:
        epoch_length = _resolve_epoch_length(epoch_length, times, path)
        durations = np.full(len(df), float(epoch_length), dtype="float64")

    if times is None:
        # No time column: epochs are contiguous from t=0.
        epoch_length = _resolve_epoch_length(epoch_length, None, path)
        return (
            intervals_from_labels(labels, float(epoch_length))
            if merge
            else {
                "time": np.arange(len(labels), dtype="float64") * float(epoch_length),
                "duration": durations,
                "label": labels,
            }
        )

    hypnogram = {"time": times, "duration": durations, "label": labels}
    return merge_consecutive(hypnogram) if merge else hypnogram


def _pick_stage_column(df, stage_column, time_column, duration_column, path):
    if stage_column is None:
        candidates = [c for c in df.columns if c not in (time_column, duration_column)]
        if not candidates:
            raise KeyError(f"{path!r} has no column that could hold the stage.")
        return candidates[0]
    if isinstance(stage_column, (int, np.integer)):
        return df.columns[int(stage_column)]
    if stage_column not in df.columns:
        raise KeyError(
            f"{path!r} has no column {stage_column!r}. Found: {list(df.columns)}."
        )
    return stage_column


def _resolve_epoch_length(epoch_length, times, path) -> float:
    if epoch_length is not None:
        return float(epoch_length)
    if times is not None and len(times) > 1:
        step = float(times[1] - times[0])
        if step > 0:
            return step
    raise ValueError(
        f"Cannot determine the epoch length for {path!r}. Pass epoch_length=<seconds>, "
        f"or give a time_column/duration_column it can be inferred from."
    )


@register_annotation_reader("column_csv")
def _read_column_csv(
    path: str,
    column: str | None = None,
    epoch_length: float = 4.0,
    label_map: dict | None = None,
    header: int | None = 1,
    index_col: int | None = 0,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read one recording out of a CSV that holds several side by side.

    Each column is one recording and each row is one epoch.

    Parameters
    ----------
    column
        Which column to read. Required -- without it there is no way to know
        which recording the scores belong to.
    """
    if column is None:
        raise ValueError(
            f"{path!r} holds several recordings side by side, so column= is required "
            f"(usually the recording name)."
        )

    df = pd.read_csv(path, header=header, index_col=index_col)
    # Column headers in these files are often quoted; normalise before matching.
    df.columns = [str(c).strip().strip("'").strip('"') for c in df.columns]

    key = str(column).strip().strip("'").strip('"')
    if key not in df.columns:
        raise KeyError(
            f"{path!r} has no column {column!r}. Available columns: {list(df.columns)}."
        )

    stages = df[key].dropna()
    labels = _apply_label_map(stages.to_numpy(), label_map)
    return intervals_from_labels(labels, float(epoch_length))


@register_annotation_reader("epoch_npy")
def _read_epoch_npy(
    path: str,
    epoch_length: float = 30.0,
    label_map: dict | None = None,
    key: str = "label",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read one stage code per fixed-length epoch from a ``.npy`` file.

    Handles both a bare array and a pickled dict holding the array under
    ``key``. Used by the Dreem Open Datasets (dodh / dodo), whose default
    coding is ``{-1: L, 0: W, 1: N1, 2: N2, 3: N3, 4: R}``.
    """
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    codes = data[key] if isinstance(data, dict) else data

    labels = _apply_label_map(np.asarray(codes).ravel(), label_map)
    return intervals_from_labels(labels, float(epoch_length))


@register_annotation_reader("epoch_mat")
def _read_epoch_mat(
    path: str,
    struct: str = "stageData",
    stage_field: str = "stages",
    epoch_field: str | None = "win",
    epoch_length: float | None = None,
    label_map: dict | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read a hypnogram out of a MATLAB ``.mat`` struct.

    Parameters
    ----------
    struct, stage_field
        Name of the struct and of the field holding one stage code per epoch.
    epoch_field
        Field holding the epoch length in seconds. Falls back to
        ``epoch_length`` if absent.
    """
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if struct not in mat:
        available = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"{path!r} has no struct {struct!r}. Found: {available}.")
    data = mat[struct]

    codes = np.atleast_1d(getattr(data, stage_field)).astype(int)

    seconds = epoch_length
    if epoch_field is not None and hasattr(data, epoch_field):
        seconds = float(np.atleast_1d(getattr(data, epoch_field)).flat[0])
    if not seconds:
        raise ValueError(
            f"Cannot determine the epoch length for {path!r}: field {epoch_field!r} "
            f"is missing and epoch_length was not given."
        )

    labels = _apply_label_map(codes, label_map)
    return intervals_from_labels(labels, float(seconds))


# The stage vocabulary used in NSRR XML annotation files. Stage 4 is folded into
# N3 to follow AASM.
NSRR_CONCEPTS = {
    "Wake|0": "WAKE",
    "Stage 1 sleep|1": "NREM1",
    "Stage 2 sleep|2": "NREM2",
    "Stage 3 sleep|3": "NREM3",
    "Stage 4 sleep|4": "NREM3",
    "REM sleep|5": "REM",
}


@register_annotation_reader("nsrr_xml")
def _read_nsrr_xml(
    path: str,
    label_map: dict | None = None,
    event_type: str = "Stages",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read an NSRR ``*-nsrr.xml`` annotation file.

    This is the National Sleep Research Resource's public annotation format,
    shared by MESA, CHAT, CCSHS, SHHS and others -- so it is worth having as a
    built-in rather than per dataset.

    Only ``ScoredEvent`` entries whose ``EventType`` contains ``event_type``
    are read; everything else (arousals, desaturations, respiratory events) is
    ignored.
    """
    import xml.etree.ElementTree as ET

    concepts = NSRR_CONCEPTS if label_map is None else label_map

    root = ET.parse(path).getroot()
    times, durations, labels = [], [], []
    for event in root.iter("ScoredEvent"):
        if event_type not in (event.findtext("EventType", "") or ""):
            continue
        concept = event.findtext("EventConcept", "") or ""
        times.append(float(event.findtext("Start", "0") or 0.0))
        durations.append(float(event.findtext("Duration", "0") or 0.0))
        labels.append(concepts.get(concept, "UNKNOWN"))

    if not labels:
        raise ValueError(
            f"No {event_type!r} events found in {path!r}. Check that this is an NSRR "
            f"annotation file and that event_type matches its EventType values."
        )

    order = np.argsort(times)
    return merge_consecutive(
        {
            "time": np.asarray(times, dtype="float64")[order],
            "duration": np.asarray(durations, dtype="float64")[order],
            "label": np.asarray(labels, dtype="U")[order],
        }
    )


@register_annotation_reader("visbrain_hyp")
def _read_visbrain_hyp(path: str, **kwargs) -> dict[str, np.ndarray]:
    """Read a visbrain/somnotate ``.hyp`` file (stage + interval end time).

    See http://visbrain.org/sleep.html#save-hypnogram. Adapted from
    https://github.com/paulbrodersen/somnotate/blob/master/example_pipeline/data_io.py
    """
    dtype = [("Stage", "|S30"), ("stop", float)]
    data = np.genfromtxt(path, skip_header=2, dtype=dtype, delimiter="\t")
    data = np.atleast_1d(data)
    if data.size == 0:
        return _empty()

    states = [state.astype(str).strip() for state in data["Stage"]]
    transitions = np.r_[0, data["stop"]]
    return {
        "time": np.asarray(transitions[:-1], dtype="float64"),
        "duration": np.asarray(transitions[1:] - transitions[:-1], dtype="float64"),
        "label": np.asarray(states, dtype="U"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_annotations(path: str, format: str = "auto", **kwargs) -> dict[str, np.ndarray]:
    """Load a manually scored hypnogram.

    Parameters
    ----------
    path
        Scoring file to read.
    format
        One of :data:`ANNOTATION_READERS`, or ``"auto"`` to guess from the file
        extension (``.hyp`` -> ``visbrain_hyp``, ``.csv``/``.txt`` ->
        ``interval_csv`` when the file has time/duration/label columns and
        ``epoch_csv`` otherwise).
    **kwargs
        Passed to the reader, e.g. ``label_map``, ``epoch_length``,
        ``stage_column``.

    Returns
    -------
    dict
        ``{"time": ..., "duration": ..., "label": ...}``
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if format == "auto":
        format = _infer_annotation_format(path)

    if format not in ANNOTATION_READERS:
        raise ValueError(
            f"Unknown annotation format {format!r}. "
            f"Available: {sorted(ANNOTATION_READERS)}. "
            f"Register your own with nyx.io.register_annotation_reader."
        )

    return ANNOTATION_READERS[format](path, **kwargs)


def _infer_annotation_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hyp":
        return "visbrain_hyp"
    if ext in (".csv", ".txt", ".tsv"):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            return "epoch_csv"
        if {"time", "duration", "label"} <= columns:
            return "interval_csv"
        return "epoch_csv"
    raise ValueError(
        f"Cannot infer the annotation format of {path!r} from its extension ({ext!r}). "
        f"Pass format= explicitly, one of: {sorted(ANNOTATION_READERS)}."
    )
