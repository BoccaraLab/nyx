"""Layouts and reader settings for the datasets used in the nyx paper.

Each recipe is a small function mapping a config dict onto a :class:`DatasetSpec`.
Adding a dataset here is optional -- it only buys you a short name instead of
spelling out paths and reader options every time.

``params`` on each spec records which parameter file was used for the published
analysis, so a result can be reproduced without guessing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nyx.io.annotations import read_annotations
from nyx.io.recordings import read_recording
from nyx.types import Recording

__all__ = [
    "DatasetSpec",
    "DATASETS",
    "list_datasets",
    "describe_dataset",
    "resolve_dataset",
    "load_dataset",
]


@dataclass
class DatasetSpec:
    """Everything needed to load one recording of a known dataset."""

    recording_path: str
    recording_format: str = "auto"
    eeg_channel: int | str = 0
    emg_channel: int | str = 1
    recording_kwargs: dict[str, Any] = field(default_factory=dict)

    annotation_path: str | None = None
    annotation_format: str = "auto"
    annotation_kwargs: dict[str, Any] = field(default_factory=dict)

    params: str | None = None
    source: str = ""


DATASETS: dict[str, Callable[[dict], DatasetSpec]] = {}


def _register(name: str) -> Callable:
    def decorator(func: Callable[[dict], DatasetSpec]) -> Callable[[dict], DatasetSpec]:
        DATASETS[name] = func
        return func

    return decorator


def _require(cfg: dict, *keys: str) -> tuple:
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise KeyError(f"This dataset needs {missing} in the config. Given: {sorted(cfg)}.")
    return tuple(cfg[k] for k in keys)


def _check_exists(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recording file not found at: {path}")
    return path


# ---------------------------------------------------------------------------
# Oxford mouse benchmark (somnotate)
# ---------------------------------------------------------------------------


@_register("oxford_mouse_benchmark_dataset")
def _oxford_mouse(cfg: dict) -> DatasetSpec:
    data_root, dataset, sub_dataset, recording_name = _require(
        cfg, "data_root", "dataset", "sub_dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset, sub_dataset)
    return DatasetSpec(
        recording_path=_check_exists(
            os.path.join(base, "recordings", f"{recording_name}.edf")
        ),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=os.path.join(
            base, "annotations", _oxford_annotation_filename(sub_dataset, recording_name)
        ),
        annotation_format="visbrain_hyp",
        params="mouse",
        source="https://doi.org/10.5281/zenodo.10200482",
    )


def _oxford_annotation_filename(sub_dataset: str, recording_name: str) -> str:
    """The annotation filename is derived from the recording name, per sub-dataset."""
    if sub_dataset == "test":
        return f"{recording_name}_consensus_state_annotation.hyp"
    if sub_dataset == "sleep_deprivation":
        parts = recording_name.split("_")
        return f"{parts[0]}_{parts[-2]}_{parts[-1]}_CBD.hyp"
    if sub_dataset == "optogenetic_stimulation":
        parts = recording_name.split("-")
        return f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[-1]}_TY.hyp"
    raise ValueError(
        f"Unknown sub_dataset {sub_dataset!r}. Expected one of: "
        f"'test', 'sleep_deprivation', 'optogenetic_stimulation'."
    )


# ---------------------------------------------------------------------------
# Sippel -- Morris water maze
# ---------------------------------------------------------------------------


@_register("sippel_morris_water_maze")
def _sippel_morris(cfg: dict) -> DatasetSpec:
    data_root, dataset, sub_dataset, recording_name = _require(
        cfg, "data_root", "dataset", "sub_dataset", "recording_name"
    )
    animal = recording_name.split("_")[0]
    base = os.path.join(data_root, dataset, "Raw_Data_Sleep", sub_dataset, animal)

    annotation_path = os.path.join(base, f"{recording_name}_epochs.txt")
    return DatasetSpec(
        recording_path=_check_exists(os.path.join(base, f"{recording_name}.edf")),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=annotation_path,
        annotation_format="epoch_csv",
        annotation_kwargs=_sippel_annotation_kwargs(annotation_path),
        params="rat",
    )


def _sippel_annotation_kwargs(path: str) -> dict:
    """These files come with either (time, duration, label) or (time, label) columns."""
    kwargs: dict[str, Any] = {
        "names": ["time", "duration", "label"],
        "time_column": "time",
        "time_format": "datetime",
        "duration_column": "duration",
        "stage_column": "label",
        "header": 0,
    }
    if _column_count(path) < 3:
        # No duration column: these files are scored in fixed 4 s epochs.
        kwargs["names"] = ["time", "label"]
        kwargs["duration_column"] = None
        kwargs["epoch_length"] = 4.0
    return kwargs


def _column_count(path: str) -> int:
    try:
        with open(path) as handle:
            handle.readline()  # header
            first_row = handle.readline()
    except OSError:
        return 3
    return len(first_row.split(",")) if first_row.strip() else 3


# ---------------------------------------------------------------------------
# Ellen / Dash
# ---------------------------------------------------------------------------


@_register("ellen_dash")
def _ellen_dash(cfg: dict) -> DatasetSpec:
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset)
    return DatasetSpec(
        recording_path=_check_exists(os.path.join(base, f"{recording_name}.edf")),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=os.path.join(base, "manual_scoring_all_rats.csv"),
        annotation_format="column_csv",
        # One CSV holds every animal, one column each -- so the reader has to be
        # told which recording it is loading.
        annotation_kwargs={
            "column": recording_name,
            "epoch_length": 4.0,
            "label_map": {0: "WAKE", 1: "NREM", 2: "REM"},
            "header": 1,
            "index_col": 0,
        },
        params="rat",
    )


# ---------------------------------------------------------------------------
# SIESTA
# ---------------------------------------------------------------------------


@_register("siesta_DLI")
def _siesta_dli(cfg: dict) -> DatasetSpec:
    data_root, dataset, sub_dataset, recording_name = _require(
        cfg, "data_root", "dataset", "sub_dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset, sub_dataset)
    return DatasetSpec(
        recording_path=_check_exists(os.path.join(base, "edf", f"{recording_name}.edf")),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=os.path.join(base, "scores", f"{recording_name}.csv"),
        annotation_format="epoch_csv",
        annotation_kwargs={
            "epoch_length": 10.0,
            "label_map": {1: "WAKE", 2: "NREM", 3: "REM", 255: "UNDEFINED"},
        },
        params="mouse",
        source="https://doi.org/10.5281/zenodo.15322394",
    )


# ---------------------------------------------------------------------------
# Boccara lab (DSI telemetry)
# ---------------------------------------------------------------------------

@_register("boccara_lab")
@_register("boccaralab_dsi")
def _boccaralab_dsi(cfg: dict) -> DatasetSpec:
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset)
    return DatasetSpec(
        recording_path=_check_exists(os.path.join(base, "recordings", recording_name)),
        recording_format="spikeinterface",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=os.path.join(base, "annotations", f"{recording_name}.csv"),
        annotation_format="interval_csv",
        params="mouse",
    )


# ---------------------------------------------------------------------------
# Dreem Open Datasets
# ---------------------------------------------------------------------------

#: Stage coding used by the Dreem Open Datasets' .npy annotations.
DREEM_STAGES = {
    -1: "NOSIGNAL",
    0: "WAKE",
    1: "NREM1",
    2: "NREM2",
    3: "NREM3",
    4: "REM",
}


@_register("dodo")
@_register("dodh")
def _dreem(cfg: dict) -> DatasetSpec:
    """Dreem Open Dataset, healthy (dodh) or with obstructive apnoea (dodo).

    Channels are selected by name rather than position, because the montage
    order is not consistent across recordings.
    """
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset)
    return DatasetSpec(
        recording_path=_check_exists(
            os.path.join(base, "recordings", f"{recording_name}.edf")
        ),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", "C3_M2"),
        emg_channel=cfg.get("emg_channel", "EMG"),
        annotation_path=os.path.join(base, "annotations", f"{recording_name}.npy"),
        annotation_format="epoch_npy",
        annotation_kwargs={"epoch_length": 30.0, "label_map": DREEM_STAGES},
        params="human",
        # The repository's own download script fetches from an S3 bucket that no
        # longer exists (Dreem folded); this Zenodo record is the live mirror.
        source="https://doi.org/10.5281/zenodo.15900394",
    )


# ---------------------------------------------------------------------------
# National Sleep Research Resource (MESA, CHAT, CCSHS)
# ---------------------------------------------------------------------------
#
# These share a folder layout and the NSRR XML annotation format. Access is by
# free application at https://sleepdata.org.


def _nsrr_paths(data_root: str, dataset: str, recording_name: str, sub: str = ""):
    base = os.path.join(data_root, dataset, "polysomnography")
    recording = os.path.join(base, "edfs", sub, f"{recording_name}.edf")
    annotation = os.path.join(
        base, "annotations-events-nsrr", sub, f"{recording_name}-nsrr.xml"
    )
    return os.path.normpath(recording), os.path.normpath(annotation)


@_register("mesa_aged")
@_register("mesa")
def _mesa(cfg: dict) -> DatasetSpec:
    """MESA Sleep -- Multi-Ethnic Study of Atherosclerosis."""
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    recording, annotation = _nsrr_paths(data_root, dataset, recording_name)
    return DatasetSpec(
        recording_path=_check_exists(recording),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", "EEG1"),
        emg_channel=cfg.get("emg_channel", "EMG"),
        annotation_path=annotation,
        annotation_format="nsrr_xml",
        params="human",
        source="https://sleepdata.org/datasets/mesa",
    )


@_register("chat")
def _chat(cfg: dict) -> DatasetSpec:
    """CHAT -- Childhood Adenotonsillectomy Trial.

    Split across visits, so ``sub_dataset`` selects one (``baseline``,
    ``followup``, ``nonrandomized``).
    """
    data_root, dataset, sub_dataset, recording_name = _require(
        cfg, "data_root", "dataset", "sub_dataset", "recording_name"
    )
    recording, annotation = _nsrr_paths(data_root, dataset, recording_name, sub_dataset)
    return DatasetSpec(
        recording_path=_check_exists(recording),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", "C4"),
        # Chin EMG is spelled inconsistently across recordings.
        emg_channel=cfg.get("emg_channel", ["Lchin", "LChin"]),
        annotation_path=annotation,
        annotation_format="nsrr_xml",
        params="human",
        source="https://sleepdata.org/datasets/chat",
    )


@_register("ccshs")
def _ccshs(cfg: dict) -> DatasetSpec:
    """CCSHS -- Cleveland Children's Sleep and Health Study.

    EEG and EMG live in different EDF streams, sampled at different rates, so
    the EMG is read from stream 1 and its rate carried separately.
    """
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    recording, annotation = _nsrr_paths(data_root, dataset, recording_name)
    return DatasetSpec(
        recording_path=_check_exists(recording),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", "C4"),
        emg_channel=cfg.get("emg_channel", "EMG1"),
        recording_kwargs={"stream_id": "0", "emg_stream_id": "1"},
        annotation_path=annotation,
        annotation_format="nsrr_xml",
        params="human",
        source="https://sleepdata.org/datasets/ccshs",
    )


# ---------------------------------------------------------------------------
# ANPHY
# ---------------------------------------------------------------------------


@_register("anphy_sleep_human")
def _anphy(cfg: dict) -> DatasetSpec:
    """ANPHY-Sleep -- one folder per subject, scores as a tab-separated text file."""
    data_root, dataset, recording_name = _require(
        cfg, "data_root", "dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset, recording_name)
    return DatasetSpec(
        recording_path=_check_exists(os.path.join(base, f"{recording_name}.edf")),
        recording_format="edf",
        # Referenced and unreferenced montages appear in different recordings.
        eeg_channel=cfg.get("eeg_channel", ["C4", "C4-Ref"]),
        emg_channel=cfg.get("emg_channel", ["ChEMG1", "ChEMG1-Ref"]),
        annotation_path=os.path.join(base, f"{recording_name}.txt"),
        annotation_format="interval_csv",
        annotation_kwargs={
            "names": ["label", "time", "duration"],
            "sep": "\t",
            "header": None,
        },
        params="human",
        source="https://doi.org/10.6084/m9.figshare.c.7016359",
    )


# ---------------------------------------------------------------------------
# Gulledge 2025
# ---------------------------------------------------------------------------


@_register("gulledge-2025")
def _gulledge_2025(cfg: dict) -> DatasetSpec:
    data_root, dataset, sub_dataset, recording_name = _require(
        cfg, "data_root", "dataset", "sub_dataset", "recording_name"
    )
    base = os.path.join(data_root, dataset, "original")
    return DatasetSpec(
        recording_path=_check_exists(
            os.path.join(base, sub_dataset, f"{recording_name}.edf")
        ),
        recording_format="edf",
        eeg_channel=cfg.get("eeg_channel", 0),
        emg_channel=cfg.get("emg_channel", 1),
        annotation_path=os.path.join(
            base,
            "Sleep Scoring and Behavioral Data",
            "Sleep Staging",
            sub_dataset,
            f"{recording_name}_staging.csv",
        ),
        annotation_format="epoch_csv",
        annotation_kwargs={
            "stage_column": "Rodent Sleep",
            "time_column": "Time Stamp",
            "time_format": "clock",
            "label_map": {3: "NREM", 4: "REM", 5: "WAKE"},
        },
        params="mouse",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def list_datasets() -> list[str]:
    """Names accepted by :func:`load_dataset`."""
    return sorted(DATASETS)


def resolve_dataset(name: str, **cfg) -> DatasetSpec:
    """Work out the file paths and reader settings for one recording."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {list_datasets()}. "
            f"You do not need a dataset recipe to use nyx -- see nyx.io.read_recording."
        )
    cfg.setdefault("dataset", name)
    return DATASETS[name](cfg)


def describe_dataset(name: str, **cfg) -> str:
    """Show which files a dataset recipe resolves to, without loading them."""
    spec = resolve_dataset(name, **cfg)
    return (
        f"{name}\n"
        f"  recording:   {spec.recording_path}  [{spec.recording_format}]\n"
        f"  channels:    eeg={spec.eeg_channel!r}  emg={spec.emg_channel!r}\n"
        f"  annotations: {spec.annotation_path}  [{spec.annotation_format}]\n"
        f"  params used: {spec.params}"
    )


def load_dataset(
    name: str, with_annotations: bool = True, **cfg
) -> tuple[Recording, dict | None]:
    """Load one recording of a known dataset, with its manual scoring if present.

    Parameters
    ----------
    name
        A name from :func:`list_datasets`.
    with_annotations
        Set to ``False`` to skip the manual scoring even when it exists.
    **cfg
        Dataset-specific keys, typically ``data_root``, ``recording_name`` and
        (for some datasets) ``sub_dataset``.

    Returns
    -------
    (Recording, annotations or None)
    """
    spec = resolve_dataset(name, **cfg)

    recording = read_recording(
        spec.recording_path,
        format=spec.recording_format,
        eeg_channel=spec.eeg_channel,
        emg_channel=spec.emg_channel,
        name=str(cfg.get("recording_name", "")),
        **spec.recording_kwargs,
    )

    annotations = None
    if with_annotations and spec.annotation_path:
        if os.path.exists(spec.annotation_path):
            annotations = read_annotations(
                spec.annotation_path,
                format=spec.annotation_format,
                **spec.annotation_kwargs,
            )
        else:
            print(
                f"No manual scoring found at {spec.annotation_path} -- "
                f"continuing without a reference scoring."
            )

    return recording, annotations
