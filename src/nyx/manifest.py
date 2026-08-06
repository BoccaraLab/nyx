"""Batch scoring driven by a CSV manifest, one row per recording.

A manifest is the practical way to score a study: labs already keep the
recording list in a spreadsheet, and a CSV lets each recording carry its own
channels, window and parameter file without a config file per recording.

Minimal manifest::

    recording_path,annotation_path
    data/m01.edf,scores/m01.csv
    data/m02.edf,scores/m02.csv

Everything else is optional and falls back to the defaults you pass to
:func:`load_manifest`:

============================  ==========================================
column                        meaning
============================  ==========================================
``recording_path``            signal file (required, unless ``dataset``)
``recording_format``          ``edf`` / ``spikeinterface`` / ``npz``
``eeg_channel``               index or channel name
``emg_channel``               index or channel name
``annotation_path``           reference scoring; blank means none
``annotation_format``         reader name
``params``                    parameter file for this recording
``window_start``              analysis window, seconds
``window_end``                analysis window, seconds
``name``                      output folder name
``dataset``                   dataset recipe name, instead of paths
``recording_name``            recording id within that dataset
``data_root``                 dataset root, for recipes
============================  ==========================================

Unknown columns are passed to the dataset recipe, so a recipe needing
``sub_dataset`` just gets a ``sub_dataset`` column.

Paths are resolved relative to the manifest file.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from typing import Any

from nyx.config import RunConfig

__all__ = ["load_manifest", "write_manifest", "MANIFEST_COLUMNS"]

#: Columns nyx interprets. Anything else becomes a dataset option.
MANIFEST_COLUMNS = (
    "recording_path",
    "recording_format",
    "eeg_channel",
    "emg_channel",
    "annotation_path",
    "annotation_format",
    "params",
    "window_start",
    "window_end",
    "name",
    "dataset",
)


def load_manifest(
    path: str,
    params: str | None = None,
    output_root: str = "results",
    **defaults: Any,
) -> list[RunConfig]:
    """Read a manifest into one :class:`~nyx.config.RunConfig` per row.

    Parameters
    ----------
    path
        The CSV file.
    params
        Parameter file to use for rows that do not name their own. Resolved
        relative to the *current* directory, since that is where you typed it;
        a ``params`` column resolves relative to the manifest, like the other
        paths in it.
    output_root
        Where results go; each row gets a subfolder. Like ``params``, resolved
        relative to the current directory rather than to the manifest.
    **defaults
        Applied to any row that leaves the corresponding column blank -- for
        example ``eeg_channel=0, data_root="/data"``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    base = os.path.dirname(os.path.abspath(path))

    # Paths *in* the manifest resolve against the manifest, like every other
    # path there. These two are arguments, typed by the caller in the caller's
    # directory, so resolve them there before the manifest's rule can claim
    # them -- otherwise `output_root="results"` lands inside the manifest's
    # folder rather than beside the notebook that asked for it.
    if params and (os.sep in params or "/" in params or params.endswith(".json")):
        params = os.path.abspath(params)  # a path; a bare preset name is left alone
    output_root = os.path.abspath(output_root)

    configs: list[RunConfig] = []

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty -- a manifest needs a header row.")

        for number, row in enumerate(reader, start=2):  # row 1 is the header
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items() if k}
            if not any(row.values()):
                continue  # blank line
            try:
                configs.append(
                    _row_to_config(row, base, params, output_root, defaults)
                )
            except Exception as exc:
                raise ValueError(f"{path} line {number}: {exc}") from exc

    if not configs:
        raise ValueError(f"{path} has a header but no rows.")
    return configs


def _row_to_config(
    row: dict, base: str, params: str | None, output_root: str, defaults: dict
) -> RunConfig:
    def value(key, fallback=None):
        got = row.get(key) or ""
        if got == "":
            got = defaults.get(key, fallback)
        return got if got != "" else None

    spec: dict[str, Any] = {"output": {"root": output_root}}

    if value("name"):
        spec["output"]["name"] = value("name")

    row_params = value("params") or params
    if row_params:
        spec["params"] = row_params

    start, end = value("window_start"), value("window_end")
    if start is not None or end is not None:
        spec["window"] = [float(start or 0.0), float(end if end is not None else 1e12)]

    dataset = value("dataset")
    if dataset:
        spec["dataset"] = dataset
        # Every remaining column is a recipe argument (recording_name,
        # sub_dataset, data_root, ...).
        for key, val in {**defaults, **row}.items():
            if key not in MANIFEST_COLUMNS and val not in ("", None):
                spec[key] = val
        for key in ("data_root", "recording_name"):
            if value(key) is not None:
                spec[key] = value(key)
    else:
        recording_path = value("recording_path")
        if not recording_path:
            raise KeyError(
                "a row needs either 'recording_path' or 'dataset'; "
                f"got columns {sorted(k for k, v in row.items() if v)}"
            )
        spec["recording"] = {
            "path": recording_path,
            "format": value("recording_format", "auto"),
            "eeg_channel": _channel(value("eeg_channel", 0)),
            "emg_channel": _channel(value("emg_channel", 1)),
        }
        if value("annotation_path"):
            spec["annotations"] = {
                "path": value("annotation_path"),
                "format": value("annotation_format", "auto"),
            }

    return RunConfig.from_dict(spec, base=base)


def _channel(raw: Any) -> int | str:
    """A channel column holds either a position or a name."""
    if isinstance(raw, (int, float)):
        return int(raw)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return str(raw).strip()


def write_manifest(path: str, rows: Iterator[dict] | list[dict]) -> str:
    """Write a manifest, using only the columns the rows actually populate."""
    rows = list(rows)
    if not rows:
        raise ValueError("Nothing to write.")

    used = [c for c in MANIFEST_COLUMNS if any(r.get(c) not in ("", None) for r in rows)]
    extra = sorted({k for r in rows for k in r} - set(MANIFEST_COLUMNS))

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=used + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
