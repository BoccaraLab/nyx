"""Any format Neo reads, through spikeinterface's Neo extractors.

nyx reads EDF, ``.npz`` and spikeinterface folders itself. Everything else an
acquisition system writes -- Open Ephys, Intan, SpikeGLX, Spike2, Blackrock,
Plexon, Neuralynx, TDT and the rest -- is read here, as ``format="neo"``::

    recording = nyx.read_recording(
        "session/", format="neo", neo_format="OpenEphysBinary",
        eeg_channel="CH12", emg_channel="CH31",
    )

``neo_format`` can usually be left out: Neo guesses it from the file
extension, or from what a folder contains. The data stays lazy on disk, as with
every other reader.

Files from these systems often hold several *streams* -- sets of channels
sampled together, like an amplifier stream and an auxiliary one at a lower
rate. The EEG is read from ``stream_id`` (the first stream by default), and
the EMG from ``emg_stream_id`` when it lives in another one.

Which channel is which is rarely obvious from a raw file, so
:func:`recording_streams` and :func:`describe_channels` open every stream
without choosing anything, for looking at before picking two channels.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from functools import cache

import numpy as np

from nyx.io.recordings import (
    ChannelSpec,
    register_channel_lister,
    register_recording_reader,
)

__all__ = [
    "ChannelInfo",
    "NeoFormat",
    "neo_formats",
    "guess_neo_format",
    "neo_streams",
    "open_neo",
    "recording_streams",
    "describe_channels",
]


@dataclass(frozen=True)
class NeoFormat:
    """One format readable through Neo."""

    #: Neo's name for it, without the ``RawIO`` suffix: ``"OpenEphysBinary"``.
    name: str
    #: Whether it is opened as a folder rather than a single file.
    is_folder: bool
    #: File extensions Neo associates with it, without the dot.
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class ChannelInfo:
    """One channel in a file, with what the file says about it."""

    stream_id: str
    stream_name: str
    #: Position within its stream (0 = the stream's first channel).
    index: int
    #: The id to select it by.
    channel_id: str
    #: The human-readable name, when the file has one; else the id.
    name: str
    fs: float
    #: Physical unit (``"uV"``, ``"mV"``...), ``""`` when the file has none.
    unit: str
    #: Raw sample -> ``unit``, ``None`` when the file does not say.
    gain: float | None
    offset: float | None
    #: Recording length, in seconds.
    duration: float


@cache
def neo_formats() -> dict[str, NeoFormat]:
    """Every Neo format spikeinterface can read, by name.

    Derived from spikeinterface's extractors rather than listed here, so a
    format that a newer spikeinterface learns to read turns up without a
    change to nyx.
    """
    return {name: fmt for name, (fmt, _cls) in _extractors().items()}


@cache
def _extractors() -> dict[str, tuple[NeoFormat, type]]:
    import spikeinterface.extractors.neoextractors as neoextractors
    from spikeinterface.extractors.neoextractors.neobaseextractor import (
        NeoBaseRecordingExtractor,
    )

    found: dict[str, tuple[NeoFormat, type]] = {}
    for obj in vars(neoextractors).values():
        if not (
            isinstance(obj, type)
            and issubclass(obj, NeoBaseRecordingExtractor)
            and obj is not NeoBaseRecordingExtractor
        ):
            continue
        rawio = getattr(obj, "NeoRawIOClass", None)
        if not rawio:
            continue
        name = rawio.removesuffix("RawIO")
        if name in found:
            continue
        found[name] = (
            NeoFormat(
                name=name,
                is_folder="folder_path" in inspect.signature(obj.__init__).parameters,
                extensions=tuple(_rawio_class(rawio).extensions or ()),
            ),
            obj,
        )
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def _rawio_class(name: str):
    import neo.rawio

    return getattr(neo.rawio, name)


def _extractor(neo_format: str):
    try:
        return _extractors()[neo_format][1]
    except KeyError:
        raise ValueError(
            f"Unknown Neo format {neo_format!r}. Available: "
            f"{', '.join(neo_formats())}."
        ) from None


def guess_neo_format(path: str) -> str | None:
    """Neo's guess at the format of a file or folder, ``None`` if it has none.

    Only formats spikeinterface can read are returned: Neo recognises a few
    more that it cannot hand over as a recording.
    """
    import warnings

    import neo.rawio

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rawio = neo.rawio.get_rawio(path)
    except Exception:  # noqa: BLE001 - "cannot tell" is an answer here
        return None
    if rawio is None:
        return None
    name = rawio.__name__.removesuffix("RawIO")
    return name if name in neo_formats() else None


def _resolve_format(path: str, neo_format: str | None) -> str:
    if neo_format and neo_format != "auto":
        return neo_format
    guessed = guess_neo_format(path)
    if guessed is None:
        raise ValueError(
            f"Neo cannot tell the format of {path!r}. Pass neo_format=..., "
            f"one of: {', '.join(neo_formats())}."
        )
    return guessed


def _path_kwargs(path: str, neo_format: str) -> dict:
    key = "folder_path" if neo_formats()[neo_format].is_folder else "file_path"
    # A folder format given a file inside the folder: open the folder.
    if key == "folder_path" and os.path.isfile(path):
        path = os.path.dirname(path)
    return {key: path}


def neo_streams(path: str, neo_format: str | None = None) -> list[tuple[str, str]]:
    """``(stream_id, stream_name)`` for every stream in the file."""
    neo_format = _resolve_format(path, neo_format)
    names, ids = _extractor(neo_format).get_streams(**_path_kwargs(path, neo_format))
    return [(str(i), str(n)) for i, n in zip(ids, names)]


def open_neo(path: str, neo_format: str | None = None, stream_id: str | None = None):
    """One stream of a Neo-readable file, every channel, as a lazy recording."""
    neo_format = _resolve_format(path, neo_format)
    if stream_id is None:
        stream_id = neo_streams(path, neo_format)[0][0]
    return _extractor(neo_format)(
        **_path_kwargs(path, neo_format), stream_id=str(stream_id)
    )


@register_recording_reader("neo")
def _read_neo(path: str, eeg_channel: ChannelSpec, emg_channel: ChannelSpec, **kwargs):
    """Read any Neo format. See the module docstring for the options."""
    from nyx.io.recordings import _resolve_channel, _split_channels

    neo_format = _resolve_format(path, kwargs.get("neo_format"))
    stream_id = kwargs.get("stream_id")
    if stream_id is None:
        stream_id = neo_streams(path, neo_format)[0][0]
    rec = open_neo(path, neo_format, stream_id)

    emg_stream_id = kwargs.get("emg_stream_id")
    if (
        emg_channel is not None
        and emg_stream_id is not None
        and str(emg_stream_id) != str(stream_id)
    ):
        emg_rec = open_neo(path, neo_format, emg_stream_id)
        return (
            rec.select_channels([_resolve_channel(rec, eeg_channel)]),
            emg_rec.select_channels([_resolve_channel(emg_rec, emg_channel)]),
            float(rec.get_sampling_frequency()),
            float(emg_rec.get_sampling_frequency()),
        )
    return _split_channels(rec, eeg_channel, emg_channel)


@register_channel_lister("neo")
def _list_neo_channels(path: str, **kwargs) -> list[str]:
    rec = open_neo(path, kwargs.get("neo_format"), kwargs.get("stream_id"))
    return [str(c) for c in rec.get_channel_ids()]


# ---------------------------------------------------------------------------
# Looking before choosing
# ---------------------------------------------------------------------------


def recording_streams(path: str, format: str = "auto", **kwargs) -> list:
    """Every stream in a recording, all channels, nothing selected yet.

    ``[(stream_id, stream_name, recording), ...]``, each recording lazy and
    multi-channel -- for showing what a file holds before two channels are
    picked out of it. Covers ``"neo"``, ``"edf"`` and ``"spikeinterface"``;
    raises ``LookupError`` for a format it cannot open this way (``"npz"``
    holds its two arrays by name and has nothing to choose between).
    """
    from nyx.io.recordings import _infer_format

    if not os.path.exists(path):
        raise FileNotFoundError(f"Recording not found: {path}")
    if format == "auto":
        format = _infer_format(path)

    if format == "spikeinterface":
        import spikeinterface as si

        return [("0", "recording", si.load(path))]

    if format == "edf":
        neo_format = "EDF"
    elif format == "neo":
        neo_format = _resolve_format(path, kwargs.get("neo_format"))
    else:
        raise LookupError(
            f"Cannot open the streams of a {format!r} recording; "
            f"'neo', 'edf' and 'spikeinterface' can be."
        )

    return [
        (stream_id, stream_name, open_neo(path, neo_format, stream_id))
        for stream_id, stream_name in neo_streams(path, neo_format)
    ]


def describe_channels(streams) -> list[ChannelInfo]:
    """What the file says about each channel of :func:`recording_streams`."""
    channels: list[ChannelInfo] = []
    for stream_id, stream_name, rec in streams:
        ids = [str(c) for c in rec.get_channel_ids()]
        names = _property(rec, "channel_name", ids)
        units = _property(rec, "physical_unit", [""] * len(ids))
        gains = _property(rec, "gain_to_physical_unit", None) or _property(
            rec, "gain_to_uV", None
        )
        offsets = _property(rec, "offset_to_physical_unit", None) or _property(
            rec, "offset_to_uV", None
        )
        if units == [""] * len(ids) and _property(rec, "gain_to_uV", None):
            units = ["uV"] * len(ids)

        fs = float(rec.get_sampling_frequency())
        duration = float(rec.get_num_samples(segment_index=0)) / fs
        for i, channel_id in enumerate(ids):
            channels.append(
                ChannelInfo(
                    stream_id=str(stream_id),
                    stream_name=str(stream_name),
                    index=i,
                    channel_id=channel_id,
                    name=str(names[i]) or channel_id,
                    fs=fs,
                    unit=str(units[i]),
                    gain=None if gains is None else float(gains[i]),
                    offset=None if offsets is None else float(offsets[i]),
                    duration=duration,
                )
            )
    return channels


def _property(rec, key: str, default):
    try:
        values = rec.get_property(key)
    except Exception:  # noqa: BLE001 - absent is the common case
        return default
    if values is None:
        return default
    values = list(np.asarray(values).tolist())
    return values if values else default
