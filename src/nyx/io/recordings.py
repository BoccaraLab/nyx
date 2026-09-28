"""Readers that turn a recording file into a :class:`~nyx.types.Recording`.

Readers are keyed by *file format* so that using nyx on new
data needs no code -- only a format name and two channel selectors. 
Formats nyx does not know about can be added from user code::

    from nyx.io import register_recording_reader

    @register_recording_reader("my_format")
    def read_my_format(path, eeg_channel, emg_channel, **kwargs):
        ...
        return eeg, emg, fs   # single-channel BaseRecordings + sampling rate
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from nyx.types import Recording

__all__ = [
    "read_recording",
    "list_channels",
    "register_recording_reader",
    "register_channel_lister",
    "RECORDING_READERS",
    "RECORDING_EXTENSIONS",
    "CHANNEL_LISTERS",
    "ChannelSpec",
]

# A channel is selected either by position (0 = first channel in the file) or by
# name (as stored in the file header).
ChannelSpec = int | str

RECORDING_READERS: dict[str, Callable[..., tuple]] = {}

#: File extension -> format name, used by ``format="auto"``. A folder is always
#: read as spikeinterface. Anything that builds a file dialog should read this
#: rather than hardcoding the list.
RECORDING_EXTENSIONS = {
    ".edf": "edf",
    ".bdf": "edf",
    ".npz": "npz",
}

#: Format name -> a function returning the channel names in a file, so a
#: channel can be chosen before anything is loaded. Optional: a format with no
#: lister simply cannot be browsed.
CHANNEL_LISTERS: dict[str, Callable[..., list[str]]] = {}


def register_recording_reader(name: str) -> Callable:
    """Register a reader under ``name`` so it can be used as ``format=name``."""

    def decorator(func: Callable[..., tuple]) -> Callable[..., tuple]:
        RECORDING_READERS[name] = func
        return func

    return decorator


def register_channel_lister(name: str) -> Callable:
    """Register a channel lister for the format ``name``.

    ``lister(path, **kwargs) -> list[str]``. Registering one is what lets a
    caller offer the file's channels to pick from instead of asking for an
    index typed blind.
    """

    def decorator(func: Callable[..., list[str]]) -> Callable[..., list[str]]:
        CHANNEL_LISTERS[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------


def _resolve_channel(rec, spec: ChannelSpec):
    """Map an index or a channel name onto a spikeinterface channel id.

    A list or tuple of names is tried in order and the first present wins,
    which covers datasets that spell the same electrode differently across
    recordings (``C4`` vs ``C4-Ref``, ``Lchin`` vs ``LChin``).
    """
    channel_ids = list(rec.get_channel_ids())

    if isinstance(spec, (list, tuple)):
        for candidate in spec:
            try:
                return _resolve_channel(rec, candidate)
            except (KeyError, IndexError):
                continue
        raise KeyError(
            f"None of the channels {list(spec)} were found. "
            f"Available channels: {channel_ids}."
        )

    if isinstance(spec, (int, np.integer)):
        if not -len(channel_ids) <= int(spec) < len(channel_ids):
            raise IndexError(
                f"Channel index {spec} is out of range: the file has {len(channel_ids)} "
                f"channel(s) {channel_ids}."
            )
        return channel_ids[int(spec)]

    if spec in channel_ids:
        return spec

    # Fall back to matching the human-readable channel names.
    for key in ("channel_name", "channel_names"):
        try:
            names = rec.get_property(key)
        except Exception:
            continue
        if names is None:
            continue
        names = [str(n) for n in names]
        if str(spec) in names:
            return channel_ids[names.index(str(spec))]

    raise KeyError(
        f"Channel {spec!r} not found. Available channels: {channel_ids}. "
        f"You can also select a channel by position, e.g. eeg_channel=0."
    )


def _split_channels(rec, eeg_channel: ChannelSpec, emg_channel: ChannelSpec):
    """Split a multi-channel recording into single-channel EEG and EMG recordings.

    ``emg_channel=None`` means the file has no EMG, or you do not want to use it.
    """
    eeg = rec.select_channels([_resolve_channel(rec, eeg_channel)])
    emg = (None if emg_channel is None
           else rec.select_channels([_resolve_channel(rec, emg_channel)]))
    return eeg, emg, float(rec.get_sampling_frequency())


def _as_single_channel(trace: np.ndarray, fs: float, name: str):
    """Wrap a 1-D numpy trace as a single-channel spikeinterface recording."""
    from spikeinterface.core import NumpyRecording

    rec = NumpyRecording([np.asarray(trace, dtype=float).reshape(-1, 1)], float(fs))
    rec.set_property(key="channel_name", values=[name])
    return rec


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


@register_recording_reader("edf")
def _read_edf(path: str, eeg_channel: ChannelSpec, emg_channel: ChannelSpec, **kwargs):
    """Read an EDF/EDF+/BDF file.

    Tries spikeinterface first, then pyedflib, then MNE. The fallbacks exist
    because a lot of real EDF+ files are not standard-compliant and
    the stricter readers refuse them.

    Some polysomnography files split signals across EDF *streams* sampled at
    different rates -- EEG in one, EMG in another. Pass ``emg_stream_id`` to
    read the EMG from a different stream; the two rates are then carried
    separately.
    """
    errors: list[str] = []
    stream_id = str(kwargs.get("stream_id", "0"))
    emg_stream_id = kwargs.get("emg_stream_id")

    # 1. spikeinterface (keeps the data lazy on disk)
    try:
        import spikeinterface.extractors as se

        rec = se.read_edf(path, stream_id=stream_id)
        if emg_stream_id is not None and str(emg_stream_id) != stream_id:
            emg_rec = se.read_edf(path, stream_id=str(emg_stream_id))
            eeg = rec.select_channels([_resolve_channel(rec, eeg_channel)])
            emg = emg_rec.select_channels([_resolve_channel(emg_rec, emg_channel)])
            return (
                eeg,
                emg,
                float(rec.get_sampling_frequency()),
                float(emg_rec.get_sampling_frequency()),
            )
        return _split_channels(rec, eeg_channel, emg_channel)
    except Exception as exc:  # noqa: BLE001 - we deliberately try the next reader
        errors.append(f"spikeinterface: {exc}")

    # 2. pyedflib
    try:
        import pyedflib

        reader = pyedflib.EdfReader(path)
        try:
            names = [reader.getLabel(i) for i in range(reader.signals_in_file)]
            eeg_idx = _resolve_index_by_name(eeg_channel, names, "EEG")
            fs = float(reader.getSampleFrequency(eeg_idx))
            eeg = _as_single_channel(reader.readSignal(eeg_idx), fs, names[eeg_idx])
            emg = None
            if emg_channel is not None:
                emg_idx = _resolve_index_by_name(emg_channel, names, "EMG")
                emg = _as_single_channel(reader.readSignal(emg_idx), fs, names[emg_idx])
        finally:
            reader.close()
        return eeg, emg, fs
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pyedflib: {exc}")

    # 3. MNE (the most permissive reader)
    try:
        import mne

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        names = list(raw.ch_names)
        eeg_idx = _resolve_index_by_name(eeg_channel, names, "EEG")
        fs = float(raw.info["sfreq"])
        eeg = _as_single_channel(raw.get_data(picks=[names[eeg_idx]])[0], fs, names[eeg_idx])
        emg = None
        if emg_channel is not None:
            emg_idx = _resolve_index_by_name(emg_channel, names, "EMG")
            emg = _as_single_channel(
                raw.get_data(picks=[names[emg_idx]])[0], fs, names[emg_idx]
            )
        return eeg, emg, fs
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mne: {exc}")

    raise OSError(
        f"Could not read {path!r} as EDF with any available reader.\n  "
        + "\n  ".join(errors)
    )


def _resolve_index_by_name(spec: ChannelSpec, names: list[str], role: str) -> int:
    """Resolve a channel spec against a plain list of channel names."""
    if isinstance(spec, (int, np.integer)):
        if not -len(names) <= int(spec) < len(names):
            raise IndexError(
                f"{role} channel index {spec} is out of range: the file has "
                f"{len(names)} channel(s) {names}."
            )
        return int(spec) % len(names)
    if str(spec) in names:
        return names.index(str(spec))
    raise KeyError(f"{role} channel {spec!r} not found. Available channels: {names}.")


@register_recording_reader("spikeinterface")
def _read_spikeinterface(
    path: str, eeg_channel: ChannelSpec, emg_channel: ChannelSpec, **kwargs
):
    """Read a folder previously saved with spikeinterface (``recording.save(...)``)."""
    import spikeinterface as si

    rec = si.load(path)
    return _split_channels(rec, eeg_channel, emg_channel)


@register_recording_reader("npz")
def _read_npz(path: str, eeg_channel: ChannelSpec, emg_channel: ChannelSpec, **kwargs):
    """Read a ``.npz`` archive holding ``eeg``, ``emg`` and (optionally) ``fs``.

    The arrays are already named, so ``eeg_channel`` is ignored. Passing
    ``emg_channel=None`` skips the EMG even when the archive has one.
    """
    data = np.load(path)
    wanted = ("eeg",) if emg_channel is None else ("eeg", "emg")
    missing = [key for key in wanted if key not in data]
    if missing:
        raise KeyError(
            f"{path!r} is missing the {missing} array(s). Expected keys: 'eeg', 'emg', "
            f"and either an 'fs' key or an explicit fs=... argument. Found: {list(data)}."
        )
    fs = float(data["fs"]) if "fs" in data else kwargs.get("fs")
    if fs is None:
        raise KeyError(
            f"{path!r} has no 'fs' array; pass the sampling rate explicitly, e.g. fs=256."
        )
    eeg = _as_single_channel(np.asarray(data["eeg"], dtype=float), fs, "EEG")
    emg = (None if emg_channel is None
           else _as_single_channel(np.asarray(data["emg"], dtype=float), fs, "EMG"))
    return eeg, emg, fs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_recording(
    path: str,
    format: str = "auto",
    eeg_channel: ChannelSpec = 0,
    emg_channel: ChannelSpec = 1,
    name: str = "",
    **kwargs,
) -> Recording:
    """Load one EEG channel and one EMG channel from a recording file.

    Parameters
    ----------
    path
        File (``.edf``, ``.npz``, or anything Neo reads) or folder
        (spikeinterface, or a Neo folder format such as Open Ephys) to read.
    format
        One of :data:`RECORDING_READERS`, or ``"auto"`` to infer it from the
        path. Currently: ``"edf"``, ``"spikeinterface"``, ``"npz"``, and
        ``"neo"`` for everything else -- see :mod:`nyx.io.neo` for its
        options (``neo_format``, ``stream_id``, ``emg_stream_id``).
    eeg_channel, emg_channel
        Channel position (``0`` = first channel in the file) or channel name.
        ``emg_channel=None`` loads no EMG at all -- possible, but a long way
        from advisable: EMG power is what separates wake from sleep, and
        without it wake has to come out of the EEG spectrum alone, where quiet
        wake and REM look much alike. If the file has any other wideband
        channels, :func:`nyx.emg_from_lfp` will build a surrogate from them,
        which is far better than nothing.
    name
        Label carried through to the results; defaults to the file stem.

    Returns
    -------
    Recording
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recording not found: {path}")

    if format == "auto":
        format = _infer_format(path)

    if format not in RECORDING_READERS:
        raise ValueError(
            f"Unknown recording format {format!r}. "
            f"Available: {sorted(RECORDING_READERS)}. "
            f"Register your own with nyx.io.register_recording_reader."
        )

    result = RECORDING_READERS[format](
        path, eeg_channel=eeg_channel, emg_channel=emg_channel, **kwargs
    )
    # A reader may return a separate EMG sampling rate when the two channels
    # come from streams recorded at different rates.
    if len(result) == 4:
        eeg, emg, fs, emg_fs = result
    else:
        (eeg, emg, fs), emg_fs = result, None

    return Recording(
        eeg=eeg,
        emg=emg,
        fs=float(fs),
        name=name or os.path.splitext(os.path.basename(path.rstrip("/\\")))[0],
        source_path=path,
        emg_fs_=(float(emg_fs) if emg_fs and float(emg_fs) != float(fs) else None),
        source_format=format,
        source_options=dict(kwargs),
    )


# ---------------------------------------------------------------------------
# Listing channels
# ---------------------------------------------------------------------------


@register_channel_lister("edf")
def _list_edf_channels(path: str, **kwargs) -> list[str]:
    """Channel names in an EDF/BDF, via the same fallback chain as the reader.

    Nothing here loads sample data -- MNE is asked for ``preload=False``, which
    is the difference between reading a header and reading a night.
    """
    errors: list[str] = []
    stream_id = str(kwargs.get("stream_id", "0"))

    try:
        import spikeinterface.extractors as se

        return [str(c) for c in se.read_edf(path, stream_id=stream_id).get_channel_ids()]
    except Exception as exc:  # noqa: BLE001 - try the next reader
        errors.append(f"spikeinterface: {exc}")

    try:
        import pyedflib

        reader = pyedflib.EdfReader(path)
        try:
            return [reader.getLabel(i) for i in range(reader.signals_in_file)]
        finally:
            reader.close()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pyedflib: {exc}")

    try:
        import mne

        return list(mne.io.read_raw_edf(path, preload=False, verbose=False).ch_names)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mne: {exc}")

    raise OSError(
        f"Could not read the channel names of {path!r}.\n  " + "\n  ".join(errors)
    )


@register_channel_lister("spikeinterface")
def _list_spikeinterface_channels(path: str, **kwargs) -> list[str]:
    import spikeinterface as si

    return [str(c) for c in si.load(path).get_channel_ids()]


@register_channel_lister("npz")
def _list_npz_channels(path: str, **kwargs) -> list[str]:
    """The arrays an npz archive actually carries, of the ones nyx reads."""
    with np.load(path) as data:
        return [key for key in ("eeg", "emg") if key in data]


def list_channels(path: str, format: str = "auto", **kwargs) -> list[str]:
    """Channel names in a recording file, without loading it.

    For choosing ``eeg_channel`` and ``emg_channel`` before calling
    :func:`read_recording` -- which needs them up front, and so cannot tell you
    what there is to choose from.

    Raises ``LookupError`` for a format with no registered lister; register one
    with :func:`register_channel_lister`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recording not found: {path}")

    if format == "auto":
        format = _infer_format(path)

    lister = CHANNEL_LISTERS.get(format)
    if lister is None:
        raise LookupError(
            f"No channel lister for format {format!r}. Available: "
            f"{sorted(CHANNEL_LISTERS)}. Register one with "
            f"nyx.io.register_channel_lister."
        )
    return [str(name) for name in lister(path, **kwargs)]


#: Files spikeinterface writes into a folder it saves, by version.
_SPIKEINTERFACE_MARKERS = ("si_folder.json", "spikeinterface_info.json")


def _infer_format(path: str) -> str:
    """nyx's own formats by extension, then whatever Neo recognises.

    A folder spikeinterface saved is ``"spikeinterface"``; any other folder
    (Open Ephys, SpikeGLX, Neuralynx...) is offered to Neo, and falls back to
    spikeinterface only if Neo does not recognise it either.
    """
    from nyx.io.neo import guess_neo_format

    if os.path.isdir(path):
        if any(os.path.exists(os.path.join(path, m)) for m in _SPIKEINTERFACE_MARKERS):
            return "spikeinterface"
        return "neo" if guess_neo_format(path) else "spikeinterface"
    ext = os.path.splitext(path)[1].lower()
    if ext in RECORDING_EXTENSIONS:
        return RECORDING_EXTENSIONS[ext]
    if guess_neo_format(path):
        return "neo"
    raise ValueError(
        f"Cannot infer the format of {path!r} from its extension ({ext!r}). "
        f"Pass format= explicitly, one of: {sorted(RECORDING_READERS)}; for "
        f"format='neo', neo_format= names the Neo format."
    )
