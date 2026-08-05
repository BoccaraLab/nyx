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
    "register_recording_reader",
    "RECORDING_READERS",
    "ChannelSpec",
]

# A channel is selected either by position (0 = first channel in the file) or by
# name (as stored in the file header).
ChannelSpec = int | str

RECORDING_READERS: dict[str, Callable[..., tuple]] = {}

_EXTENSION_TO_FORMAT = {
    ".edf": "edf",
    ".bdf": "edf",
    ".npz": "npz",
}


def register_recording_reader(name: str) -> Callable:
    """Register a reader under ``name`` so it can be used as ``format=name``."""

    def decorator(func: Callable[..., tuple]) -> Callable[..., tuple]:
        RECORDING_READERS[name] = func
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
        File (``.edf``, ``.npz``) or folder (spikeinterface) to read.
    format
        One of :data:`RECORDING_READERS`, or ``"auto"`` to infer it from the
        path. Currently: ``"edf"``, ``"spikeinterface"``, ``"npz"``.
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
    )


def _infer_format(path: str) -> str:
    if os.path.isdir(path):
        return "spikeinterface"
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXTENSION_TO_FORMAT:
        return _EXTENSION_TO_FORMAT[ext]
    raise ValueError(
        f"Cannot infer the format of {path!r} from its extension ({ext!r}). "
        f"Pass format= explicitly, one of: {sorted(RECORDING_READERS)}."
    )
