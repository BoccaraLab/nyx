"""Signal conditioning, applied before features are computed.

Everything here is spikeinterface preprocessing applied to the recording rather
than to a numpy trace, which buys two things:

* it is **lazy** -- nothing is filtered until a trace is actually read, so a
  24 h recording is not copied to notch it;
* it happens **once, up front**, so :func:`nyx.check_signals` and
  :func:`nyx.score_recording` look at exactly the same signal. Filtering inside
  feature computation instead means the figures and the scoring can disagree
  about what was analysed.

Settings live per channel in the params, because notching the EMG but not the
EEG is a normal thing to want. Two common cases have shorthands::

    "EMG": { "notch": 50, "resample": 250 }

============================  ============================================
shorthand                     effect
============================  ============================================
``notch``                     mains frequency to remove: 50 (Europe) or 60
                              (Americas), plus harmonics. ``null`` disables.
``notch_q``                   notch width; higher is narrower. Default 30.
``notch_harmonics``           also remove 2x, 3x, ... Default true.
``resample``                  target sampling rate in Hz. Lowering it speeds
                              everything up; keep it above twice the highest
                              frequency you care about.
============================  ============================================

Anything else in :mod:`spikeinterface.preprocessing` can be named directly, as
an ordered list::

    "EEG": {
      "preprocess": [
        {"notch_filter":    {"freq": 50, "q": 30}},
        {"bandpass_filter": {"freq_min": 0.5, "freq_max": 45}},
        {"resample":        {"resample_rate": 250}}
      ]
    }

nyx does not wrap these one by one -- the name is looked up in
``spikeinterface.preprocessing``, so the whole catalogue is available and new
ones work without a change here. Because the chain lives in the params it is
written into the run record and replays with everything else, which is the
reason to configure it here rather than calling spikeinterface by hand.

The order is the order you write, and it matters: notching before resampling
is not the same as after.
"""

from __future__ import annotations

import warnings

from nyx.types import Recording

__all__ = ["preprocess_recording", "preprocess_channel", "DEFAULT_MAINS_HZ"]

#: Mains frequency assumed when none is given. 50 Hz covers Europe, Asia,
#: Africa and most of South America; set 60 for North America and Japan.
DEFAULT_MAINS_HZ = 50.0


def _expand_shorthands(rec, settings: dict) -> list[dict]:
    """Turn the ``notch`` / ``resample`` shorthands into explicit steps."""
    steps: list[dict] = []

    notch = settings.get("notch")
    if notch:
        notch = float(notch)
        quality = float(settings.get("notch_q", settings.get("notch_quality", 30.0)))
        nyquist = rec.get_sampling_frequency() / 2.0

        targets = [notch]
        if settings.get("notch_harmonics", True):
            # Mains pickup is not a pure sinusoid, so the harmonics matter too.
            targets += [notch * k for k in range(2, int(nyquist // notch) + 1)]
        steps += [
            {"notch_filter": {"freq": target, "q": quality}}
            for target in targets
            if target < nyquist * 0.99
        ]

    rate = settings.get("resample")
    if rate:
        steps.append({"resample": {"resample_rate": int(round(float(rate)))}})

    return steps


def preprocess_channel(rec, settings: dict):
    """Apply one channel's preprocessing chain.

    ``settings`` may hold an explicit ordered list::

        {"preprocess": [{"notch_filter": {"freq": 50}},
                        {"resample": {"resample_rate": 250}}]}

    or the ``notch`` / ``resample`` shorthands, which expand to the same thing.
    Anything in :mod:`spikeinterface.preprocessing` can be named, so the whole
    catalogue is available without nyx having to wrap each function -- and
    because it lives in the params, it is recorded and replayed with everything
    else.

    Returns the recording unchanged when nothing is configured.
    """
    import spikeinterface.preprocessing as spre

    explicit = settings.get("preprocess")
    if explicit:
        if settings.get("notch") or settings.get("resample"):
            warnings.warn(
                "Both a 'preprocess' list and the notch/resample shorthands are "
                "set; the explicit list wins and the shorthands are ignored. "
                "Put everything in the list so the order is unambiguous.",
                stacklevel=3,
            )
        steps = list(explicit)
    else:
        steps = _expand_shorthands(rec, settings)

    for step in steps:
        for name, kwargs in _as_call(step).items():
            function = getattr(spre, name, None)
            if function is None or not callable(function):
                available = sorted(
                    n for n in dir(spre)
                    if not n.startswith("_") and callable(getattr(spre, n, None))
                )
                raise ValueError(
                    f"Unknown preprocessing step {name!r}. It must name something "
                    f"in spikeinterface.preprocessing, e.g. 'notch_filter', "
                    f"'bandpass_filter', 'resample', 'common_reference'. "
                    f"Available: {available}"
                )

            if name == "resample":
                rate = float(kwargs.get("resample_rate", 0))
                current = rec.get_sampling_frequency()
                if rate > current:
                    warnings.warn(
                        f"resample to {rate:g} Hz is above the recording's "
                        f"{current:g} Hz; upsampling adds no information. "
                        f"Skipping it.",
                        stacklevel=3,
                    )
                    continue
                if rate == current:
                    continue

            rec = function(rec, **kwargs)

    return rec


def _as_call(step) -> dict:
    """Accept ``{"name": {...}}``, ``{"name": null}`` or a bare ``"name"``."""
    if isinstance(step, str):
        return {step: {}}
    if isinstance(step, dict):
        if len(step) == 1:
            name, kwargs = next(iter(step.items()))
            return {name: dict(kwargs or {})}
        # {"name": "notch_filter", "freq": 50} form
        step = dict(step)
        name = step.pop("name", None)
        if name:
            return {name: step}
    raise ValueError(
        f"Cannot read the preprocessing step {step!r}. Expected "
        f'{{"notch_filter": {{"freq": 50}}}} or {{"name": "notch_filter", "freq": 50}}.'
    )


def preprocess_recording(recording: Recording, params: dict) -> Recording:
    """Apply each channel's preprocessing, returning a new :class:`Recording`.

    Nothing is read or filtered here; the work happens lazily when a trace is
    requested. Decide what to configure by running :func:`nyx.check_signals`,
    which measures mains pickup and names the frequency to notch.
    """
    eeg = preprocess_channel(recording.eeg, params.get("EEG", {}))
    emg = (None if recording.emg is None
           else preprocess_channel(recording.emg, params.get("EMG", {})))

    eeg_fs = float(eeg.get_sampling_frequency())
    emg_fs = eeg_fs if emg is None else float(emg.get_sampling_frequency())

    return Recording(
        eeg=eeg,
        emg=emg,
        fs=eeg_fs,
        name=recording.name,
        source_path=recording.source_path,
        emg_fs_=(emg_fs if emg_fs != eeg_fs else None),
        source_format=recording.source_format,
        source_options=recording.source_options,
    )
