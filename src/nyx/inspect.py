"""Step 0: look at the signal before scoring anything.

Plots the EEG and EMG traces together with their spectrograms, which is what
you need to answer the three questions that come before any scoring:

* **Is the signal usable?** Flat stretches, saturation and dead channels are
  obvious in a spectrogram and easy to miss in a trace.
* **Does it need a notch?** Mains pickup shows as a hard horizontal line at
  50 or 60 Hz. :meth:`SignalCheck.summary` measures it so you do not have to
  judge by eye.
* **Where should the analysis window start and end?** Handling artefacts at the
  beginning and end of a recording are usually visible immediately.

Long recordings are previewed at the start and end rather than in full, because
that is where the artefacts are and because plotting 24 h of raw trace is
neither fast nor readable. Short recordings (a single night of human PSG) are
shown whole.

Usage::

    check = nyx.check_signals(recording)
    check.plot()
    print(check.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import spectrogram as _scipy_spectrogram

from nyx.preprocessing import apply_notch, line_noise_score
from nyx.types import Recording

__all__ = ["SignalCheck", "SignalPreview", "check_signals"]

# Mains frequencies worth testing for: 50 Hz almost everywhere, 60 Hz in North
# America and Japan. Both are checked so the data tells you which applies.
_MAINS_FREQUENCIES = (50.0, 60.0)

# A line-noise score above this many dB is worth notching out.
LINE_NOISE_THRESHOLD_DB = 3.0


@dataclass
class SignalPreview:
    """One window of the recording, as traces plus spectrograms."""

    name: str  # "start", "end" or "full"
    start: float
    end: float
    times: np.ndarray
    eeg: np.ndarray
    emg: np.ndarray
    eeg_spectrogram: tuple[np.ndarray, np.ndarray, np.ndarray]  # (Sxx, freqs, times)
    emg_spectrogram: tuple[np.ndarray, np.ndarray, np.ndarray]

    def mean_spectrum(self, channel: str = "eeg") -> tuple[np.ndarray, np.ndarray]:
        """Time-averaged spectrum, as ``(freqs, power)``."""
        Sxx, freqs, _ = self.eeg_spectrogram if channel == "eeg" else self.emg_spectrogram
        return freqs, Sxx.mean(axis=1)


@dataclass
class SignalCheck:
    """Result of :func:`check_signals`."""

    recording_name: str
    fs: float
    duration: float
    previews: list[SignalPreview]
    notch_applied: float | None = None
    line_noise: dict[str, float] = field(default_factory=dict)

    def suggested_notch(self) -> float | None:
        """The mains frequency worth notching out, or ``None`` if the signal is clean."""
        candidates = {
            float(key.split("_")[-1].rstrip("Hz")): value
            for key, value in self.line_noise.items()
            if np.isfinite(value)
        }
        if not candidates:
            return None
        best = max(candidates, key=candidates.get)
        return best if candidates[best] >= LINE_NOISE_THRESHOLD_DB else None

    def summary(self) -> str:
        lines = [
            f"{self.recording_name or 'recording'}: "
            f"{self.duration:.0f}s at {self.fs:g} Hz",
            "  previewed: " + ", ".join(f"{p.name} {p.start:.0f}-{p.end:.0f}s"
                                         for p in self.previews),
        ]
        if self.notch_applied:
            lines.append(f"  notch applied at {self.notch_applied:g} Hz")

        if self.line_noise:
            lines.append("  line-noise check (peak above local baseline):")
            for key, value in sorted(self.line_noise.items()):
                if not np.isfinite(value):
                    continue
                verdict = "  <- consider a notch" if value >= LINE_NOISE_THRESHOLD_DB else ""
                lines.append(f"    {key:<14} {value:+6.1f} dB{verdict}")

        suggestion = self.suggested_notch()
        if suggestion:
            lines.append(
                f"  suggestion: set \"notch\": {suggestion:g} in the EEG/EMG params."
            )
        elif self.line_noise:
            lines.append("  suggestion: no notch needed.")
        return "\n".join(lines)

    def plot(
        self,
        fmax: float | None = None,
        figsize: tuple[float, float] | None = None,
        max_points: int = 4000,
    ):
        """Draw traces and spectrograms, one column per previewed window.

        ``max_points`` caps how many points each trace is drawn with. A 10000 s
        preview at 500 Hz is 5 million samples, which is slow to render and
        indistinguishable from the decimated version -- the envelope keeps
        spikes visible, so artefacts still stand out.
        """
        import matplotlib.pyplot as plt

        n = len(self.previews)
        figsize = figsize or (9 * n, 10)
        fig, axes = plt.subplots(4, n, figsize=figsize, squeeze=False)

        for col, preview in enumerate(self.previews):
            for row, (channel, trace) in enumerate(
                (("EEG", preview.eeg), ("EMG", preview.emg))
            ):
                ax_trace = axes[row * 2][col]
                ax_spec = axes[row * 2 + 1][col]

                t_plot, y_plot = _decimate_envelope(preview.times, trace, max_points)
                ax_trace.plot(t_plot, y_plot, lw=0.3, color="#333333")
                ax_trace.set_xlim(preview.times[0], preview.times[-1])
                ax_trace.set_ylabel(f"{channel}")
                ax_trace.set_xticks([])
                for spine in ax_trace.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax_trace.set_title(
                        f"{preview.name}  ({preview.start:.0f}-{preview.end:.0f}s)"
                    )

                Sxx, freqs, times = (
                    preview.eeg_spectrogram if channel == "EEG" else preview.emg_spectrogram
                )
                top = fmax or freqs[-1]
                band = freqs <= top
                mesh = ax_spec.pcolormesh(
                    times + preview.start,
                    freqs[band],
                    Sxx[band],
                    shading="nearest",
                    cmap="jet",
                    rasterized=True,
                )
                # Robust limits: a few saturated bins otherwise wash the image out.
                finite = Sxx[band][np.isfinite(Sxx[band])]
                if finite.size:
                    mesh.set_clim(*np.percentile(finite, [5, 99]))
                ax_spec.set_ylabel(f"{channel} freq (Hz)")
                fig.colorbar(mesh, ax=ax_spec, pad=0.01)

                for mains in _MAINS_FREQUENCIES:
                    if mains <= top:
                        ax_spec.axhline(mains, color="w", ls=":", lw=0.8, alpha=0.6)

            axes[-1][col].set_xlabel("Time (s)")

        fig.suptitle(
            f"Signal check - {self.recording_name or 'recording'}"
            + (f"  (notch {self.notch_applied:g} Hz applied)" if self.notch_applied else ""),
            fontweight="bold",
        )
        fig.tight_layout()
        return fig


def check_signals(
    recording: Recording,
    preview: float = 10000.0,
    notch: float | None = None,
    fmax: float | None = None,
    binsize: float = 4.0,
    return_in_uV: bool = False,
) -> SignalCheck:
    """Inspect a recording before scoring it.

    Parameters
    ----------
    recording
        The recording to look at, before any trimming.
    preview
        Seconds to show from the start and from the end. Recordings shorter
        than ``2 * preview`` are shown whole in a single panel. Set to ``None``
        or ``0`` to always show everything.
    notch
        Apply a notch at this frequency before computing the preview, to check
        whether it cleans the signal up. This does **not** change the recording;
        put ``"notch"`` in the params to apply it during scoring.
    fmax
        Highest frequency to compute. Defaults to just under Nyquist, capped at
        128 Hz, which keeps both mains frequencies and their first harmonics in
        view.
    binsize
        Spectrogram window in seconds.

    Returns
    -------
    SignalCheck
    """
    duration = recording.duration
    fs = recording.fs
    fmax = fmax or min(fs / 2.0 * 0.99, 128.0)

    if not preview or duration <= 2 * preview:
        windows = [("full", 0.0, duration)]
    else:
        windows = [("start", 0.0, preview), ("end", duration - preview, duration)]

    previews: list[SignalPreview] = []
    for name, start, end in windows:
        window = recording.time_slice(start, end)
        eeg = window.eeg_trace(return_in_uV=return_in_uV)
        emg = window.emg_trace(return_in_uV=return_in_uV)

        if notch:
            eeg = apply_notch(eeg, fs, notch)
            emg = apply_notch(emg, fs, notch)

        previews.append(
            SignalPreview(
                name=name,
                start=start,
                end=end,
                times=start + np.arange(len(eeg)) / fs,
                eeg=eeg,
                emg=emg,
                eeg_spectrogram=_preview_spectrogram(eeg, fs, binsize, fmax),
                emg_spectrogram=_preview_spectrogram(emg, fs, binsize, fmax),
            )
        )

    check = SignalCheck(
        recording_name=recording.name,
        fs=fs,
        duration=duration,
        previews=previews,
        notch_applied=notch,
    )

    # Score mains pickup on the first previewed window, averaged over time.
    first = previews[0]
    for channel in ("eeg", "emg"):
        freqs, spectrum = first.mean_spectrum(channel)
        for mains in _MAINS_FREQUENCIES:
            if mains < fmax:
                check.line_noise[f"{channel.upper()}_{mains:g}Hz"] = line_noise_score(
                    freqs, spectrum, mains
                )
    return check


def _decimate_envelope(
    times: np.ndarray, trace: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to ~``max_points`` while keeping its visual envelope.

    Plain subsampling would drop the spikes that make artefacts visible, so
    each output block contributes both its minimum and its maximum.
    """
    n = len(trace)
    if max_points <= 0 or n <= max_points:
        return times, trace

    n_blocks = max(max_points // 2, 1)
    block = n // n_blocks
    if block < 2:
        return times, trace

    usable = n_blocks * block
    blocks = trace[:usable].reshape(n_blocks, block)
    lows, highs = blocks.min(axis=1), blocks.max(axis=1)

    # Interleave min and max so each block draws as a vertical span.
    y = np.empty(n_blocks * 2, dtype=float)
    y[0::2], y[1::2] = lows, highs
    t = np.repeat(times[:usable:block], 2)
    return t, y


def _preview_spectrogram(
    trace: np.ndarray, fs: float, binsize: float, fmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A plain dB spectrogram, deliberately independent of the scoring params.

    The point of the check is to see the signal as it is, so this does not apply
    the normalisation or smoothing the scoring features use.
    """
    nperseg = max(int(binsize * fs), 8)
    nperseg = min(nperseg, len(trace))
    freqs, times, Sxx = _scipy_spectrogram(
        trace,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    band = freqs <= fmax
    Sxx = 10.0 * np.log10(Sxx[band] + np.finfo(float).eps)
    return Sxx, freqs[band], times
