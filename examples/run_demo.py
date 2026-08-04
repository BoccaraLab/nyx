"""Run nyx end to end on synthetic data and write out every figure.

Needs no data and no downloads, so it is the quickest way to check that an
installation works and to see what nyx produces::

    python examples/run_demo.py

Writes to ``examples/demo_output/``:

    clean/plots/    scoring of a clean recording
    messy/plots/    the same after finding and fixing signal problems
    messy/signal_check.png

Two runs, because they show different things. The first is the happy path. The
second starts from a recording with mains interference and saturated signal at
both ends -- the two problems you actually meet -- and shows the signal check
finding them, then what fixing them is worth.

Note on the accuracy: the signal is synthesised to contain exactly the
structure nyx looks for, so it scores near-perfectly by construction. This
demonstrates the workflow, not the method's performance. For that, run one of
the public datasets (see nyx.datasets.list_datasets).
"""

from __future__ import annotations

import os

# Draw to files rather than windows, so this runs on a headless machine.
import matplotlib

matplotlib.use("Agg")

import nyx

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "demo_output")


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def clean_run() -> float:
    """Score a clean recording, the happy path."""
    rule("1. A clean recording")

    recording, truth = nyx.demo_recording()
    print(recording.describe())

    result = nyx.score_recording(
        recording, nyx.demo_params(), reference=truth, verbose=False
    )
    print()
    print(result.agreement.summary())

    out = os.path.join(OUTPUT, "clean")
    nyx.save_results(result, out)
    print(f"\nwrote {out}")
    return result.agreement.mf1


def messy_run() -> tuple[float, float]:
    """Find and fix signal problems, then score."""
    rule("2. A recording that needs a notch filter and trimming")

    recording, truth = nyx.demo_messy_recording()
    print(recording.describe())

    # --- what is wrong with it? ------------------------------------------
    print("\n--- signal check ---")
    check = nyx.check_signals(recording, preview=None)
    print(check.summary())

    out = os.path.join(OUTPUT, "messy")
    os.makedirs(out, exist_ok=True)
    figure = check.plot()
    figure.savefig(os.path.join(out, "signal_check.png"), dpi=140,
                   bbox_inches="tight")

    notch = check.suggested_notch()

    # --- does the notch actually help? -----------------------------------
    after = nyx.check_signals(recording, preview=None, notch=notch)
    print("\n--- with the notch applied ---")
    for channel in ("EEG", "EMG"):
        key = f"{channel}_{notch:g}Hz"
        print(f"  {channel} at {notch:g} Hz: "
              f"{check.line_noise[key]:+.1f} dB -> {after.line_noise[key]:+.1f} dB")

    # --- score it twice, to show what the fixes are worth -----------------
    naive = nyx.score_recording(
        recording, nyx.demo_params(), reference=truth, verbose=False
    )

    params = nyx.demo_params()
    params["EEG"]["notch"] = notch
    params["EMG"]["notch"] = notch

    # The saturated stretches the demo added at each end. On real data you
    # would read these off the signal-check figure.
    artefact = 600.0
    window = (artefact, recording.duration - artefact)
    print(f"\ntrimming to {window[0]:.0f}-{window[1]:.0f}s "
          f"of {recording.duration:.0f}s")

    fixed = nyx.score_recording(
        recording, params, window=window, reference=truth, verbose=False
    )
    print()
    print(fixed.agreement.summary())

    nyx.save_results(fixed, out)
    print(f"\nwrote {out}")
    return naive.agreement.mf1, fixed.agreement.mf1


def main() -> None:
    print(f"nyx {nyx.__version__}")
    clean_mf1 = clean_run()
    naive_mf1, fixed_mf1 = messy_run()

    rule("Summary")
    print(f"  clean recording                        MF1 {clean_mf1:.3f}")
    print(f"  messy recording, nothing fixed         MF1 {naive_mf1:.3f}")
    print(f"  messy recording, notch + trim applied  MF1 {fixed_mf1:.3f}")
    print(f"\nFigures are in {OUTPUT}")
    print("Start with plots/summary.png in each folder.")


if __name__ == "__main__":
    main()
