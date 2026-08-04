"""Sleep stage names, integer codes and plotting colours.

Everything in nyx that needs to know what a stage is called, or what colour to
draw it, imports from here -- so there is one place to change when adding a
stage.

Two representations are used:

* **Stage names** (``"WAKE"``, ``"NREM"``, ``"REM"``, ...) in hypnograms and in
  anything a user reads or writes.
* **Integer codes** internally, while classifying, because they are cheap to
  put in a numpy array. :data:`STATE_NAMES` converts back.
"""

from __future__ import annotations

__all__ = [
    "NOSIGNAL",
    "UNCLASSIFIED",
    "WAKE",
    "QW",
    "REM",
    "NREM1",
    "NREM2",
    "NREM3",
    "NREM",
    "SLEEP",
    "STATE_NAMES",
    "COLORS",
    "state_names",
    "colors_dict",
]

# Integer codes used while classifying.
NOSIGNAL = 2
UNCLASSIFIED = 1
WAKE = 0
QW = -1
REM = -2
NREM1 = -3
NREM2 = -4
NREM3 = -5
NREM = -6
SLEEP = -7

STATE_NAMES: dict[int, str] = {
    NOSIGNAL: "NOSIGNAL",
    UNCLASSIFIED: "UNCLASSIFIED",
    WAKE: "WAKE",
    QW: "QW",
    REM: "REM",
    NREM1: "NREM1",
    NREM2: "NREM2",
    NREM3: "NREM3",
    NREM: "NREM",
    SLEEP: "SLEEP",
}

# Colours used by every nyx plot. Previously there were two conflicting copies
# of this dict (one in the scoring code, one in the plotting code); this is the
# plotting one, which is what the figures actually used.
COLORS: dict[str, str] = {
    "NOSIGNAL": "#000000",
    "UNCLASSIFIED": "#5f5e5f",
    "MICROAWAKE": "#cfb474",
    "QW": "#cfb474",
    "WAKE": "#F09E05",
    "REM": "#b1d6b0",
    "NREM1": "#5a5db1",
    "NREM2": "#a38bc1",
    "NREM3": "#402864",
    "NREM": "#402864",
    "SLEEP": "#402864",
}

# Backwards-compatible aliases for the old lowercase names.
state_names = STATE_NAMES
colors_dict = COLORS
