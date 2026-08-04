"""Rules applied to a finished hypnogram, before it is saved or compared.

Clustering scores each epoch on its own, so it has no notion of what a plausible
*sequence* of stages looks like. These rules add that back: they rewrite bouts
that are implausible given their neighbours, and they merge away segments too
short to be a real stage.

They are **off by default**. Every one of them trades agreement for
plausibility, and which trade is right depends on what the scoring is for --
see :ref:`what the rules cost <cost>` below.

Configure them in the params, as an ordered list::

    "postprocess": [
      {"rem_after_wake": {"min_wake_duration": 0}},
      {"min_duration":   {"seconds": 4}}
    ]

Order matters and is the order you write, exactly as in ``preprocess``. Because
the list lives in the params it is written into the run record and replays.

The rules
---------

============================  =============================================
rule                          effect
============================  =============================================
``rem_flanked_by_wake``       REM with WAKE on *both* sides becomes WAKE.
                              ``max_duration`` limits it to short REM bouts.
``rem_after_wake``            REM *following* a WAKE bout becomes WAKE.
                              ``min_wake_duration`` limits it to REM that
                              follows a long enough wake bout.
``min_duration``              No segment shorter than ``seconds``. Short
                              segments are absorbed into their neighbours.
============================  =============================================

Correspondence with the rule comparison in the paper, which is what these
defaults were chosen from:

======  ================================================================
R1      ``rem_flanked_by_wake``
R2      ``rem_flanked_by_wake`` with ``max_duration: 20``
R3      ``rem_after_wake`` with ``min_wake_duration: 20``
R4      ``rem_after_wake``
R5      ``min_duration`` with ``seconds: 4``
R6      ``rem_after_wake`` then ``min_duration``
R7      ``min_duration`` then ``rem_after_wake``
======  ================================================================

.. _cost:

What the rules cost
-------------------

Global agreement barely moves -- Cohen's kappa sat at ~0.83 for every rule in
the paper's comparison, including no rule at all. The rules are not an accuracy
story. What they change is the biology:

* ``rem_after_wake`` is the REM knob. Unruled scoring over-estimates REM, and
  removing REM that follows wake pulls the REM/NREM ratio towards the manual
  value -- at the cost of moving total sleep time slightly further from it.
* ``min_duration`` is the microarousal knob. Merging sub-threshold wake bouts
  drops the microarousal count by roughly a sixth, but it overshoots: unruled
  scoring over-estimates them and this under-estimates them.

Pick the rule that protects the readout your analysis depends on, and say in
the methods which one you used. ``run.json`` records it either way.

NOSIGNAL
--------

Artefact and dropout segments are **transparent**: a NOSIGNAL segment between
two WAKE bouts does not stop them counting as neighbours, and no rule ever
relabels NOSIGNAL itself. Without that, a single artefact epoch would silently
switch a rule off for the bout it interrupts. Change it with ``ignore``.

Human recordings are scored in 30 s epochs, so ``min_duration`` there wants
``seconds: 30`` rather than the rodent default of 4.
"""

from __future__ import annotations

import numpy as np

from nyx.io.annotations import merge_consecutive

__all__ = [
    "RULES",
    "apply_rules",
    "rules_from_params",
    "rem_flanked_by_wake",
    "rem_after_wake",
    "min_duration",
]

#: Labels that are transparent to the rules: they neither block a neighbour
#: relationship nor get rewritten themselves.
DEFAULT_IGNORE = ("NOSIGNAL",)


def _parts(hypnogram: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return (
        np.asarray(hypnogram["time"], dtype="float64"),
        np.asarray(hypnogram["duration"], dtype="float64"),
        [str(label) for label in np.asarray(hypnogram["label"], dtype="U")],
    )


def _rebuild(times, durations, labels) -> dict[str, np.ndarray]:
    return merge_consecutive({
        "time": np.asarray(times, dtype="float64"),
        "duration": np.asarray(durations, dtype="float64"),
        "label": np.asarray(labels, dtype="U"),
    })


def _visible(labels: list[str], ignore: tuple[str, ...]) -> list[int]:
    """Indices of the segments a rule can see, in order."""
    return [i for i, label in enumerate(labels) if label not in ignore]


def rem_flanked_by_wake(
    hypnogram: dict,
    max_duration: float | None = None,
    stage: str = "REM",
    into: str = "WAKE",
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
) -> dict[str, np.ndarray]:
    """REM with WAKE on both sides becomes WAKE (paper rule R1/R2).

    Parameters
    ----------
    max_duration
        Only rewrite REM bouts shorter than this many seconds. ``None`` (the
        default) rewrites them whatever their length -- R1. The paper's R2 used
        20 s, on the argument that a long REM bout between two wake bouts is
        more likely to be real than a short one.

    This is the rule that used to be called ``apply_forbidden_transitions``.
    """
    times, durations, labels = _parts(hypnogram)
    visible = _visible(labels, ignore)

    for position in range(1, len(visible) - 1):
        index = visible[position]
        if labels[index] != stage:
            continue
        if max_duration is not None and durations[index] >= max_duration:
            continue
        if (labels[visible[position - 1]] == into
                and labels[visible[position + 1]] == into):
            labels[index] = into

    return _rebuild(times, durations, labels)


def rem_after_wake(
    hypnogram: dict,
    min_wake_duration: float = 0.0,
    stage: str = "REM",
    after: str = "WAKE",
    into: str = "WAKE",
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
) -> dict[str, np.ndarray]:
    """REM following a WAKE bout becomes WAKE (paper rule R3/R4).

    Sleep-onset REM is rare outside narcolepsy, so REM scored straight out of
    wake is usually a clustering error -- typically a wake bout whose spectrum
    is theta-dominated.

    Parameters
    ----------
    min_wake_duration
        Only rewrite REM whose preceding WAKE bout is *longer* than this many
        seconds. ``0`` (the default) means any preceding wake bout -- R4. The
        paper's R3 used 20 s.

    The preceding bout's duration is read from the input, so a run of rewritten
    segments does not cascade: one left-to-right pass, as in the paper.
    """
    times, durations, labels = _parts(hypnogram)
    visible = _visible(labels, ignore)
    original = list(labels)

    for position in range(1, len(visible)):
        index, previous = visible[position], visible[position - 1]
        if (labels[index] == stage
                and original[previous] == after
                and durations[previous] > min_wake_duration):
            labels[index] = into

    return _rebuild(times, durations, labels)


def min_duration(
    hypnogram: dict,
    seconds: float = 4.0,
    order: tuple[str, ...] = ("REM", "NREM", "WAKE"),
    fallback: dict[str, str] | None = None,
    default_fallback: str = "WAKE",
    ignore: tuple[str, ...] = DEFAULT_IGNORE,
) -> dict[str, np.ndarray]:
    """No segment shorter than ``seconds`` survives (paper rule R5).

    Each stage in ``order`` is cleaned in turn, re-merging in between so
    neighbour labels stay current. A short segment is absorbed into its
    neighbours when both carry the same label; otherwise it takes the label
    ``fallback`` gives for its stage, or ``default_fallback``. A segment at
    either end of the recording has one neighbour, and takes it.

    The defaults reproduce the paper: short REM and NREM go to WAKE unless
    they sit inside a single other stage, and short WAKE goes to NREM -- a
    micro-arousal in the middle of sleep is wake that is too brief to score.
    The one deliberate difference is the boundary case above, where the
    paper's fallback chain can cycle and leave a sub-threshold segment alive.

    Parameters
    ----------
    seconds
        4 for rodents (the epoch length the rodent params use); 30 for human
        recordings, which are scored in 30 s epochs.
    order
        Which stages to clean, in which order. Stages not listed are left
        alone, however short.
    """
    fallback = {"WAKE": "NREM"} if fallback is None else dict(fallback)
    current = merge_consecutive(hypnogram)

    for stage in order:
        times, durations, labels = _parts(current)
        visible = _visible(labels, ignore)
        rewritten = list(labels)

        for position, index in enumerate(visible):
            if labels[index] != stage or durations[index] >= seconds:
                continue
            before = labels[visible[position - 1]] if position > 0 else None
            after = (labels[visible[position + 1]]
                     if position < len(visible) - 1 else None)
            if before is None or after is None:
                # First or last segment: only one neighbour to merge into, so
                # take it. Sending it to the fallback instead can leave a
                # sub-threshold segment alive: a short NREM at the end next to
                # REM would go to WAKE on the NREM pass and straight back to
                # NREM on the WAKE pass, and no later pass looks at it again.
                rewritten[index] = before or after or stage
            elif before == after:
                rewritten[index] = before
            else:
                rewritten[index] = fallback.get(stage, default_fallback)

        current = _rebuild(times, durations, rewritten)

    return current


#: Rule name -> function. Add to this to register your own; anything callable
#: as ``rule(hypnogram, **kwargs) -> hypnogram`` will do.
RULES = {
    "rem_flanked_by_wake": rem_flanked_by_wake,
    "rem_after_wake": rem_after_wake,
    "min_duration": min_duration,
}


def _as_call(rule) -> tuple[str, dict]:
    """Accept ``{"name": {...}}``, ``{"name": null}`` or a bare ``"name"``."""
    if isinstance(rule, str):
        return rule, {}
    if isinstance(rule, dict):
        if len(rule) == 1:
            name, kwargs = next(iter(rule.items()))
            return str(name), dict(kwargs or {})
        rule = dict(rule)
        name = rule.pop("name", None)
        if name:
            return str(name), rule
    raise ValueError(
        f"Cannot read the postprocessing rule {rule!r}. Expected "
        f'{{"min_duration": {{"seconds": 4}}}} or {{"name": "min_duration", "seconds": 4}}.'
    )


def rules_from_params(params: dict) -> list:
    """The ``postprocess`` list from a params dict, or an empty list."""
    return list(params.get("postprocess") or [])


def apply_rules(hypnogram: dict, rules, verbose: bool = False) -> dict[str, np.ndarray]:
    """Apply an ordered list of rules to a hypnogram.

    ``rules`` is the ``postprocess`` list from the params. Returns the
    hypnogram unchanged when the list is empty, so this is safe to call
    unconditionally.
    """
    if not rules:
        return hypnogram

    for rule in rules:
        name, kwargs = _as_call(rule)
        function = RULES.get(name)
        if function is None:
            raise ValueError(
                f"Unknown postprocessing rule {name!r}. Available: "
                f"{sorted(RULES)}. Register your own by adding it to "
                f"nyx.postprocess.RULES."
            )
        before = _stage_seconds(hypnogram)
        hypnogram = function(hypnogram, **kwargs)
        if verbose:
            print(f"  {name}: {_changes(before, _stage_seconds(hypnogram))}")

    return hypnogram


def _stage_seconds(hypnogram: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for duration, label in zip(hypnogram["duration"], hypnogram["label"]):
        totals[str(label)] = totals.get(str(label), 0.0) + float(duration)
    return totals


def _changes(before: dict[str, float], after: dict[str, float]) -> str:
    moved = {
        stage: after.get(stage, 0.0) - before.get(stage, 0.0)
        for stage in {**before, **after}
        if abs(after.get(stage, 0.0) - before.get(stage, 0.0)) > 1e-9
    }
    if not moved:
        return "no change"
    return ", ".join(f"{stage} {delta:+.0f}s" for stage, delta in sorted(moved.items()))
