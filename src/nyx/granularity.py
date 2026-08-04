"""Collapsing a hypnogram to a coarser set of stages.

Scoring runs once at the finest granularity it can reach; coarser hypnograms
are then produced by merging labels, not by clustering again. So a human run
that resolves five stages also yields the four- and three-stage versions for
free, and they are guaranteed to be consistent with each other.

Which granularity you report depends on what you are comparing against -- many
rodent references only score three stages, and inter-scorer agreement on N1 is
poor enough that four stages is often the fairer comparison for humans.
"""

from __future__ import annotations

import numpy as np

from nyx.metrics import merge_states

__all__ = ["GRANULARITIES", "collapse", "available_granularities"]


#: Merge maps keyed by the number of stages they produce.
#:
#: 4-stage folds N1 into N2 (they are the hardest pair for human scorers to
#: separate, and N1 is a small fraction of the night). 3-stage folds all NREM
#: together, which is the vocabulary rodent scoring uses.
GRANULARITIES: dict[int, dict[str, str]] = {
    5: {},
    4: {
        "NREM1": "NREM2",
        "N1": "NREM2",
    },
    3: {
        "NREM1": "NREM",
        "NREM2": "NREM",
        "NREM3": "NREM",
        "N1": "NREM",
        "N2": "NREM",
        "N3": "NREM",
    },
}


def collapse(hypnogram: dict, n_stages: int) -> dict[str, np.ndarray]:
    """Merge a hypnogram down to ``n_stages`` distinct sleep stages.

    Parameters
    ----------
    hypnogram
        A hypnogram dict, typically from a 5-stage human run.
    n_stages
        5 (unchanged), 4 (N1 folded into NREM2) or 3 (all NREM merged).

    Returns
    -------
    dict
        A new hypnogram; adjacent intervals that end up sharing a label are
        merged into one.
    """
    if n_stages not in GRANULARITIES:
        raise ValueError(
            f"n_stages must be one of {sorted(GRANULARITIES)}; got {n_stages}."
        )
    merge_map = GRANULARITIES[n_stages]
    if not merge_map:
        return {key: np.asarray(value) for key, value in hypnogram.items()}
    return merge_states(hypnogram, merge_map)


#: Labels that are not sleep stages and so do not count towards granularity.
_NON_STAGES = {"NOSIGNAL", "UNKNOWN", "UNCLASSIFIED", "ARTEFACT"}


def available_granularities(hypnogram: dict) -> list[int]:
    """Which granularities this hypnogram can be reported at.

    Determined by how many distinct stages it actually resolves: a 3-stage
    rodent scoring returns ``[3]`` rather than pretending 4- and 5-stage
    versions exist, while a 5-stage human scoring returns ``[5, 4, 3]``.

    Returns an empty list if the hypnogram resolves fewer stages than any
    standard granularity -- a bare wake/sleep split, for instance.
    """
    labels = set(np.asarray(hypnogram["label"]).tolist()) - _NON_STAGES
    return [n for n in sorted(GRANULARITIES, reverse=True) if n <= len(labels)]
