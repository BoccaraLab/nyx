"""Recipes for public datasets nyx has been run on.

A recipe knows one thing: how to turn ``(data_root, recording_name)`` into a
concrete recording file, a concrete scoring file, and the reader settings each
needs. Nothing in the core pipeline depends on this package -- it exists so
that the published analyses can be reproduced, and as worked examples of how to
wire up an awkward folder layout.

**You do not need a recipe to use nyx on your own data.** Point
:func:`nyx.io.read_recording` and :func:`nyx.io.read_annotations` at your files
directly.

Usage::

    from nyx.datasets import load_dataset, list_datasets

    list_datasets()
    recording, annotations = load_dataset(
        "gulledge-2025",
        data_root="/data",
        sub_dataset="Saline (N=4)",
        recording_name="A31_D04",
    )
"""

from nyx.datasets.builtin import (
    DATASETS,
    DatasetSpec,
    describe_dataset,
    list_datasets,
    load_dataset,
    resolve_dataset,
)

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "describe_dataset",
    "list_datasets",
    "load_dataset",
    "resolve_dataset",
]
