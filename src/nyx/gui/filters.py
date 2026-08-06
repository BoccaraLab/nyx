"""File-dialog filters, derived from the reader registries.

Nothing here hardcodes a format. Both registries are open --
:func:`nyx.io.register_recording_reader` and
:func:`nyx.io.register_annotation_reader` -- so a lab that adds a reader for
its own acquisition system should find it in the GUI's file dialog without
touching the GUI. That only works if the dialog asks the registry.

Qt-free by design, like the rest of the session layer: these are strings.
"""

from __future__ import annotations

from nyx.io import (
    ANNOTATION_READERS,
    CHANNEL_LISTERS,
    RECORDING_EXTENSIONS,
    RECORDING_READERS,
)

__all__ = [
    "recording_formats",
    "annotation_formats",
    "recording_filters",
    "annotation_filters",
    "extensions_for",
    "format_is_folder",
    "can_list_channels",
]

#: Formats read from a directory rather than a file. spikeinterface saves a
#: folder, and ``_infer_format`` branches on ``os.path.isdir``, so a dialog
#: offering it has to ask for a directory.
FOLDER_FORMATS = frozenset({"spikeinterface"})

#: Extensions for annotation formats that cannot be guessed from the path.
#: ``read_annotations`` infers .hyp and .csv itself but raises for the rest,
#: so these exist to keep the dialog honest about what it can open.
ANNOTATION_EXTENSIONS = {
    "interval_csv": (".csv", ".txt", ".tsv"),
    "epoch_csv": (".csv", ".txt", ".tsv"),
    "column_csv": (".csv", ".txt", ".tsv"),
    "epoch_npy": (".npy",),
    "epoch_mat": (".mat",),
    "nsrr_xml": (".xml",),
    "visbrain_hyp": (".hyp",),
}


def recording_formats() -> list[str]:
    """Every registered recording format, in a stable order."""
    return sorted(RECORDING_READERS)


def annotation_formats() -> list[str]:
    """Every registered annotation format, in a stable order."""
    return sorted(ANNOTATION_READERS)


def format_is_folder(format: str) -> bool:
    """Whether this format is read from a directory rather than a file."""
    return format in FOLDER_FORMATS


def can_list_channels(format: str) -> bool:
    """Whether the channels of this format can be offered as a list."""
    return format in CHANNEL_LISTERS


def extensions_for(format: str, *, annotations: bool = False) -> tuple[str, ...]:
    """Extensions belonging to a format, ``()`` when it has none registered."""
    if annotations:
        return tuple(ANNOTATION_EXTENSIONS.get(format, ()))
    return tuple(
        sorted(ext for ext, name in RECORDING_EXTENSIONS.items() if name == format)
    )


def _filter(label: str, extensions) -> str:
    patterns = " ".join(f"*{ext}" for ext in extensions) if extensions else "*"
    return f"{label} ({patterns})"


def _filters(formats, *, annotations: bool, everything: str) -> str:
    known: list[str] = []
    entries: list[str] = []

    for format in formats:
        if not annotations and format_is_folder(format):
            continue  # picked with a directory dialog, not this one
        extensions = extensions_for(format, annotations=annotations)
        if not extensions:
            continue  # nothing to match on; "All files" still covers it
        known.extend(extensions)
        entries.append(_filter(format, extensions))

    ordered = sorted(set(known))
    return ";;".join([_filter(everything, ordered), *entries, "All files (*)"])


def recording_filters() -> str:
    """A Qt name filter string covering every readable recording format."""
    return _filters(recording_formats(), annotations=False, everything="Recordings")


def annotation_filters() -> str:
    """A Qt name filter string covering every readable annotation format."""
    return _filters(annotation_formats(), annotations=True, everything="Scorings")
