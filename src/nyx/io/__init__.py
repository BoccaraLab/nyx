"""Reading recordings and annotations into nyx's internal formats."""

from nyx.io.annotations import (
    ANNOTATION_READERS,
    read_annotations,
    register_annotation_reader,
)
from nyx.io.recordings import (
    RECORDING_READERS,
    read_recording,
    register_recording_reader,
)

__all__ = [
    "read_recording",
    "register_recording_reader",
    "RECORDING_READERS",
    "read_annotations",
    "register_annotation_reader",
    "ANNOTATION_READERS",
]
