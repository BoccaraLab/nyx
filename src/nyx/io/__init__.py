"""Reading recordings and annotations into nyx's internal formats."""

from nyx.io.annotations import (
    ANNOTATION_READERS,
    read_annotations,
    register_annotation_reader,
)
from nyx.io.neo import (
    ChannelInfo,
    describe_channels,
    guess_neo_format,
    neo_formats,
    neo_streams,
    recording_streams,
)
from nyx.io.recordings import (
    CHANNEL_LISTERS,
    RECORDING_EXTENSIONS,
    RECORDING_READERS,
    list_channels,
    read_recording,
    register_channel_lister,
    register_recording_reader,
)

__all__ = [
    "read_recording",
    "list_channels",
    "recording_streams",
    "describe_channels",
    "ChannelInfo",
    "neo_formats",
    "guess_neo_format",
    "neo_streams",
    "register_recording_reader",
    "register_channel_lister",
    "RECORDING_READERS",
    "RECORDING_EXTENSIONS",
    "CHANNEL_LISTERS",
    "read_annotations",
    "register_annotation_reader",
    "ANNOTATION_READERS",
]
