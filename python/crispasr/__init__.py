"""CrispASR — lightweight speech recognition via ggml."""

from ._binding import (
    CrispASR,
    DiarizeMethod,
    DiarizeSegment,
    Segment,
    Session,
    SessionSegment,
    SessionWord,
    diarize_segments,
)

__all__ = [
    "CrispASR",
    "DiarizeMethod",
    "DiarizeSegment",
    "Segment",
    "Session",
    "SessionSegment",
    "SessionWord",
    "diarize_segments",
]
__version__ = "0.4.5"
