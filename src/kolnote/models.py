from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


AUDIO_EXTENSIONS = {".ogg", ".opus", ".oga", ".mp3", ".m4a", ".wav", ".flac", ".aac", ".webm", ".mp4"}


@dataclass(frozen=True)
class IncomingAudio:
    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    audio: bytes
    mime_type: str = "audio/ogg"
    duration_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class UnsupportedMessage:
    """A message that is not audio (text, image, ...). The bot tells the sender it cannot handle it."""

    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    kind: str


Incoming = IncomingAudio | UnsupportedMessage


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str
    processing_s: float
    language: str | None = None
    duration_s: float | None = None
