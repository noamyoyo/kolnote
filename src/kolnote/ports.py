"""Contracts every adapter implements. Core code depends only on these."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, Self, runtime_checkable

from .models import Incoming, Transcript


@runtime_checkable
class Channel(Protocol):
    """A medium voice notes arrive on and transcripts are sent back to."""

    name: str

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self: ...

    def messages(self) -> AsyncIterator[Incoming]: ...

    async def reply(self, to: Incoming, text: str) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class STTEngine(Protocol):
    """Turns audio bytes into text."""

    name: str

    @classmethod
    def from_config(cls, options: dict[str, Any]) -> Self: ...

    async def transcribe(
        self, audio: bytes, *, mime_type: str, language: str | None
    ) -> Transcript: ...

    async def close(self) -> None: ...


@runtime_checkable
class Service(Protocol):
    """A long-running listener that shares the engine with the channels (e.g. a Wyoming server)."""

    async def serve_forever(self) -> None: ...

    async def close(self) -> None: ...
