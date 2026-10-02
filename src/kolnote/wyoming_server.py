"""Wyoming speech-to-text service, so Home Assistant (or any Wyoming client) can use the engine.

Needs the optional `wyoming` package (`pip install "kolnote[wyoming]"`). It runs inside the same
process as the channels (`kolnote run` with a `[wyoming]` section) and shares their engine, so a
model loaded for a voice note is the one Home Assistant uses, and the other way round.

    [wyoming]
    host = "0.0.0.0"
    port = 10300
    languages = ["en"]         # only for a single engine; a `multi` engine advertises its own
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import Any

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .ports import STTEngine

log = logging.getLogger("kolnote.wyoming")

ATTRIBUTION = Attribution(name="kolnote", url="https://github.com/noamyoyo/kolnote")


def build_info(engine: STTEngine, language: str | None, languages: list[str] | None) -> Info:
    models: dict[str, list[str]] = dict(getattr(engine, "models", None) or {})
    if not models:
        codes = languages or ([language] if language else [])
        if not codes:
            raise ValueError("set [wyoming] languages (or a top-level language) so clients know what this serves")
        models = {engine.name: codes}
    return Info(
        asr=[
            AsrProgram(
                name="kolnote",
                description="kolnote speech-to-text",
                attribution=ATTRIBUTION,
                installed=True,
                version=None,
                models=[
                    AsrModel(
                        name=name,
                        description=name,
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=None,
                        languages=codes,
                    )
                    for name, codes in models.items()
                ],
            )
        ]
    )


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, engine: STTEngine, language: str | None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = info.event()
        self._engine = engine
        self._default_language = language
        self._language = language
        self._pcm = bytearray()
        self._format: tuple[int, int, int] | None = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
        elif Transcribe.is_type(event.type):
            self._language = Transcribe.from_event(event).language or self._default_language
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._format = (chunk.rate, chunk.width, chunk.channels)
            self._pcm += chunk.audio
        elif AudioStop.is_type(event.type):
            await self._transcribe()
            return False
        return True

    async def _transcribe(self) -> None:
        text = ""
        if self._format and self._pcm:
            rate, width, channels = self._format
            wav = io.BytesIO()
            with wave.open(wav, "wb") as out:
                out.setframerate(rate)
                out.setsampwidth(width)
                out.setnchannels(channels)
                out.writeframes(bytes(self._pcm))
            try:
                transcript = await self._engine.transcribe(wav.getvalue(), mime_type="audio/wav", language=self._language)
                text = transcript.text
                log.info(
                    "transcribed %.1fs of audio in %.2fs, %d chars, language=%s",
                    len(self._pcm) / (rate * width * channels), transcript.processing_s, len(text), self._language,
                )
            except Exception:
                log.exception("transcription failed")
        await self.write_event(Transcript(text=text).event())


class WyomingServer:
    def __init__(
        self,
        engine: STTEngine,
        *,
        host: str = "127.0.0.1",
        port: int = 10300,
        language: str | None = None,
        languages: list[str] | None = None,
    ) -> None:
        self._engine = engine
        self._language = language
        self._info = build_info(engine, language, languages)
        self._host, self.port = host, port
        self._server: AsyncServer | None = None

    async def start(self) -> None:
        self._server = AsyncServer.from_uri(f"tcp://{self._host}:{self.port}")
        await self._server.start(lambda reader, writer: Handler(self._info, self._engine, self._language, reader, writer))
        sockets = getattr(getattr(self._server, "_server", None), "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]
        log.info("wyoming server listening on %s:%s", self._host, self.port)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        await asyncio.Event().wait()

    async def close(self) -> None:
        if self._server:
            await self._server.stop()
