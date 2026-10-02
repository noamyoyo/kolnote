"""Serve any STT engine over the OpenAI `/v1/audio/transcriptions` API.

One process holds the model; any number of channel processes (or any OpenAI-compatible
client) point their `openai_compat` engine at it. Dependency-free, like the OpenWA channel:
a small asyncio HTTP server, with the stdlib `email` parser for multipart bodies.

Only `file`, `language` and `response_format` (`json` or `text`) are honoured. `model` is
ignored: the server always uses the engine it was started with.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from email.parser import BytesParser
from email.policy import HTTP
from typing import Any

from .config import Settings
from .ports import STTEngine
from .registry import build_stt

log = logging.getLogger(__name__)

TRANSCRIBE_PATH = "/v1/audio/transcriptions"
MAX_HEADER_LINES = 100
DEFAULT_MAX_BODY_MB = 64
REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    411: "Length Required",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


def _json(status: int, payload: dict[str, Any]) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode(), "application/json"


def _error(status: int, message: str) -> tuple[int, bytes, str]:
    return _json(status, {"error": {"message": message}})


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], tuple[bytes, str] | None]:
    """Split a multipart/form-data body into text fields and the `file` part (bytes, mime type)."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("body is not multipart")
    fields: dict[str, str] = {}
    file: tuple[bytes, str] | None = None
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if name == "file":
            file = (payload, part.get_content_type())
        elif name:
            fields[str(name)] = payload.decode("utf-8", errors="replace")
    return fields, file


class TranscriptionServer:
    def __init__(
        self,
        engine: STTEngine,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        language: str | None = None,
        max_body_mb: float = DEFAULT_MAX_BODY_MB,
    ) -> None:
        self._engine = engine
        self._host, self.port = host, port
        self._api_key = api_key
        self._language = language
        self._max_body = int(max_body_mb * 1024 * 1024)
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self._host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("kolnote server listening on %s:%s (auth %s)", self._host, self.port, "on" if self._api_key else "off")

    async def serve_forever(self) -> None:
        assert self._server is not None, "call start() first"
        await self._server.serve_forever()

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await self._engine.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload, content_type = _error(500, "internal error")
        try:
            request_line = (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1").split()
            headers: dict[str, str] = {}
            while line := (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1").strip():
                if len(headers) >= MAX_HEADER_LINES:
                    raise ValueError("too many headers")
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
            if len(request_line) < 2:
                raise ValueError("bad request line")
            status, payload, content_type = await self._route(request_line[0], request_line[1].split("?")[0], headers, reader)
        except (TimeoutError, ValueError, asyncio.IncompleteReadError, ConnectionError):
            status, payload, content_type = _error(400, "bad request")
        except Exception:
            log.exception("request failed")
        try:
            writer.write(
                f"HTTP/1.1 {status} {REASONS.get(status, 'X')}\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        except ConnectionError:
            pass
        writer.close()

    def _authorized(self, headers: dict[str, str]) -> bool:
        if not self._api_key:
            return True
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), self._api_key)

    async def _route(
        self, method: str, path: str, headers: dict[str, str], reader: asyncio.StreamReader
    ) -> tuple[int, bytes, str]:
        if path == "/healthz":
            return _json(200, {"status": "ok"})
        if path != TRANSCRIBE_PATH:
            return _error(404, "not found")
        if method != "POST":
            return _error(405, "use POST")
        if not self._authorized(headers):
            return _error(401, "invalid or missing API key")
        if "transfer-encoding" in headers or "content-length" not in headers:
            return _error(411, "Content-Length is required")
        length = int(headers["content-length"])
        if length <= 0:
            return _error(400, "empty body")
        if length > self._max_body:
            return _error(413, f"body larger than {self._max_body} bytes")
        content_type = headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            return _error(400, "expected multipart/form-data")
        body = await asyncio.wait_for(reader.readexactly(length), 120)
        fields, file = parse_multipart(content_type, body)
        if file is None or not file[0]:
            return _error(400, "missing or empty 'file' field")
        response_format = fields.get("response_format", "json")
        if response_format not in ("json", "text"):
            return _error(400, "response_format must be 'json' or 'text'")
        audio, mime_type = file
        transcript = await self._engine.transcribe(
            audio, mime_type=mime_type, language=fields.get("language") or self._language
        )
        log.info(
            "transcribed %d bytes, stt %.2fs, %d chars, language=%s",
            len(audio), transcript.processing_s, len(transcript.text), transcript.language,
        )
        if response_format == "text":
            return 200, transcript.text.encode(), "text/plain; charset=utf-8"
        return _json(200, {"text": transcript.text})


async def serve(settings: Settings) -> None:
    options = dict(settings.server)
    key_env = options.pop("api_key_env", None)
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise ValueError(f"environment variable {key_env} is not set")
    preload = options.pop("preload", True)
    engine = build_stt(settings.stt)
    if preload and hasattr(engine, "preload"):
        log.info("loading model")
        await engine.preload()
        log.info("model ready")
    server = TranscriptionServer(engine, api_key=api_key, language=settings.language, **options)
    if not api_key and server._host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("listening on %s without an API key: anyone who can reach the port can use the engine", server._host)
    await server.start()
    try:
        await server.serve_forever()
    finally:
        await server.close()
